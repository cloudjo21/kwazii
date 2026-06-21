"""STaRK-MAG data loading: corpus builder and query loader."""

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from asmr.datasets.stark_prime.loader import field_text as _field_text

# mFAR schema (microsoft/multifield-adaptive-retrieval mfar/data/schema.py)
MAG_FIELD_NAMES: tuple[str, ...] = (
    "abstract",
    "author___affiliated_with___institution",
    "paper___cites___paper",
    "paper___has_topic___field_of_study",
    "title",
)


class MagQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: int
    query: str
    answer_ids: list[str]


class MagCorpus(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_ids: list[str]
    documents: list[dict[str, Any]]


def field_text(document: dict[str, Any], field_name: str) -> str:
    """Serialize one MAG document field to indexable text."""
    return _field_text(document, field_name)


def single_field_text(document: dict[str, Any]) -> str:
    """Concatenate all MAG fields (mFAR single / MFAR2 baseline)."""
    chunks = [field_text(document, f) for f in MAG_FIELD_NAMES]
    return "\n".join(c for c in chunks if c)


def build_mag_corpus(data_root: Path, max_docs: int = -1) -> MagCorpus:
    """Load MAG corpus from JSON and return MagCorpus.

    Expects: data_root/skb/mag/corpus.json
    Format: list of dicts with an "id" key plus MAG field keys.
    """
    corpus_path = data_root / "skb" / "mag" / "corpus.json"
    with corpus_path.open("r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    if max_docs > 0:
        raw = raw[:max_docs]

    doc_ids: list[str] = []
    documents: list[dict[str, Any]] = []
    for item in raw:
        doc_ids.append(str(item["id"]))
        doc = {field: item.get(field, "") for field in MAG_FIELD_NAMES}
        documents.append(doc)

    return MagCorpus(doc_ids=doc_ids, documents=documents)


def load_mag_queries(data_root: Path, split: str) -> list[MagQuery]:
    """Load QA rows for train / val / test split.

    Expects:
      data_root/qa/mag/split/{split}.index
      data_root/qa/mag/stark_qa/stark_qa.csv
    """
    split_path = data_root / "qa" / "mag" / "split" / f"{split}.index"
    csv_path = data_root / "qa" / "mag" / "stark_qa" / "stark_qa.csv"
    indices = [int(x.strip()) for x in split_path.read_text().splitlines() if x.strip()]
    df = pd.read_csv(csv_path)
    rows: list[MagQuery] = []
    for idx in indices:
        row = df.iloc[idx]
        answer_ids = [str(a) for a in ast.literal_eval(str(row["answer_ids"]))]
        rows.append(
            MagQuery(
                query_id=int(row["id"]),
                query=str(row["query"]),
                answer_ids=answer_ids,
            )
        )
    return rows
