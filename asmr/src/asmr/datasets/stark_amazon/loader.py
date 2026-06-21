"""STaRK-Amazon data loading: corpus builder and query loader."""

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from asmr.datasets.stark_prime.loader import field_text as _field_text

# mFAR schema (microsoft/multifield-adaptive-retrieval mfar/data/schema.py)
AMAZON_FIELD_NAMES: tuple[str, ...] = (
    "also_buy",
    "also_view",
    "brand",
    "description",
    "feature",
    "qa",
    "review",
    "title",
)


class AmazonQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: int
    query: str
    answer_ids: list[str]


class AmazonCorpus(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_ids: list[str]
    documents: list[dict[str, Any]]


def field_text(document: dict[str, Any], field_name: str) -> str:
    """Serialize one Amazon document field to indexable text."""
    return _field_text(document, field_name)


def build_amazon_corpus(data_root: Path, max_docs: int = -1) -> AmazonCorpus:
    """Load Amazon corpus from JSON and return AmazonCorpus.

    Expects: data_root/skb/amazon/corpus.json
    Format: list of dicts with an "id" key plus Amazon field keys.
    """
    corpus_path = data_root / "skb" / "amazon" / "corpus.json"
    with corpus_path.open("r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    if max_docs > 0:
        raw = raw[:max_docs]

    doc_ids: list[str] = []
    documents: list[dict[str, Any]] = []
    for item in raw:
        doc_ids.append(str(item["id"]))
        doc = {field: item.get(field, "") for field in AMAZON_FIELD_NAMES}
        documents.append(doc)

    return AmazonCorpus(doc_ids=doc_ids, documents=documents)


def load_amazon_queries(data_root: Path, split: str) -> list[AmazonQuery]:
    """Load QA rows for train / val / test split.

    Expects:
      data_root/qa/amazon/split/{split}.index
      data_root/qa/amazon/stark_qa/stark_qa.csv
    """
    split_path = data_root / "qa" / "amazon" / "split" / f"{split}.index"
    csv_path = data_root / "qa" / "amazon" / "stark_qa" / "stark_qa.csv"
    indices = [int(x.strip()) for x in split_path.read_text().splitlines() if x.strip()]
    df = pd.read_csv(csv_path)
    rows: list[AmazonQuery] = []
    for idx in indices:
        row = df.iloc[idx]
        answer_ids = [str(a) for a in ast.literal_eval(str(row["answer_ids"]))]
        rows.append(
            AmazonQuery(
                query_id=int(row["id"]),
                query=str(row["query"]),
                answer_ids=answer_ids,
            )
        )
    return rows
