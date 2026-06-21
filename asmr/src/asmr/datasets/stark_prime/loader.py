"""STaRK-Prime data loading: corpus builder and query loader."""

import ast
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

# mFAR schema (microsoft/multifield-adaptive-retrieval mfar/data/schema.py)
PRIME_FIELD_NAMES: tuple[str, ...] = (
    "associated with",
    "carrier",
    "contraindication",
    "details",
    "enzyme",
    "expression absent",
    "expression present",
    "indication",
    "interacts with",
    "linked to",
    "name",
    "off-label use",
    "parent-child",
    "phenotype absent",
    "phenotype present",
    "ppi",
    "side effect",
    "source",
    "synergistic interaction",
    "target",
    "transporter",
    "type",
)

PRIME_EDGE_FIELDS: dict[str, str] = {
    "ppi": "name",
    "carrier": "name",
    "enzyme": "name",
    "target": "name",
    "transporter": "name",
    "contraindication": "name",
    "indication": "name",
    "off-label use": "name",
    "synergistic interaction": "name",
    "associated with": "name",
    "parent-child": "name",
    "phenotype absent": "name",
    "phenotype present": "name",
    "side effect": "name",
    "interacts with": "name",
    "linked to": "name",
    "expression present": "name",
    "expression absent": "name",
}


@dataclass(frozen=True)
class PrimeQuery:
    """One STaRK-Prime QA row."""

    query_id: int
    query: str
    answer_ids: list[str]


@dataclass(frozen=True)
class PrimeCorpus:
    """Candidate documents with mFAR multi-field JSON payloads."""

    doc_ids: list[str]
    documents: list[dict[str, Any]]


def _format_details(details: dict[str, Any], node_type: str) -> str:
    if not details:
        return ""
    gene_keys = {
        "name": "gene name",
        "type_of_gene": "gene types",
        "alias": "other gene names",
        "other_names": "extended other gene names",
        "genomic_pos": "genomic position",
        "generif": "PubMed text",
        "interpro": "protein family and classification information",
        "summary": "protein summary text",
    }
    parts: list[str] = []
    for key, value in details.items():
        if str(value) in {"", "nan"} or key.startswith("_") or "_id" in key:
            continue
        label = gene_keys.get(key, key) if node_type == "gene/protein" else key
        if isinstance(value, (list, dict)):
            parts.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
        else:
            parts.append(f"{label}: {value}")
    return "; ".join(parts)


def _neighbor_names(
    neighbors: list[int],
    node_info: dict[int, dict[str, Any]],
    attr: str,
) -> list[str]:
    names: list[str] = []
    for n in neighbors:
        info = node_info.get(n)
        if not info:
            continue
        val = info.get(attr, "")
        if val in (-1, "-1", "", None):
            continue
        names.append(str(val))
    return names


def build_prime_document(
    node_pos: int,
    node_info: dict[int, dict[str, Any]],
    node_type_dict: dict[int, str],
    out_neighbors: dict[int, dict[str, dict[int, list[int]]]],
) -> dict[str, Any]:
    """Build mFAR-compatible Prime document for one corpus node."""
    base = node_info[node_pos]
    doc: dict[str, Any] = {
        "name": base.get("name", ""),
        "type": base.get("type", ""),
        "source": base.get("source", ""),
        "details": _format_details(base.get("details") or {}, base.get("type", "")),
    }

    edge_buckets = out_neighbors.get(node_pos, {})
    for edge_name in PRIME_EDGE_FIELDS:
        grouped = edge_buckets.get(edge_name)
        if not grouped:
            continue
        edge_value: dict[str, list[str]] = {}
        for type_id, neighbor_positions in grouped.items():
            type_name = node_type_dict.get(type_id, str(type_id))
            names = _neighbor_names(neighbor_positions, node_info, "name")
            if names:
                edge_value[type_name] = names
        if edge_value:
            doc[edge_name] = edge_value

    return doc


def field_text(document: dict[str, Any], field_name: str) -> str:
    """Serialize one logical field to indexable text."""
    if field_name not in document:
        return ""
    value = document[field_name]
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, items in value.items():
            if isinstance(items, list):
                parts.append(f"{key}: {', '.join(str(x) for x in items)}")
            else:
                parts.append(f"{key}: {items}")
        return "; ".join(parts)
    return str(value)


def single_field_text(document: dict[str, Any]) -> str:
    """Concatenate all fields (mFAR ``single`` / MFAR2 baseline)."""
    chunks = [field_text(document, f) for f in PRIME_FIELD_NAMES]
    return "\n".join(c for c in chunks if c)


def load_prime_skb(
    data_root: Path,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, str],
    dict[int, dict[str, dict[int, list[int]]]],
]:
    """Load processed Prime SKB tensors and adjacency grouped by edge type."""
    proc = data_root / "skb" / "processed"
    with (proc / "node_info.pkl").open("rb") as f:
        node_info: dict[int, dict[str, Any]] = pickle.load(f)
    with (proc / "node_type_dict.pkl").open("rb") as f:
        node_type_dict: dict[int, str] = pickle.load(f)
    with (proc / "edge_type_dict.pkl").open("rb") as f:
        edge_type_dict: dict[int, str] = pickle.load(f)

    edge_index = torch.load(proc / "edge_index.pt", weights_only=True)
    edge_types = torch.load(proc / "edge_types.pt", weights_only=True)
    node_types = torch.load(proc / "node_types.pt", weights_only=True)

    out_neighbors: dict[int, dict[str, dict[int, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        tgt = int(edge_index[1, i])
        et_name = edge_type_dict[int(edge_types[i])]
        if et_name not in PRIME_EDGE_FIELDS:
            continue
        tgt_type = int(node_types[tgt])
        out_neighbors[src][et_name][tgt_type].append(tgt)

    return node_info, node_type_dict, dict(out_neighbors)


def build_prime_corpus(data_root: Path, max_docs: int = -1) -> PrimeCorpus:
    """Materialize all candidate Prime documents."""
    node_info, node_type_dict, out_neighbors = load_prime_skb(data_root)
    positions = sorted(node_info.keys())
    if max_docs > 0:
        positions = positions[:max_docs]

    documents: list[dict[str, Any]] = []
    doc_ids: list[str] = []
    for pos in positions:
        doc = build_prime_document(pos, node_info, node_type_dict, out_neighbors)
        doc_ids.append(str(node_info[pos]["id"]))
        documents.append(doc)
    return PrimeCorpus(doc_ids=doc_ids, documents=documents)


def _answer_positions_to_node_ids(
    answer_positions: list[int],
    node_info: dict[int, dict[str, Any]],
) -> list[str]:
    node_ids: list[str] = []
    for pos in answer_positions:
        key = int(pos)
        info = node_info.get(key)
        if info is None:
            continue
        node_ids.append(str(info["id"]))
    return node_ids


def queries_to_qrels(queries: list[PrimeQuery]) -> dict[int, dict[str, int]]:
    """Convert PrimeQuery list to TREC-style qrels dict {query_id: {doc_id: 1}}."""
    return {q.query_id: {aid: 1 for aid in q.answer_ids} for q in queries}


def load_prime_queries(data_root: Path, split: str) -> list[PrimeQuery]:
    """Load QA rows for ``train`` / ``val`` / ``test`` split.

    ``answer_ids`` in the CSV are **node positions** in the Prime SKB;
    they are converted to corpus **node ids** (matching ``PrimeCorpus``).
    """
    split_path = data_root / "qa" / "prime" / "split" / f"{split}.index"
    csv_path = data_root / "qa" / "prime" / "stark_qa" / "stark_qa.csv"
    indices = [int(x.strip()) for x in split_path.read_text().splitlines() if x.strip()]
    df = pd.read_csv(csv_path)
    node_info, _, _ = load_prime_skb(data_root)
    rows: list[PrimeQuery] = []
    for idx in indices:
        row = df.iloc[idx]
        answer_positions = ast.literal_eval(str(row["answer_ids"]))
        rows.append(
            PrimeQuery(
                query_id=int(row["id"]),
                query=str(row["query"]),
                answer_ids=_answer_positions_to_node_ids(
                    [int(a) for a in answer_positions],
                    node_info,
                ),
            )
        )
    return rows
