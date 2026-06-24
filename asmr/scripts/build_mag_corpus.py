"""Build corpus.json for STaRK-MAG from raw graph data.

Reads:
  data/stark_mag/skb/mag/processed/  (node_info.pkl, edge_index.pt, edge_types.pt, ...)
  data/stark_mag/skb/mag/idx_title_abs.tsv  (from idx_title_abs.zip)

Writes:
  data/stark_mag/skb/mag/corpus.json
"""

import json
import os
import pickle
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "snap-stanford/stark"
REPO_TYPE = "dataset"

# Edge type IDs (from edge_type_dict.pkl)
ET_AUTHOR_AFFIL = 0  # author -> institution
ET_PAPER_CITES = 1  # paper -> paper
ET_PAPER_TOPIC = 2  # paper -> field_of_study
ET_AUTHOR_WRITES = 3  # author -> paper


def ensure_title_abs_tsv(data_root: Path) -> Path:
    tsv_path = data_root / "skb" / "mag" / "idx_title_abs.tsv"
    if tsv_path.exists():
        return tsv_path
    print("Downloading idx_title_abs.zip ...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, filename="skb/mag/idx_title_abs.zip"
    )
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("idx_title_abs.tsv") as src, tsv_path.open("wb") as dst:
            dst.write(src.read())
    return tsv_path


def load_title_abs(tsv_path: Path) -> dict[int, tuple[str, str]]:
    """Returns {mag_id: (title, abstract)}. TSV first column is mag_id."""
    result: dict[int, tuple[str, str]] = {}
    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            mag_id = int(parts[0])
            title = parts[1] if len(parts) > 1 else ""
            abstract = parts[2] if len(parts) > 2 else ""
            result[mag_id] = (title, abstract)
    return result


def build_corpus(data_root: Path) -> None:
    proc = data_root / "skb" / "mag" / "processed"
    corpus_path = data_root / "skb" / "mag" / "corpus.json"

    if corpus_path.exists():
        corpus_path.unlink()
        print("Removed old corpus.json (ID scheme was wrong).")

    t0 = time.time()

    # --- title/abstract: keyed by mag_id ---
    tsv_path = ensure_title_abs_tsv(data_root)
    print("Loading title/abstract TSV ...")
    title_abs_by_mag: dict[int, tuple[str, str]] = load_title_abs(tsv_path)
    print(f"  {len(title_abs_by_mag):,} papers in TSV  ({time.time()-t0:.0f}s)")

    # --- node_info: build mag_id -> new_id mapping for papers ---
    # QA answer_ids use new_id; TSV uses mag_id. We must key corpus by new_id.
    print("Loading node_info.pkl ...")
    with (proc / "node_info.pkl").open("rb") as f:
        node_info: dict[int, dict] = pickle.load(f)  # type: ignore[type-arg]
    print(f"  {len(node_info):,} nodes  ({time.time()-t0:.0f}s)")

    # new_id == internal dict key for all nodes (verified empirically)
    # Build: mag_id -> new_id (for papers only)
    mag_to_new: dict[int, int] = {}
    key_to_new_id: dict[int, int] = {}
    for k, v in node_info.items():
        new_id = v.get("new_id")
        if new_id is not None:
            key_to_new_id[k] = int(new_id)
        if v.get("type") == "paper":
            mag_id = v.get("mag_id")
            if mag_id is not None and new_id is not None:
                mag_to_new[int(mag_id)] = int(new_id)

    # Build title_abs keyed by new_id
    title_abs: dict[int, tuple[str, str]] = {}
    for mag_id, ta in title_abs_by_mag.items():
        new_id = mag_to_new.get(mag_id)
        if new_id is not None:
            title_abs[new_id] = ta
    print(f"  Mapped {len(title_abs):,} papers mag_id->new_id  ({time.time()-t0:.0f}s)")

    corpus_new_ids = set(title_abs.keys())
    new_id_to_key: dict[int, int] = {v: k for k, v in key_to_new_id.items() if v in corpus_new_ids}

    # Map internal key -> display name for all nodes
    def display_name(k: int) -> str:
        info = node_info.get(k, {})
        return str(
            info.get("DisplayName")
            or info.get("title")
            or info.get("name")
            or info.get("OriginalTitle")
            or ""
        )

    print(f"  Built id mappings  ({time.time()-t0:.0f}s)")

    # --- load edges ---
    print("Loading edge tensors (may take a minute) ...")
    edge_index = torch.load(proc / "edge_index.pt", weights_only=True)  # [2, E]
    edge_types = torch.load(proc / "edge_types.pt", weights_only=True)  # [E]
    print(f"  {edge_index.shape[1]:,} edges  ({time.time()-t0:.0f}s)")

    # --- build adjacency lists ---
    # paper_new_id -> [cited paper keys]
    paper_cites: dict[int, list[int]] = defaultdict(list)
    # paper_new_id -> [field keys]
    paper_topics: dict[int, list[int]] = defaultdict(list)
    # paper_new_id -> [author keys]
    paper_authors: dict[int, list[int]] = defaultdict(list)
    # author key -> [institution keys]
    author_affil: dict[int, list[int]] = defaultdict(list)

    print("Building adjacency lists ...")
    E = edge_index.shape[1]
    log_every = max(E // 20, 1_000_000)
    for i in range(E):
        et = int(edge_types[i])
        src = int(edge_index[0, i])
        tgt = int(edge_index[1, i])
        if et == ET_PAPER_CITES:
            src_nid = key_to_new_id.get(src)
            if src_nid and src_nid in corpus_new_ids:
                paper_cites[src_nid].append(tgt)
        elif et == ET_PAPER_TOPIC:
            src_nid = key_to_new_id.get(src)
            if src_nid and src_nid in corpus_new_ids:
                paper_topics[src_nid].append(tgt)
        elif et == ET_AUTHOR_WRITES:
            tgt_nid = key_to_new_id.get(tgt)
            if tgt_nid and tgt_nid in corpus_new_ids:
                paper_authors[tgt_nid].append(src)
        elif et == ET_AUTHOR_AFFIL:
            author_affil[src].append(tgt)
        if (i + 1) % log_every == 0:
            print(f"  edges {i+1:,}/{E:,}  ({time.time()-t0:.0f}s)")

    # Free edge tensors
    del edge_index, edge_types
    print(f"  Adjacency done  ({time.time()-t0:.0f}s)")

    # --- build corpus documents ---
    print("Building corpus.json ...")
    corpus: list[dict] = []  # type: ignore[type-arg]
    for nid, (title, abstract) in title_abs.items():
        # author -> {institution list}
        author_affil_text: dict[str, list[str]] = {}
        for auth_key in paper_authors.get(nid, []):
            auth_name = display_name(auth_key)
            if not auth_name:
                continue
            insts = [
                display_name(inst_k)
                for inst_k in author_affil.get(auth_key, [])
                if display_name(inst_k)
            ]
            if auth_name not in author_affil_text:
                author_affil_text[auth_name] = insts
            else:
                author_affil_text[auth_name].extend(insts)

        cites_titles = [
            display_name(k) for k in paper_cites.get(nid, []) if display_name(k)
        ]
        topic_names = [
            display_name(k) for k in paper_topics.get(nid, []) if display_name(k)
        ]

        corpus.append(
            {
                "id": nid,
                "title": title,
                "abstract": abstract,
                "author___affiliated_with___institution": author_affil_text,
                "paper___cites___paper": cites_titles,
                "paper___has_topic___field_of_study": topic_names,
            }
        )

    print(f"  {len(corpus):,} documents built  ({time.time()-t0:.0f}s)")

    print(f"Writing corpus.json to {corpus_path} ...")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    size_mb = corpus_path.stat().st_size / 1_048_576
    print(f"Done: {corpus_path} ({size_mb:.0f} MB)  total {time.time()-t0:.0f}s")


def main() -> None:
    data_root = Path(os.getenv("ASMR_DATA_ROOT", "data/stark_mag"))
    build_corpus(data_root)


if __name__ == "__main__":
    main()
