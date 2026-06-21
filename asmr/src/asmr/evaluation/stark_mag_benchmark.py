"""STaRK-MAG MFARAll benchmark runner (hybrid multi-field retrieval)."""

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from torch import optim

from asmr.datasets.stark_mag.loader import (
    MAG_FIELD_NAMES,
    MagCorpus,
    MagQuery,
    build_mag_corpus,
    field_text,
    load_mag_queries,
    single_field_text,
)
from asmr.datasets.stark_prime.torch_dataset import (
    StarkRankingDataset,
    StarkRankingExample,
    collate_stark_batch,
)
from asmr.evaluation.metrics import hit_at_k, mean_reciprocal_rank, recall_at_k
from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig
from asmr.train.inference import apply_aggregation_head
from asmr.train.query_encoder import HfQueryEncoder
from asmr.train.trainer import AggregationTrainer

logger = logging.getLogger(__name__)

# mFAR paper Table 1 — STaRK-MAG test-set numbers (MFARAll+2 best result)
MFAR_MAG_TEST = {
    "BM25": {"hit@1": 0.340, "recall@20": 0.609, "mrr": 0.430},
    "Contriever-FT": {"hit@1": 0.470, "recall@20": 0.695, "mrr": 0.556},
    "MFARAll": {"hit@1": 0.559, "recall@20": 0.741, "mrr": 0.643},
}


@dataclass
class BenchmarkMetrics:
    hit_at_1: float
    recall_at_20: float
    mrr: float
    num_queries: int
    elapsed_sec: float


@dataclass
class BenchmarkReport:
    dataset: str
    protocol: str
    encoder: str
    shortlist_k: int
    train_epochs: int
    metrics: BenchmarkMetrics
    mfar_paper_test: dict[str, dict[str, float]]
    ratio_vs_mfar_mfarall: dict[str, float]
    notes: list[str]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class MagFieldIndexes:
    """Per-field lexical BM25 and dense embedding matrices for STaRK-MAG."""

    def __init__(
        self,
        corpus: MagCorpus,
        encoder: HfQueryEncoder,
        batch_size: int = 64,
    ) -> None:
        self.doc_ids = corpus.doc_ids
        self.id_to_col = {d: i for i, d in enumerate(self.doc_ids)}
        self.num_docs = len(self.doc_ids)
        self.field_names = list(MAG_FIELD_NAMES)
        self.field_mask = np.zeros((len(self.field_names), self.num_docs), dtype=bool)

        self.bm25_models: list[BM25Okapi | None] = []
        self.dense_embeddings: list[np.ndarray | None] = []

        for fi, fname in enumerate(self.field_names):
            texts = [field_text(doc, fname) for doc in corpus.documents]
            nonempty = [bool(t.strip()) for t in texts]
            self.field_mask[fi] = np.array(nonempty, dtype=bool)

            tokenized = [_tokenize(t) if t.strip() else [""] for t in texts]
            if any(nonempty):
                self.bm25_models.append(BM25Okapi(tokenized))
            else:
                self.bm25_models.append(None)

            nonempty_texts = [t if t.strip() else fname for t in texts]
            emb = encoder.encode(nonempty_texts, batch_size=batch_size).numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.dense_embeddings.append(
                (emb / np.maximum(norms, 1e-12)).astype(np.float32)
            )

        single_texts = [single_field_text(doc) for doc in corpus.documents]
        self.single_bm25 = BM25Okapi([_tokenize(t) for t in single_texts])
        single_emb = encoder.encode(single_texts, batch_size=batch_size).numpy()
        norms = np.linalg.norm(single_emb, axis=1, keepdims=True)
        self.single_dense = (single_emb / np.maximum(norms, 1e-12)).astype(np.float32)

    def _topk(self, scores: np.ndarray, k: int) -> tuple[list[str], np.ndarray]:
        k = min(k, scores.shape[0])
        if k <= 0:
            return [], np.zeros(0, dtype=np.float32)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self.doc_ids[i] for i in top_idx], scores[top_idx].astype(np.float32)

    def shortlist_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Build MFARAll shortlist: union of per-field top-k (lex + dense)."""
        f_num = len(self.field_names)
        union: dict[str, np.ndarray] = {}

        q_tok = _tokenize(query_text)
        q_emb = query_emb.astype(np.float32)

        for fi, fname in enumerate(self.field_names):
            if self.bm25_models[fi] is not None:
                lex = np.array(self.bm25_models[fi].get_scores(q_tok), dtype=np.float32)
                ids, sc = self._topk(lex, k)
                for doc_id, score in zip(ids, sc):
                    union.setdefault(doc_id, np.zeros((f_num, 2), dtype=np.float32))
                    union[doc_id][fi, 0] = max(union[doc_id][fi, 0], score)

            dense = self.dense_embeddings[fi]
            if dense is not None:
                den = dense @ q_emb
                ids, sc = self._topk(den, k)
                for doc_id, score in zip(ids, sc):
                    union.setdefault(doc_id, np.zeros((f_num, 2), dtype=np.float32))
                    union[doc_id][fi, 1] = max(union[doc_id][fi, 1], score)

        if not union:
            return (
                [],
                np.zeros((f_num, 2, 0), dtype=np.float32),
                np.zeros((f_num, 0), dtype=bool),
            )

        doc_ids = sorted(union.keys())
        d_num = len(doc_ids)
        scores = np.zeros((f_num, 2, d_num), dtype=np.float32)
        mask = np.zeros((f_num, d_num), dtype=bool)
        for di, doc_id in enumerate(doc_ids):
            scores[:, :, di] = union[doc_id]
            col = self.id_to_col[doc_id]
            mask[:, di] = self.field_mask[:, col]
        return doc_ids, scores, mask

    def rank_single_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> list[str]:
        """MFAR2-style single-field hybrid baseline."""
        lex = np.array(
            self.single_bm25.get_scores(_tokenize(query_text)), dtype=np.float32
        )
        den = self.single_dense @ query_emb.astype(np.float32)
        hybrid = lex + den
        ids, _ = self._topk(hybrid, k)
        return ids


def _example_from_shortlist(
    query: MagQuery,
    doc_ids: list[str],
    scores: np.ndarray,
    mask: np.ndarray,
) -> StarkRankingExample:
    rel = np.zeros(len(doc_ids), dtype=np.float32)
    ans = {str(a) for a in query.answer_ids}
    for i, doc_id in enumerate(doc_ids):
        if doc_id in ans:
            rel[i] = 1.0
    return StarkRankingExample(
        query_text=query.query,
        doc_ids=doc_ids,
        scores=scores,
        relevance=rel,
        field_mask=mask,
    )


def train_mag_adapter(
    train_queries: list[MagQuery],
    indexes: MagFieldIndexes,
    encoder: HfQueryEncoder,
    *,
    shortlist_k: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> MFARFieldAdapter:
    f_num = len(MAG_FIELD_NAMES)
    model = MFARFieldAdapter(encoder.embedding_dim, f_num, 2).to(device)
    cfg = TrainConfig(query_dim=encoder.embedding_dim)
    trainer = AggregationTrainer(model, cfg)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    examples: list[StarkRankingExample] = []
    for q in train_queries:
        q_emb = encoder.encode([q.query])[0].numpy()
        doc_ids, scores, mask = indexes.shortlist_hybrid(q.query, q_emb, shortlist_k)
        if not doc_ids:
            continue
        examples.append(_example_from_shortlist(q, doc_ids, scores, mask))

    ds = StarkRankingDataset(examples)
    indices = list(range(len(ds)))
    logger.info("Training MFARFieldAdapter on %d MAG shortlists", len(ds))

    for epoch in range(1, epochs + 1):
        random.shuffle(indices)
        loss_sum = 0.0
        for idx in indices:
            batch = collate_stark_batch([ds[idx]], encoder, device)
            opt.zero_grad()
            out = trainer.training_step(batch)
            out.loss.backward()
            opt.step()
            loss_sum += out.loss.item()
        logger.info(
            "Epoch %d/%d loss=%.4f", epoch, epochs, loss_sum / max(len(indices), 1)
        )
    return model


def evaluate_ranked(
    queries: list[MagQuery],
    rank_fn,
    *,
    eval_k: int = 20,
) -> BenchmarkMetrics:
    t0 = time.time()
    hits, recalls, mrrs = [], [], []
    for q in queries:
        ranked = rank_fn(q)
        rel = {str(a) for a in q.answer_ids}
        hits.append(hit_at_k(ranked, rel, 1))
        recalls.append(recall_at_k(ranked, rel, 20))
        mrrs.append(mean_reciprocal_rank(ranked, rel))
    n = max(len(hits), 1)
    return BenchmarkMetrics(
        hit_at_1=sum(hits) / n,
        recall_at_20=sum(recalls) / n,
        mrr=sum(mrrs) / n,
        num_queries=len(queries),
        elapsed_sec=time.time() - t0,
    )


def run_benchmark(
    data_root: Path,
    *,
    encoder_name: str = "facebook/contriever-msmarco",
    shortlist_k: int = 100,
    train_epochs: int = 5,
    max_docs: int = -1,
    max_train_queries: int = -1,
    max_eval_queries: int = -1,
    eval_k: int = 20,
    seed: int = 42,
) -> BenchmarkReport:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    corpus = build_mag_corpus(data_root, max_docs=max_docs)
    logger.info(
        "MAG corpus: %d docs, %d fields", len(corpus.doc_ids), len(MAG_FIELD_NAMES)
    )

    encoder = HfQueryEncoder(model_name=encoder_name, device=str(device))
    indexes = MagFieldIndexes(corpus, encoder)

    train_queries = load_mag_queries(data_root, "train")
    test_queries = load_mag_queries(data_root, "test")
    if max_train_queries > 0:
        train_queries = train_queries[:max_train_queries]
    if max_eval_queries > 0:
        test_queries = test_queries[:max_eval_queries]

    adapter = train_mag_adapter(
        train_queries,
        indexes,
        encoder,
        shortlist_k=shortlist_k,
        epochs=train_epochs,
        lr=1e-3,
        device=device,
    )

    def rank_mfar_all(q: MagQuery) -> list[str]:
        q_emb = encoder.encode([q.query])[0].numpy()
        doc_ids, scores, mask = indexes.shortlist_hybrid(q.query, q_emb, shortlist_k)
        if not doc_ids:
            return []
        sorted_ids, _ = apply_aggregation_head(
            doc_ids,
            scores,
            mask,
            q_emb,
            adapter,
            device=device,
        )
        return [str(x) for x in sorted_ids[:eval_k]]

    metrics = evaluate_ranked(test_queries, rank_mfar_all, eval_k=eval_k)

    paper = MFAR_MAG_TEST["MFARAll"]
    ratio = {
        "hit@1": metrics.hit_at_1 / paper["hit@1"] if paper["hit@1"] else 0.0,
        "recall@20": metrics.recall_at_20 / paper["recall@20"]
        if paper["recall@20"]
        else 0.0,
        "mrr": metrics.mrr / paper["mrr"] if paper["mrr"] else 0.0,
    }

    notes = [
        "Encoder is frozen contriever-msmarco (not STaRK-finetuned Contriever-FT).",
        "Phase 1 head-only training; mFAR paper jointly fine-tunes encoder + G.",
        f"Shortlist k={shortlist_k} per field×scorer; eval reports top-{eval_k}.",
        "MAG has 5 fields — abstract, author affiliation, cites, topic, title.",
    ]
    if max_docs > 0 or max_eval_queries > 0 or max_train_queries > 0:
        notes.append(
            "Partial run: max_docs/max_train_queries/max_eval_queries limits applied."
        )

    return BenchmarkReport(
        dataset="STaRK-MAG",
        protocol="MFARAll",
        encoder=encoder_name,
        shortlist_k=shortlist_k,
        train_epochs=train_epochs,
        metrics=metrics,
        mfar_paper_test=MFAR_MAG_TEST,
        ratio_vs_mfar_mfarall=ratio,
        notes=notes,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="STaRK-MAG MFARAll benchmark")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("asmr/data/stark_mag"),
    )
    parser.add_argument("--encoder", default="facebook/contriever-msmarco")
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--max-train-queries", type=int, default=-1)
    parser.add_argument("--max-eval-queries", type=int, default=-1)
    parser.add_argument("--eval-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_benchmark(
        args.data_root,
        encoder_name=args.encoder,
        shortlist_k=args.shortlist_k,
        train_epochs=args.train_epochs,
        max_docs=args.max_docs,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        eval_k=args.eval_k,
    )

    payload = {
        "dataset": report.dataset,
        "protocol": report.protocol,
        "encoder": report.encoder,
        "shortlist_k": report.shortlist_k,
        "train_epochs": report.train_epochs,
        "metrics": asdict(report.metrics),
        "mfar_paper_test": report.mfar_paper_test,
        "ratio_vs_mfar_mfarall": report.ratio_vs_mfar_mfarall,
        "notes": report.notes,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
