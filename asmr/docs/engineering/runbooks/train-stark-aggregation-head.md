# Runbook: STaRK 데이터셋으로 Aggregation Head 학습

> 관련 ADR: [001-head-only-training-phase1.md](../adr/001-head-only-training-phase1.md)  
> 관련 스펙: [mfar-trainable-retrieval.md](../specs/mfar-trainable-retrieval.md)

---

## 개요

Phase 1 학습: query encoder를 **frozen** 상태로 두고 `MFARFieldAdapter` (linear G) 또는 `AggregationHead` (MLP G_θ)만 학습한다.  
학습 entrypoint: `asmr/src/asmr/train/train_script.py` (`python -m asmr.train`)

---

## 사전 요건

| 항목 | 요구 사항 |
|------|-----------|
| Python | 3.13 |
| PyTorch | 2.6.0+cu124 |
| GPU | RTX 3090 Ti 권장 (CPU/MPS에서도 smoke 실행 가능) |
| 작업 디렉토리 | `asmr/` |

---

## Encoder 모델

| 용도 | 모델 ID | query_dim | 비고 |
|------|---------|-----------|------|
| **기본 (Phase 1)** | `facebook/contriever-msmarco` | **768** | STaRK 논문과 동일 백본. HuggingFace 자동 다운로드 |
| 대체 | `BAAI/bge-m3` | 1024 | dense 채널만 사용 시. `--encoder BAAI/bge-m3` 지정 |

`HfQueryEncoder`는 `@torch.no_grad()`로 동작 — Phase 1에서 파라미터 업데이트 없음.  
인코딩 방식: attention mask 가중 mean-pool + L2 normalize.

---

## Step 1 — 환경 확인

```bash
cd asmr/

# PyTorch + CUDA 확인
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# smoke (HF 다운로드 없이 dummy 텐서)
uv run python -m asmr.train --dummy --epochs 2
```

정상 출력 예시:
```
2.6.0+cu124 True
```

---

## Step 2 — STaRK 데이터 준비

STaRK 공식 GitHub: [snap-stanford/stark](https://github.com/snap-stanford/stark)  
mFAR 전처리 스크립트: [microsoft/multifield-adaptive-retrieval](https://github.com/microsoft/multifield-adaptive-retrieval)

### 2-1. 데이터 형식

`StarkRankingDataset`은 `StarkRankingExample` 리스트를 인메모리로 보유한다.  
JSONL 로더는 **현재 미구현** (P0 작업) — 직접 파싱 후 아래 구조로 변환한다.

```python
from asmr.train.data_stark import StarkRankingExample
import numpy as np

example = StarkRankingExample(
    query_text="What products are rated 5 stars and made by Sony?",
    doc_ids=["doc_001", "doc_002", ...],          # length D
    scores=np.zeros((F, M, D), dtype=np.float32), # F=필드수, M=2(lex/dense), D=shortlist 크기
    relevance=np.array([1.0, 0.0, ...], dtype=np.float32),  # length D
    field_mask=np.ones((F, D), dtype=bool),        # optional
)
```

### 2-2. scores 행렬 채우기

`scores[f, 0, d]` = 필드 f에 대한 lexical(BM25) 점수  
`scores[f, 1, d]` = 필드 f에 대한 dense 점수  
필드가 없는 문서 → `scores[f, :, d] = 0.0`, `field_mask[f, d] = False`

---

## Step 3 — 학습 실행

### 3-1. MFARFieldAdapter (linear G, 권장 시작점)

```bash
uv run python -m asmr.train \
    --encoder facebook/contriever-msmarco \
    --epochs 20 \
    --lr 1e-3 \
    --checkpoint ckpt/mfar_linear.pt
```

### 3-2. AggregationHead (MLP G_θ)

현재 `train_script.py`는 `MFARFieldAdapter`만 기본 사용한다.  
`AggregationHead`로 전환하려면 `train_script.py`에서 모델 선택 분기를 추가해야 한다.

```python
# train_script.py 수정 예시
from asmr.train.aggregation import AggregationHead
from asmr.train.config import TrainConfig

cfg = TrainConfig(query_dim=768, hidden_dim=256, aux_dim=9)
model = AggregationHead(
    query_dim=cfg.query_dim,
    num_fields=f_num,
    num_scorers=m_num,
    hidden_dim=cfg.hidden_dim,
    aux_dim=cfg.aux_dim,
    dropout=cfg.dropout,
).to(device)
```

### 3-3. 주요 하이퍼파라미터 (`TrainConfig`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `query_dim` | 768 | encoder 출력 차원. contriever=768, bge-m3=1024 |
| `hidden_dim` | 256 | AggregationHead MLP hidden size |
| `temperature` | 0.07 | listwise softmax CE 온도 |
| `lambda_rank` | 1.0 | ranking loss 가중치 |
| `normalize_scores` | False | True 시 (field,scorer)별 batch normalization 적용 |
| `use_aux_features` | True | AggregationHead에서 build_aux_features 사용 여부 |
| `aux_dim` | 9 | aux 특성 수 (변경 금지) |

---

## Step 4 — 학습 루프 구조 확인

```
RankingBatch (scores [B,F,M,D], query_emb [B,H], relevance [B,D])
    ↓ normalize_scores_per_field_scorer (normalize_scores=True 시)
    ↓ build_aux_features → aux [B,D,9]   (AggregationHead 사용 시)
    ↓ MFARFieldAdapter / AggregationHead
logits [B,D]
    ↓ listwise_logit_loss(logits, relevance, τ)
loss
    ↓ loss.backward() + AdamW.step()
```

손실 함수: `listwise_logit_loss` (softmax cross-entropy, soft target)  
현재 배치 크기: **B=1** (shortlist 크기 D가 예시마다 다름 — 패딩 미구현)

---

## Step 5 — 체크포인트 저장 및 로드

```python
import torch
from asmr.train.aggregation import MFARFieldAdapter
from asmr.train.config import TrainConfig

# 저장
torch.save({"model": model.state_dict(), "config": cfg}, "ckpt/mfar_linear.pt")

# 로드
ckpt = torch.load("ckpt/mfar_linear.pt", map_location="cpu")
cfg: TrainConfig = ckpt["config"]
model = MFARFieldAdapter(cfg.query_dim, f_num, m_num)
model.load_state_dict(ckpt["model"])
```

---

## Step 6 — 추론 (shortlist reranking)

```python
from asmr.train.inference import apply_aggregation_head

sorted_ids, logits = apply_aggregation_head(
    doc_ids,      # list[str], length D
    scores,       # np.ndarray [F, M, D]
    field_mask,   # np.ndarray [F, D]
    query_emb,    # np.ndarray [H]  ← encoder.encode(["query"])[0].numpy()
    model,        # MFARFieldAdapter or AggregationHead (eval mode 자동 설정)
)
# sorted_ids: D개 문서 ID (logit 내림차순)
# logits: np.ndarray [D]
```

---

## Step 7 — 평가

```bash
# runs.jsonl 형식: {"query_id": "q1", "doc_ids": ["d1", "d2", ...]}
# qrels.json 형식: {"q1": ["d1", "d3"], ...}

uv run python -m asmr.evaluation.stark_eval \
    --runs results/runs.jsonl \
    --qrels data/stark/qrels.json
```

출력 지표: Hit@1, Recall@20, MRR

Python API:

```python
from asmr.evaluation.metrics import hit_at_k, recall_at_k, mean_reciprocal_rank

hit  = hit_at_k(ranked_ids, relevant_set, k=1)
rec  = recall_at_k(ranked_ids, relevant_set, k=20)
mrr  = mean_reciprocal_rank(ranked_ids, relevant_set)
```

---

## GPU 메모리 추정 및 실행 가능성

> 측정 환경: RTX 3090 Ti (24 GB VRAM), PyTorch 2.6.0+cu124, B=1, D=100, F=2, M=2  
> 측정 방법: `torch.cuda.max_memory_allocated()` (학습 1 step, `no_grad` encoder)

### 모델별 파라미터

| 컴포넌트 | 파라미터 수 | fp32 크기 |
|----------|------------|-----------|
| `facebook/contriever-msmarco` (BERT-base) | 110M | ~419 MB |
| `MFARFieldAdapter` (Linear G) | 3,072 | ~12 KB |
| `AggregationHead` (MLP G_θ, hidden=256) | 200,450 | ~783 KB |
| `BAAI/bge-m3` (XLM-RoBERTa-large 기반) | ~560M | ~2,136 MB |

### 학습 1 step 실측 VRAM (Phase 1, encoder frozen)

| 구성 | Peak VRAM | 비고 |
|------|-----------|------|
| contriever + MFARFieldAdapter, B=1, D=100 | **435 MB (0.43 GB)** | 실측값 |
| contriever + AggregationHead, B=1, D=100 | **~437 MB (0.43 GB)** | head 차이 무시 수준 |
| bge-m3 + MFARFieldAdapter, B=1, D=100 | **~2,146 MB (2.1 GB)** | 추정값 (encoder 미다운로드) |

encoder는 `@torch.no_grad()`로 실행 → activation 메모리 극소 (실측 ~9 MB). 지배 항목은 encoder weight.

### RTX 3090 Ti (24 GB) 기준 실행 가능성

| 시나리오 | VRAM 사용 | 여유 | 실행 가능 |
|----------|-----------|------|-----------|
| contriever + MFARFieldAdapter, B=1 | ~0.43 GB | ~23.5 GB | **가능** |
| contriever + AggregationHead, B=1 | ~0.44 GB | ~23.5 GB | **가능** |
| bge-m3 + MFARFieldAdapter, B=1 | ~2.1 GB | ~21.9 GB | **가능** |
| Phase 2 (encoder fine-tuning, contriever) | ~1.5–2 GB+ | — | 가능 (그래디언트+옵티마이저 상태 추가) |

**결론: 현재 GPU(24 GB)에서 모든 Phase 1 학습 시나리오가 충분히 실행 가능하다.**  
contriever 기준 실제 VRAM 점유율은 전체의 약 1.8% 수준이며, shortlist D=1000으로 늘려도 score 텐서 크기는 수 MB 이내로 여유 충분.

### 실행 전 확인 명령

```bash
# 현재 VRAM 여유 확인
uv run python -c "
import torch
p = torch.cuda.get_device_properties(0)
free = p.total_memory - torch.cuda.memory_reserved(0)
print(f'{p.name}: free {free/1024**3:.2f} GB / {p.total_memory/1024**3:.2f} GB')
"

# smoke (HF 다운로드 없이, ~5초)
uv run python -m asmr.train --dummy --epochs 1

# 실제 encoder 사용 학습
uv run python -m asmr.train --encoder facebook/contriever-msmarco --epochs 5
```

---

## 알려진 제약 및 TODO

| 항목 | 상태 | 비고 |
|------|------|------|
| JSONL 데이터 로더 | **미구현** (P0) | `StarkRankingDataset`에 `from_jsonl(path)` 추가 필요 |
| B>1 배치 | 미지원 | shortlist 패딩(`D_max`) 도입 시 지원 가능 |
| Phase 2 (encoder fine-tuning) | 미착수 | ADR-001 Phase 2 진입 조건 충족 후 진행 |

---

## 품질 게이트

```bash
uv run pytest tests/unit/asmr/train/ -v
uv run python -m asmr.train --dummy --epochs 1
```
