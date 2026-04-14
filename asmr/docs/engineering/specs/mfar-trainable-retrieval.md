# mFAR Trainable Retrieval — Engineering Spec

> 상위 설계 문서: [`trainable-retrieval-design.md`](../../trainable-retrieval-design.md)  
> 코드 레벨 기술 상세: [`trainable-retrieval-technical-detail-design.md`](../../trainable-retrieval-technical-detail-design.md)  
> ADR: [`adr/001-head-only-training-phase1.md`](../adr/001-head-only-training-phase1.md)

---

## 1. 목적과 범위

### 1.1 무엇을 만드는가

asmr의 기존 필드별 검색 파이프라인(`QueryRouter` → `aggregate_field_scores_async`) 위에  
**학습 가능한 composition head(`G` / `G_θ`)** 를 얹는다.

핵심 아이디어: "더 나은 범용 임베딩"이 아니라, **정답이 어떤 필드 증거 조합으로 성립하는가** 를 학습한다.

### 1.2 mFAR 논문 대비 구현 범위

> **mFAR 논문(arXiv:2410.20056)의 실제 학습 설정**: query encoder와 composition head(`G`)를 **jointly** end-to-end로 학습한다. 즉 논문의 full baseline = **Phase 2**(encoder fine-tuning 포함)이다.

이 스펙이 커버하는 **Phase 1**은 encoder를 frozen으로 두고 **head만 학습**하는 단계다. Phase 1은 논문의 full baseline 재현이 아니라, 빠른 이터레이션과 ablation을 위한 **선행 단계**이다. Phase 2 채택 근거는 [ADR-001](../adr/001-head-only-training-phase1.md) 참조.

| | Phase 1 (이 스펙) | Phase 2 (추후) |
|---|---|---|
| 학습 대상 | head (`G` / `G_θ`) | encoder + head |
| 손실 | `L_rank` | `L_rank` + `L_c` + `L_b` |
| 논문 대응 | No query-conditioning ablation 재현 가능 | 논문 main result 재현 |

### 1.3 무엇을 만들지 않는가 (Non-goals)

- BGE-M3 규모의 통합 임베딩 사전학습 재현
- 전 코퍼스에 무거운 cross-encoder 1차 검색
- MIRACL/MKQA 전 벤치 재현 (teacher 회귀 스모크만 고려)

---

## 2. 전체 아키텍처

```
[IndexPerField]          [ShortlistPerFieldAndScorer]     [ScoreMatrix]    [AggregationHead]
  field_f1_index  ──→  topk_lex_f1                   ↘
  field_f2_index  ──→  topk_dense_f1  ──→ union_doc_ids → scores[F,M,D] → G_θ → doc_scores[D]
  field_fN_index  ──→  ...
                                                                             ↓
                                                                       (train) Loss
                                                                       (infer) ListwiseRank
```

| 단계 | 역할 | asmr 코드 |
|------|------|-----------|
| A | 필드별 인덱스 구축 | `index/fields.py`, `index/bm25.py`, dense 인덱스 |
| B | 필드·스코어러(lex/dense)별 top-k 후보 | `retrieve/retrievers.py` — `QueryRouter.retrieve(field, q, k)` |
| C | 후보 합집합 + `[F, M, D]` 점수 행렬 | `retrieve/aggregate.py` — `aggregate_field_scores_hybrid_async` |
| D | 문서 점수 `s(q,d)` = `G_θ(·)` | `train/aggregation.py` — `MFARFieldAdapter` / `AggregationHead` |
| E | (학습) ranking + 보조 손실 | `train/losses.py` |

---

## 3. 텐서 계약

### 3.1 핵심 텐서

| 이름 | shape | dtype | 설명 |
|------|-------|-------|------|
| `scores` | `[B, F, M, D]` | float32 | 필드×스코어러×문서 유사도. 필드 없으면 0 + mask |
| `field_mask` | `[B, F, D]` | bool | 문서 `d`가 필드 `f`를 가지면 True |
| `query_emb` | `[B, H]` | float32 | 질의 임베딩 (frozen encoder 출력) |
| `aux` | `[B, D, 9]` | float32 | `build_aux_features` 출력 (§5 참고) |
| `relevance` | `[B, D]` | float32 | 1.0=관련 / 0.0=비관련 |

### 3.2 기호 정의

- `B`: 질의 배치 크기
- `F`: 필드 수 (스키마 고정 순서)
- `M`: 스코어러 수 — `M=2` (lexical=0, dense=1)
- `D`: 후보 문서 수 (shortlist 합집합 크기, 상한 `D_max`)
- `H`: query embedding 차원 (`TrainConfig.query_dim`, default 768)

### 3.3 doc_id 타입 규칙

`aggregate_field_scores_hybrid_async`는 doc_id를 항상 **`str`** 로 정규화 (`_normalize_doc_id`).  
모든 하위 코드(shortlist 조회, 정렬, inference)는 `str` doc_id를 가정한다.

---

## 4. 모듈 상세

### 4.1 `aggregate_field_scores_hybrid_async` — shortlist 구축

**파일**: `asmr/src/asmr/retrieve/aggregate.py`

```python
async def aggregate_field_scores_hybrid_async(
    query: Union[str, Query],
    field_lex_dense_pairs: list[tuple[str, str]],  # [(lex_key, dense_key), ...]
    doc_retriever: QueryRouter,
    k: int = 100,
) -> tuple[list[str], np.ndarray]:  # doc_ids (str), scores [F, 2, D]
```

- 각 논리 필드마다 lexical(`m=0`)과 dense(`m=1`) retriever를 **별도** 호출
- 두 스코어러의 합집합을 shortlist로 사용 (mFAR §2.2 근사)
- 출력 `scores[:, 0, :]` = lexical, `scores[:, 1, :]` = dense

기존 `aggregate_field_scores_async`는 **단일 스코어러 `[F, D]`** — baseline·비교 시 사용.

---

### 4.2 `MFARFieldAdapter` — mFAR 선형 G

**파일**: `asmr/src/asmr/train/aggregation.py`

mFAR 논문 식 (1)의 쿼리 조건부 softmax 가중합:

```
G(q,f,m) = softmax_{f,m}(a_{f,m}^T q)
s(q,d)   = Σ_{f,m} G(q,f,m) · s̃_{f,m}(q,d)
```

- 입력: `query_emb [B, H]`, `scores [B, F, M, D]`
- 출력: `logits [B, D]`
- 파라미터: `Linear(H, F*M)` 1개
- **ablation "No query-conditioning"**: `MFARFieldAdapter`의 `_weight_logits`를 전역 상수로 고정

---

### 4.3 `AggregationHead` — MLP G_θ

**파일**: `asmr/src/asmr/train/aggregation.py`

```
x = concat([scores_flat [F*M], query_emb [H], aux [A]], dim=-1)  # per-document
s(q,d) = MLP(x) * scale
```

- 입력: `query_emb [B, H]`, `scores [B, F, M, D]`, `aux [B, D, A]`
- 출력: `logits [B, D]`
- 파라미터 상한: `hidden ≤ 256`, 1~2층 MLP, dropout=0.1, L2 권장
- 초기화 전략 (ADR-001 연동): 첫 에폭은 linear G warm-start 후 G_θ 미세조정

---

### 4.4 `build_aux_features` — 보조 특성

**파일**: `asmr/src/asmr/train/features.py`

출력 `[B, D, 9]` — 9개 특성:

| 인덱스 | 이름 | 설명 |
|--------|------|------|
| 0 | `var` | 필드별 점수 분산 (필드 간 불일치 신호) |
| 1 | `max` | 필드 점수 최대값 |
| 2 | `min` | 필드 점수 최소값 |
| 3 | `mean` | 전체 필드 평균 |
| 4 | `span` | max - min |
| 5 | `masked_mean` | field_mask 적용 평균 |
| 6 | `scorer_spread` | lexical vs dense 점수 차이 (M>1 시) |
| 7 | `density` | 필드 coverage 비율 (보유 필드 수 / 전체 필드 수) |
| 8 | bias | 상수 1.0 |

`TrainConfig.aux_dim = 9` (`config.py` 기본값과 일치).

---

### 4.5 `normalize_scores_per_field_scorer` — 점수 정규화 (Opt A)

**파일**: `asmr/src/asmr/train/features.py`

필드×스코어러 단위 batch normalization (mFAR §2.2 정렬):

```python
normalize_scores_per_field_scorer(scores: [B,F,M,D], field_mask: [B,F,D]) -> [B,F,M,D]
```

- 학습 배치 안에서 `(f,m)`별 mean/var → whitening
- `TrainConfig.normalize_scores: bool = False` 플래그로 학습 스텝 내 적용 여부 제어 (기본 off)
- 서빙 시에는 Opt B (고정 min-max) 또는 running stats 사용 고려 — 현재 미구현

---

### 4.6 `HfQueryEncoder` — 질의 임베딩

**파일**: `asmr/src/asmr/train/query_encoder.py`

```python
encoder = HfQueryEncoder(model_name="facebook/contriever-msmarco")
q_emb = encoder.encode(["query text"])  # Tensor [1, H], L2 normalized
```

- mean-pool (attention mask 가중) + L2 normalize
- Phase 1에서는 `@torch.no_grad()` — 파라미터 동결
- `embedding_dim` property로 `TrainConfig.query_dim` 자동 설정 가능

대체 모델: `BAAI/bge-m3` (dense 채널만 사용 시 `query_dim=1024`).

---

### 4.7 `RankingBatch` / `StarkRankingDataset` — 학습 데이터

**파일**: `asmr/src/asmr/train/data.py`, `asmr/src/asmr/train/data_stark.py`

```python
@dataclass
class RankingBatch:
    scores: Tensor      # [B, F, M, D]
    field_mask: Tensor  # [B, F, D]
    query_emb: Tensor   # [B, H]
    relevance: Tensor   # [B, D]
    aux: Tensor | None  # [B, D, A] optional
```

`StarkRankingExample` → `collate_stark_batch(batch, query_encoder, device)` → `RankingBatch`.

현재 `collate_stark_batch`는 **batch_size=1만 지원** (shortlist 너비 D가 예시마다 다름).  
패딩 또는 고정 D_max 도입 시 B>1 지원 가능.

---

### 4.8 `AggregationTrainer` — 학습 스텝

**파일**: `asmr/src/asmr/train/trainer.py`

```python
trainer = AggregationTrainer(model, config)
out = trainer.training_step(batch)  # StepOutput(loss, loss_rank)
out.loss.backward()
```

손실:

```
L = λ_rank * L_rank + (λ_field * L_field) + (λ_distill * L_distill)
```

| 손실 | 함수 | 기본 활성 |
|------|------|-----------|
| `L_rank` | `listwise_logit_loss` (softmax CE) | 항상 |
| `L_field` | BCE on field-level labels | `lambda_field > 0` 시 |
| `L_distill` | KL/MSE vs teacher score | `lambda_distill > 0` 시 |

기본값: `lambda_rank=1.0`, `lambda_field=0.0`, `lambda_distill=0.0`.

---

### 4.9 `apply_aggregation_head` — 추론 경로

**파일**: `asmr/src/asmr/train/inference.py`

```python
sorted_ids, logits = apply_aggregation_head(
    doc_ids,      # list[str], length D
    scores,       # np [F, M, D] or [F, D]
    field_mask,   # np [F, D]
    query_emb,    # np [H]
    head,         # MFARFieldAdapter or AggregationHead
)
```

- numpy 입력 → torch 변환 → head forward → argsort (descending) → numpy 반환
- `DocumentRetriever`에서 optional hook으로 호출 (`retrieve_with_aggregation_head`)

---

## 5. 학습 흐름

### 5.1 데이터 흐름

```
(query_text, doc_ids, scores[F,M,D], relevance[D], field_mask[F,D])
    ↓ collate_stark_batch
RankingBatch
    ↓ normalize_scores_per_field_scorer (TrainConfig.normalize_scores=True 시)
    ↓ build_aux_features → aux [B, D, 9]
    ↓ AggregationHead / MFARFieldAdapter
logits [B, D]
    ↓ listwise_logit_loss(logits, relevance, τ)
loss
    ↓ .backward() + optimizer.step()
```

### 5.2 학습 entrypoint

```bash
# smoke (dummy tensors, no HF download)
python -m asmr.train --dummy --epochs 2

# 실제 (contriever encoder)
python -m asmr.train --encoder facebook/contriever-msmarco --epochs 10 \
    --checkpoint ckpt/mfar_head.pt
```

---

## 6. 추론 흐름

```bash
# 학습 없이 기존 aggregate baseline
aggregate_field_scores_async(q, fields, router, k=100)
    → (doc_ids, scores[F, D])
    → 필드 합산 → ranking

# 학습된 head 사용
aggregate_field_scores_hybrid_async(q, field_pairs, router, k=100)
    → (doc_ids, scores[F, 2, D])
    → apply_aggregation_head(doc_ids, scores, mask, q_emb, head)
    → sorted_ids (reranked)
```

---

## 7. 현재 구현 상태

| 항목 | 파일 | 상태 |
|------|------|------|
| `[F,D]` → `[F,M,D]` hybrid shortlist | `retrieve/aggregate.py` — `aggregate_field_scores_hybrid_async` | 구현됨 |
| doc_id str 통일 | `retrieve/aggregate.py` — `_normalize_doc_id` | 구현됨 |
| `MFARFieldAdapter` (linear G) | `train/aggregation.py` | 구현됨 |
| `AggregationHead` (G_θ MLP) | `train/aggregation.py` | 구현됨 |
| `build_aux_features` | `train/features.py` | 구현됨 |
| `normalize_scores_per_field_scorer` | `train/features.py` | 구현됨 |
| `HfQueryEncoder` | `train/query_encoder.py` | 구현됨 |
| `RankingBatch` | `train/data.py` | 구현됨 |
| `StarkRankingDataset` + collate | `train/data_stark.py` | 구현됨 (B=1) |
| `AggregationTrainer.training_step` | `train/trainer.py` | 구현됨 |
| `apply_aggregation_head` | `train/inference.py` | 구현됨 |
| `train_script.py` (`python -m asmr.train`) | `train/train_script.py` | 구현됨 (dummy smoke) |
| `DocumentRetriever` optional head hook | `retrieve/helpers.py` | 구현됨 (`retrieve_with_aggregation_head`) |
| STaRK 실제 데이터 연동 (qrels) | `train/data_stark.py` | **미완** — 데이터 준비 필요 |
| `asmr/evaluation/metrics.py` (Hit@1, R@20, MRR) | `asmr/src/asmr/evaluation/` | **미완** |
| `asmr/evaluation/stark_eval.py` | `asmr/src/asmr/evaluation/` | **미완** |

---

## 8. 남은 작업 (우선순위 순)

### P0 — 데이터 연동 (STaRK)

- STaRK 공식 스플릿 다운로드 + [microsoft/multifield-adaptive-retrieval](https://github.com/microsoft/multifield-adaptive-retrieval) 전처리 스크립트 정합
- `StarkRankingDataset`에 실제 JSONL 로더 추가
- qrels 파일 연동 → `relevance` 텐서 생성

### P1 — 평가 파이프라인 (`asmr/evaluation/`)

- `asmr/evaluation/metrics.py`: Hit@1, Recall@20, MRR (`trec_eval` 또는 `pytrec_eval`)
- `asmr/evaluation/stark_eval.py`: asmr 인덱스 → shortlist → head → 지표 계산
- 성공 기준: mFAR 논문 핵심 실험 대비 **90% 이상** (job-request.md)

### P2 — 학습 루프 강화

- B>1 지원: shortlist 패딩 (`D_max`) 또는 dynamic padding collate
- `normalize_scores_per_field_scorer` 적용 경로 `TrainConfig`에 명시적 플래그로 추가
- `pairwise_hinge_loss` 비교 실험 (현재 `listwise_logit_loss` 기본)
- Hard negative 샘플링: single-field lure, constraint 불만족 등 체계적 생성

### P3 — Phase 2 (ADR-001)

- 인코더 fine-tuning (`TrainConfig.finetune_encoder: bool = False`)
- `L_c` (contrastive) + `L_b` (bi-directional) 손실 추가
- Phase 1 수렴 확인 후 진행

---

## 9. 품질 게이트

```bash
# 1. 스타일·타입 검사
uv run ruff format asmr/src/asmr/train/
uv run ruff check --fix asmr/src/asmr/train/
uv run mypy asmr/src/asmr/train/

# 2. 유닛 테스트
uv run pytest asmr/tests/unit/asmr/train/ -v

# 3. smoke (end-to-end, torch 환경 필요)
python -m asmr.train --dummy --epochs 1
```

유닛 테스트 위치: `asmr/tests/unit/asmr/train/`  
커버 대상: shortlist shape 계약, `RankingBatch` 필드 shape, `apply_aggregation_head` 반환 순서.

---

## 10. 설계 결정 참조

| 결정 | 문서 |
|------|------|
| head-only 학습 (Phase 1) vs 인코더 공동 학습 (Phase 2) | [ADR-001](../adr/001-head-only-training-phase1.md) |
| 점수 정규화 Opt A/B/C 선택 | [기술 상세 §3.3](../../trainable-retrieval-technical-detail-design.md) — Opt A 권장 (소데이터) |
| STaRK vs MIRACL/MKQA 벤치 역할 분리 | [기술 상세 §9](../../trainable-retrieval-technical-detail-design.md) — STaRK Primary, MIRACL Secondary |
| `G_θ` 과적합 완화 (파라미터 상한, warm-start) | [기술 상세 §11](../../trainable-retrieval-technical-detail-design.md) |
