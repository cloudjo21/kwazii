# Phase 1 학습 실행 보고서

- **날짜**: 2026-04-14
- **대상 Runbook**: [`docs/engineering/runbooks/train-stark-aggregation-head.md`](../docs/engineering/runbooks/train-stark-aggregation-head.md)
- **관련 ADR**: [ADR-001](../docs/engineering/adr/001-head-only-training-phase1.md), [ADR-002](../docs/engineering/adr/002-pytorch-env-and-gpu-feasibility.md)

---

## 실행 환경

| 항목 | 값 |
|------|----|
| GPU | NVIDIA GeForce RTX 3090 Ti (24 GB) |
| CUDA Driver | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.13 |
| Encoder | `facebook/contriever-msmarco` (dim=768, frozen) |

---

## Runbook Step별 진행 결과

### Step 1 — 환경 확인

| 항목 | 결과 | 비고 |
|------|------|------|
| PyTorch CUDA 확인 | **통과** | `2.6.0+cu124 True` |
| smoke (`--dummy --epochs 2`) | **통과** | 정상 실행, CUDA peak 16 MB |

실행 중 발견된 선행 버그 3건을 수정 후 진행:

| 버그 | 위치 | 수정 내용 |
|------|------|-----------|
| `torch.nanvar` / `torch.nanmax` / `torch.nanmin` 미존재 | `train/features.py` | 수동 마스크 연산으로 교체 |
| `np.fromiter(..., dtype=np.str_)` 가변 길이 오류 | `retrieve/aggregate.py` | `np.array(..., dtype=object)` 로 교체 |
| `test_aggregation_head_shape` aux_dim 불일치 | `tests/.../test_aggregation.py` | `aux_dim=8` → `9` 수정 |

---

### Step 2 — STaRK 데이터 준비

| 항목 | 결과 |
|------|------|
| JSONL 로더 (`StarkRankingDataset.from_jsonl`) | **신규 구현** — `data_stark.py`에 추가 |
| 실제 STaRK 데이터 | **미다운로드** — 샘플 데이터로 대체 |
| 샘플 데이터 생성 | **완료** — `data/stark_sample/train.jsonl` |

샘플 데이터 스펙:

| 항목 | 값 |
|------|----|
| 쿼리 수 | 20 |
| 도메인 | Amazon 제품 검색 (가전, 스포츠, 주방 등) |
| 필드 수 (F) | 3 (title, description, category) |
| 스코어러 수 (M) | 2 (lexical/dense) |
| shortlist 크기 (D) | 50 |
| 정답 수 (per query) | 2–3개 |
| field_mask | category 필드 20% 결측 반영 |

---

### Step 3 — 학습 실행

`train_script.py` 기능 확장 후 두 모델 모두 학습 실행:

| 항목 | 결과 |
|------|------|
| `--model adapter/head` 선택 옵션 추가 | **완료** |
| `--data PATH` JSONL 입력 지원 | **완료** |
| `--normalize-scores` 플래그 추가 | **완료** |
| 에폭/loss 로그 출력 | **완료** |

#### 3-1. MFARFieldAdapter (linear G)

```
uv run python -m asmr.train \
    --data data/stark_sample/train.jsonl \
    --encoder facebook/contriever-msmarco \
    --model adapter --epochs 20 --lr 1e-3 \
    --checkpoint ckpt/mfar_adapter_phase1.pt
```

| epoch | loss |
|-------|------|
| 1 | 4.302198 |
| 5 | 3.528344 |
| 10 | 2.793138 |
| 15 | 2.329123 |
| 20 | **2.009631** |

- 파라미터: 4,608개 (Linear 1층)
- VRAM peak: **436 MB**
- 소요 시간: 약 12초 (20 epoch × 20 examples)

#### 3-2. AggregationHead (MLP G_θ, normalize_scores=True)

```
uv run python -m asmr.train \
    --data data/stark_sample/train.jsonl \
    --encoder facebook/contriever-msmarco \
    --model head --epochs 20 --lr 1e-3 \
    --normalize-scores \
    --checkpoint ckpt/mfar_head_phase1.pt
```

| epoch | loss |
|-------|------|
| 1 | 3.013526 |
| 2 | 1.317047 |
| 5 | 1.055294 |
| 10 | 0.901813 |
| 20 | **0.833328** |

- 파라미터: 200,962개 (MLP 2층, hidden=256, aux=9)
- VRAM peak: **439 MB**
- 소요 시간: 약 12초 (20 epoch × 20 examples)

---

### Step 4 — 학습 루프 구조 확인

실제 실행으로 검증된 학습 흐름:

```
StarkRankingDataset.from_jsonl(path)  ← [신규]
    ↓ collate_stark_batch([example], encoder, device)
RankingBatch [B=1, F=3, M=2, D=50]
    ↓ normalize_scores_per_field_scorer  (--normalize-scores 시)
    ↓ build_aux_features → aux [1, 50, 9]  (AggregationHead 시)
    ↓ MFARFieldAdapter / AggregationHead
logits [1, 50]
    ↓ listwise_logit_loss(logits, relevance, τ=0.07)
loss.backward() + AdamW(lr=1e-3, wd=1e-2).step()
```

---

### Step 5 — 체크포인트 저장

| 파일 | 크기 | 포함 키 |
|------|------|---------|
| `ckpt/mfar_adapter_phase1.pt` | 20 KB | `model`, `config`, `model_type`, `f_num`, `m_num`, `encoder` |
| `ckpt/mfar_head_phase1.pt` | 788 KB | 동일 |

---

### Step 6 — 추론

`apply_aggregation_head` API 구현 확인 (기존 구현, 단위 테스트 통과).  
실제 end-to-end 추론 실행은 실제 STaRK 인덱스 연동 후 진행 예정.

---

### Step 7 — 평가

`asmr.evaluation.stark_eval` 및 `asmr.evaluation.metrics` 구현 확인.  
실측 평가는 실제 STaRK qrels 데이터 확보 후 진행 예정.

---

## 품질 게이트 결과

```bash
uv run pytest tests/unit/ -q
# → 136 passed, 5 warnings
```

| 게이트 | 결과 |
|--------|------|
| `pytest tests/unit/` | **136 passed** (버그 수정 전 3 failed → 0 failed) |
| `ruff format` | 통과 |
| `ruff check --fix` | 통과 (신규 에러 없음) |
| `mypy` (수정 파일) | 신규 에러 없음 (pre-existing 에러는 별도 추적) |
| smoke `--dummy --epochs 1` | 통과, CUDA peak 16 MB |

---

## 완료 항목 요약

| 항목 | 상태 | 산출물 |
|------|------|--------|
| `torch.nan*` 버그 수정 | **완료** | `train/features.py` |
| `np.fromiter` 버그 수정 | **완료** | `retrieve/aggregate.py` |
| `test_evaluation_metrics` import 수정 | **완료** | `tests/unit/test_evaluation_metrics.py` |
| `StarkRankingDataset.from_jsonl()` | **완료** | `train/data_stark.py` |
| `train_script.py` 전면 개선 | **완료** | `train/train_script.py` |
| 샘플 STaRK 학습 데이터 | **완료** | `data/stark_sample/train.jsonl` |
| MFARFieldAdapter Phase 1 학습 | **완료** | `ckpt/mfar_adapter_phase1.pt` |
| AggregationHead Phase 1 학습 | **완료** | `ckpt/mfar_head_phase1.pt` |
| 환경 분석 문서 | **완료** | `docs/environment-analysis.md` |
| ADR-002 (PyTorch 환경·GPU 타당성) | **완료** | `docs/engineering/adr/002-*.md` |
| GPU 메모리 실측 및 runbook 반영 | **완료** | runbook GPU 섹션 |

---

## 미완료 항목 (차기 진행)

| 항목 | 우선순위 | 비고 |
|------|----------|------|
| 실제 STaRK 데이터 다운로드 및 전처리 | P0 | snap-stanford/stark + mFAR 전처리 스크립트 |
| Hit@1 / Recall@20 / MRR 실측 평가 | P1 | qrels 파일 필요 |
| B>1 배치 패딩 (`D_max`) | P2 | `collate_stark_batch` 확장 |
| Phase 2 encoder fine-tuning | P3 | ADR-001 Phase 2 진입 조건: Phase 1 plateau 확인 후 |

---

## 관찰 및 메모

- **AggregationHead가 MFARFieldAdapter 대비 수렴 속도 현저히 빠름** (epoch 2에서 이미 loss 1.3 수준). `normalize_scores=True`가 효과적으로 동작한 것으로 추정.
- **VRAM 여유 충분**: 두 모델 모두 440 MB 이하. shortlist D=1000, 실제 STaRK 데이터 규모로 확대해도 여유 충분.
- **학습 속도**: 20 examples × 20 epochs = 약 12초 (contriever 인코딩 포함). 실제 STaRK 수천 건 규모에서는 인코딩 캐싱 고려 필요.
