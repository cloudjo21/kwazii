# ADR-001: Phase 1 — Head-only training (frozen encoder)

- **Status**: Accepted
- **Date**: 2026-04-14
- **Deciders**: asmr team

---

## Context

### 논문의 학습 설정 (baseline)

mFAR 논문(arXiv:2410.20056)은 **query encoder와 composition head `G(q,f,m)`을 jointly end-to-end로 학습**한다. 구체적으로:

- **인코더**: Contriever-msmarco를 STaRK 데이터로 fine-tune (query encoder = document field encoder 공유)
- **손실**: contrastive `L_c` + bi-directional `L_b` (논문 식 (1)(2)) + ranking loss
- **결과**: 논문 Table 1/2의 수치는 이 joint 학습 설정에서 나온 것

즉 **논문의 full baseline을 재현하려면 Phase 2(encoder fine-tuning 포함)가 필요하다.**

### 선택지

| 방식 | 학습 대상 | 논문 대응 | 계산 비용 | STaRK 소규모 위험 |
|------|-----------|-----------|-----------|-------------------|
| **Phase 1**: head-only | `MFARFieldAdapter` / `AggregationHead` | ablation (No query-conditioning 포함) | 낮음 | 낮음 |
| **Phase 2**: end-to-end | 인코더(Contriever, BGE-M3) + head | 논문 main result | 높음 | 인코더 과적합 위험 |

Phase 1은 인코더를 **frozen**으로 두고, 필드별 shortlist score(`[F, M, D]`)와 query 임베딩을 **고정된 상태**로 받아 head만 학습한다.

---

## Decision

**Phase 1 (head-only)을 구현의 시작점으로 채택한다. 논문 full baseline 재현은 Phase 2에서 수행한다.**

Phase 2 (인코더 공동 fine-tuning)는 Phase 1 수렴 후 별도 브랜치·실험으로 진행한다.

---

## Rationale

1. **STaRK 데이터 규모**: STaRK-Amazon/Prime 학습 세트는 수천~수만 건 수준. 수억 파라미터 인코더를 함께 업데이트하면 과적합 위험이 크다.
2. **빠른 이터레이션**: head 파라미터 수(`hidden ≤ 256`, MLP 1~2층)는 전체 인코더 대비 수십만 배 적다. GPU 없이 CPU/MPS에서도 실험 가능.
3. **기준선 명확화**: 인코더 품질과 head 품질 기여를 분리해야 ablation이 깨끗하다. Phase 1에서 `G`(linear) → `G_θ`(MLP) 비교가 핵심 baseline.
4. **mFAR 논문 ablation 재현**: 논문 Table 2의 "No query-conditioning" ablation은 head를 전역 가중으로 고정한 것이다. Phase 1만으로 이 ablation 방향은 확인할 수 있다. 단, 논문 Table 1의 **main result 수치는 Phase 2(joint 학습) 없이는 재현 불가**하다.

---

## Consequences

**Positive**
- 인코더 다운로드 없이 `--dummy` 플래그로 전 파이프라인 smoke 가능 (`train_script.py`).
- `AggregationTrainer.training_step`이 `L_c`(인코더 대조 손실) 없이도 동작한다.
- `DocumentRetriever`의 optional head 경로(`retrieve_with_aggregation_head`)가 추론 시 인코더를 두 번 실행하지 않는다.

**Negative / Trade-off**
- 인코더 자체 품질이 shortlist를 제한한다. BGE-M3 dense encoder를 쓸 때 그 품질이 ceiling이 된다.
- mFAR 논문의 joint 학습 대비 성능 gap이 존재할 수 있다. Phase 2에서 `L_c` + `L_b`(bi-directional) 추가가 필요.

**Phase 2 진입 조건**
- Phase 1 Hit@1/R@20 plateau 확인 후.
- 학습 루프 분리: `TrainConfig`에 `finetune_encoder: bool = False` 플래그 추가, 인코더 파라미터 unfreeze.

---

## 관련 파일

- `asmr/src/asmr/train/trainer.py` — `AggregationTrainer` (head forward + loss)
- `asmr/src/asmr/train/losses.py` — `listwise_logit_loss`, `pairwise_hinge_loss`
- `asmr/src/asmr/train/train_script.py` — Phase 1 학습 entrypoint
- `asmr/src/asmr/train/query_encoder.py` — frozen `HfQueryEncoder`
- 설계 문서 §13.7: Phase 2 인코더 대조 학습 계획
