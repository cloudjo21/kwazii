# ADR-002: PyTorch 환경 선택 및 GPU 학습 실행 가능성 검증

- **Status**: Accepted
- **Date**: 2026-04-14
- **Deciders**: asmr team

---

## Context

Phase 1 학습(ADR-001)을 실제로 실행하기 전에 두 가지를 검토해야 했다.

1. **런타임 환경**: RTX 3090 Ti + CUDA Driver 12.4 조합에서 어떤 PyTorch 버전이 최적인가?
2. **VRAM 여유**: encoder frozen 전제에서 head 학습에 필요한 GPU 메모리는 얼마이고, 현재 GPU로 실행 가능한가?

추가로 코드베이스 실행 중 `torch.nanvar`, `torch.nanmax`, `torch.nanmin` 호출이 PyTorch 2.6.0에서 `AttributeError`를 유발하는 버그가 발견되었다.

### 하드웨어 스펙

| 항목 | 값 |
|------|----|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| VRAM | 24 GB |
| CUDA Compute Capability | 8.6 (Ampere) |
| CUDA Driver | 12.4 (556.12) |

### PyTorch 설치 현황

| 항목 | 값 |
|------|-----|
| 버전 | 2.6.0+cu124 |
| 소스 | PyPI (`https://pypi.org/simple`) |
| Python | 3.13 |
| 패키지 관리 | uv (`asmr/uv.lock` 고정) |

---

## Decision

### 1. PyTorch 2.6.0+cu124 유지

PyTorch 버전을 현재 `2.6.0+cu124`에서 변경하지 않는다.

### 2. `torch.nanvar` / `torch.nanmax` / `torch.nanmin` 수동 구현으로 교체

`asmr/src/asmr/train/features.py`의 해당 호출을 PyTorch 표준 연산 조합으로 대체한다.

### 3. 기본 encoder로 `facebook/contriever-msmarco` 채택 확정

Phase 1에서 `HfQueryEncoder`의 기본 모델 ID를 `facebook/contriever-msmarco`(768-dim)로 유지한다.

---

## Rationale

### PyTorch 버전 유지 이유

PyPI 기준 최신 버전(2.11.0)까지 확인했으나 2.6.0+cu124 유지를 선택한 근거:

1. **cu124 빌드 = 드라이버 정확 매칭**: `+cu124` 빌드는 CUDA 12.4 라이브러리를 번들 포함. 드라이버(12.4)와 정확히 맞아 런타임 호환 리스크 없음.
2. **Ampere(CC 8.6) 완전 지원**: TF32, cuBLAS, Flash Attention 모두 정상 동작.
3. **업그레이드 리스크**: PyTorch 2.7.0+는 cu126(CUDA 12.6) 이상 빌드가 기본. 현재 드라이버(12.4)에서 cu126 빌드 실행 시 toolkit 버전 미스매치 가능 — cu124 빌드 별도 확인 필요.
4. **`torch.nanvar` 부재는 버전 문제가 아님**: PyTorch 전 버전에서 top-level `torch.nanvar`는 존재하지 않는다. 업그레이드로 해결되지 않음.

업그레이드가 필요한 경우 아래 방식으로 cu124 빌드를 지정한다:
```bash
uv add "torch>=2.7.0" --extra-index-url https://download.pytorch.org/whl/cu124
```

### `torch.nan*` 함수 대체 이유

PyTorch top-level namespace에 `torch.nanvar`, `torch.nanmax`, `torch.nanmin`은 존재하지 않는다(`torch.nanmean`은 존재). `features.py`가 이를 호출해 `AttributeError` 발생 → 테스트 1건 실패.

대체 구현:

| 원래 호출 | 대체 구현 |
|-----------|-----------|
| `torch.nanvar(x, dim, unbiased=False)` | 수동 마스크 분산: `nan` 위치 제외 후 mean 계산 → `(x - mean)^2` 합산 / 유효 count |
| `torch.nanmax(x, dim).values` | `x.masked_fill(isnan, -inf).max(dim).values` |
| `torch.nanmin(x, dim).values` | `x.masked_fill(isnan, +inf).min(dim).values` |

### encoder 모델 선택

| 모델 | query_dim | encoder 크기(fp32) | Phase 1 학습 peak VRAM |
|------|-----------|-------------------|------------------------|
| `facebook/contriever-msmarco` | 768 | **419 MB** | **435 MB (0.43 GB)** — 실측 |
| `BAAI/bge-m3` | 1024 | ~2,136 MB | ~2,146 MB — 추정 |

contriever가 STaRK 논문 백본과 동일하고 메모리 부담이 작아 Phase 1 시작점으로 적합.  
bge-m3도 현재 GPU에서 충분히 실행 가능하나 Phase 2(encoder fine-tuning) 시 그래디언트·옵티마이저 상태가 추가되므로 contriever 우선 검증 후 전환 권장.

---

## GPU 메모리 실측 결과

> 측정 조건: RTX 3090 Ti, PyTorch 2.6.0+cu124, B=1, D=100(shortlist), F=2, M=2  
> 측정 방법: `torch.cuda.max_memory_allocated()` (학습 1 step)

### 컴포넌트별 VRAM

| 컴포넌트 | 파라미터 수 | VRAM |
|----------|------------|------|
| `contriever-msmarco` encoder weights | 110M | **419 MB** |
| encoder forward activation (`no_grad`) | — | **9 MB** (transient) |
| `MFARFieldAdapter` (Linear G) | 3,072 | 0.012 MB |
| `AggregationHead` (MLP, hidden=256) | 200,450 | 0.783 MB |
| score tensor `[1,2,2,100]` + aux `[1,100,9]` | — | < 1 MB |

**encoder weight이 전체 VRAM의 96%를 점유. head 자체는 무시 수준.**

### 학습 시나리오별 실행 가능성

| 시나리오 | Peak VRAM | 24 GB 대비 | 가능 여부 |
|----------|-----------|-----------|-----------|
| contriever + MFARFieldAdapter, B=1 | **435 MB (1.8%)** | 충분 | **가능** |
| contriever + AggregationHead, B=1 | **~437 MB (1.8%)** | 충분 | **가능** |
| bge-m3 + MFARFieldAdapter, B=1 | **~2,146 MB (9%)** | 충분 | **가능** |
| contriever + head, D=1000 | **~440 MB** | 충분 | **가능** |
| Phase 2 (contriever fine-tuning, B=1) | **~1.5–2 GB est.** | 충분 | **가능** |

모든 Phase 1 시나리오에서 24 GB VRAM이 충분하다.

---

## Consequences

**Positive**
- PyTorch 버전 고정으로 재현성 보장 (`uv.lock`에 2.6.0 고정).
- `torch.nan*` 수동 구현은 PyTorch 버전에 의존하지 않아 이식성 향상.
- VRAM 여유(최대 22 GB)가 충분해 Phase 2 encoder fine-tuning, 배치 패딩, D 확장 모두 현재 GPU에서 가능.

**Negative / Trade-off**
- PyTorch 2.7.0+ 기능(신규 API 등) 미사용.
- `torch.nanvar` 수동 구현은 PyTorch 공식 구현 대비 극미한 수치 차이 가능(부동소수점 연산 순서 차이). 실험적으로 유의미한 오차 없음.

**후속 결정이 필요한 시점**
- Phase 2(encoder fine-tuning) 전환 시: `torch.cuda.max_memory_allocated()` 재측정 필요.
- CUDA Driver 업그레이드(12.6+) 시: PyTorch 2.7.0+cu126 빌드 전환 검토.

---

## 관련 파일

- `asmr/src/asmr/train/features.py` — `torch.nan*` 대체 구현
- `asmr/src/asmr/train/query_encoder.py` — `HfQueryEncoder` (기본 모델 ID)
- `asmr/docs/environment-analysis.md` — 환경 상세 분석
- `asmr/docs/engineering/runbooks/train-stark-aggregation-head.md` — 학습 실행 runbook (GPU 메모리 섹션 포함)
- ADR-001: Phase 1 head-only 학습 채택 근거
