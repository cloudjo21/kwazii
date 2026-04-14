# asmr 개발 환경 분석

> 작성일: 2026-04-14

---

## GPU / CUDA

| 항목 | 값 |
|------|----|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| VRAM | 24 GB |
| CUDA Compute Capability | 8.6 (Ampere) |
| CUDA Driver 버전 | 12.4 (556.12) |

---

## Python / PyTorch 환경

| 항목 | 값 |
|------|----|
| Python | 3.13 |
| 패키지 매니저 | uv |
| PyTorch | 2.6.0+cu124 |
| CUDA toolkit (번들) | 12.4 |
| 소스 | PyPI (`https://pypi.org/simple`) |
| lock 파일 | `asmr/uv.lock` |

---

## PyTorch 버전 선택 근거

PyTorch **2.6.0+cu124는 RTX 3090 Ti / CUDA Driver 12.4 조합에서 현재 최적 버전**이다.

### 이유

1. **cu124 빌드 = 드라이버 정확 매칭**  
   PyTorch의 `+cu124` 빌드는 CUDA 12.4 라이브러리를 번들로 포함한다. 드라이버 버전(12.4)과 정확히 맞아 런타임 호환 리스크가 없다.

2. **RTX 3090 Ti (Ampere, CC 8.6) 완전 지원**  
   PyTorch 2.6.0은 Ampere 아키텍처를 완전 지원한다. TF32, cuBLAS, Flash Attention 등 모두 정상 동작.

3. **상위 버전 업그레이드 시 주의 사항**  
   - PyPI 기준 최신 가용 버전: 2.11.0 (2026-04 기준)  
   - PyTorch 2.7.0+는 cu126(CUDA 12.6) 이상 빌드가 기본이다.  
   - 드라이버 12.4로 cu126 빌드 실행 시 CUDA toolkit 버전 미스매치 가능 → cu124 빌드 가용 여부 별도 확인 필요.
   - 업그레이드가 필요한 경우: `uv add "torch>=2.7.0" --extra-index-url https://download.pytorch.org/whl/cu124` 형태로 cu124 빌드 지정.

---

## 알려진 API 제약

| 함수 | 상태 | 대안 |
|------|------|------|
| `torch.nanvar` | **미존재** (PyTorch 전 버전 공통) | 수동 마스크 분산 계산 |
| `torch.nanmax` | **미존재** | `masked_fill(-inf).max()` |
| `torch.nanmin` | **미존재** | `masked_fill(+inf).min()` |
| `torch.nanmean` | 존재 (`Tensor.nanmean`, `torch.nanmean`) | — |

> `asmr/src/asmr/train/features.py`에서 위 3개 함수를 사용 중 → 수정 필요.

---

## 의존성 구조 요약

```
asmr (src/asmr/, src/fde/)
  ├── torch>=2.0  [optional: train]
  ├── transformers
  ├── accelerate
  ├── peft
  └── kwazii (로컬 editable, ../)
```

테스트: `uv run pytest tests/unit/ -v`  
스모크: `uv run python -m asmr.train --dummy --epochs 1`
