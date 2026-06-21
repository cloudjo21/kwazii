# RFC 003: stark_mag_training.py — STaRK-MAG MFARAll 학습 모듈

- **상태**: 제안
- **작성일**: 2026-06-21
- **관련 파일**: `asmr/train/`, `asmr/evaluation/`, `asmr/datasets/stark_mag/`

---

## 1. 배경 및 문제

현재 `stark_mag_benchmark.py`는 학습과 평가 로직이 한 파일에 섞여 있다.

```
stark_mag_benchmark.py
├── MagFieldIndexes.__init__()  ← 인덱스 빌드 (학습·평가 공용)
├── train_mag_adapter()         ← 학습 루프 (인라인, 캐싱 없음)
└── run_benchmark()             ← 평가 파이프라인
```

`stark_prime_training.py`가 학습을 `asmr/train/` 아래 독립 모듈로 분리하고 있는 것과 대비된다.

**문제점:**
- `train_mag_adapter()`는 shortlist를 캐싱하지 않아 매 실행마다 인덱스 빌드 + 임베딩 재계산 필요
- `MagFieldIndexes`가 benchmark 파일에 종속되어 학습 스크립트에서 재사용할 수 없음
- Phase 2 (encoder joint FT) 진입점이 없음

---

## 2. 제안 설계

### 2-1. 모듈 구조

```
asmr/src/asmr/
├── evaluation/
│   └── stark_mag_indexes.py        ← (신규) MagFieldIndexes 이전
├── datasets/stark_mag/
│   └── shortlist_cache.py          ← (신규) 단순 pickle 기반 shortlist 캐시
└── train/
    └── stark_mag_training.py       ← (신규) 학습 진입점
```

`stark_mag_benchmark.py`는 `MagFieldIndexes`를 `stark_mag_indexes.py`에서 import하도록 수정한다. 벤치마크 공개 API는 변경 없음.

### 2-2. stark_mag_indexes.py

`MagFieldIndexes`를 `stark_mag_benchmark.py`에서 추출하여 독립 모듈로 이전한다.

```python
# asmr/src/asmr/evaluation/stark_mag_indexes.py

class MagFieldIndexes:
    """Per-field BM25 + FAISS 인덱스 (5필드 × 2 scorer)."""

    def __init__(
        self,
        corpus: MagCorpus,
        encoder: TextEncoderProtocol,
        *,
        tmpdir: tempfile.TemporaryDirectory | None = None,
    ) -> None: ...

    def shortlist_hybrid(
        self, query_text: str, query_emb: np.ndarray, k: int
    ) -> tuple[list[str], np.ndarray, np.ndarray]: ...

    def close(self) -> None: ...
```

**변경 이유**: 학습 루프가 인덱스를 참조하는데 benchmark 파일을 import하면 순환 의존이 발생한다.

### 2-3. shortlist_cache.py (stark_mag)

Prime의 streaming JSONL 방식을 MAG에 그대로 쓰기엔 과도하다. MAG corpus는 최대 200 k doc, 5 field로 Prime보다 작다. 단순 pickle 기반 캐시를 제공한다.

```python
# asmr/src/asmr/datasets/stark_mag/shortlist_cache.py

CACHE_VERSION = 1

def shortlist_cache_path(
    cache_dir: Path, split: str, shortlist_k: int, encoder_name: str
) -> Path: ...

def save_shortlist_cache(
    examples: list[StarkRankingExample],
    cache_dir: Path,
    split: str,
    shortlist_k: int,
    encoder_name: str,
) -> None:
    """examples를 pickle로 직렬화하여 저장."""

def load_shortlist_cache(
    cache_dir: Path, split: str, shortlist_k: int, encoder_name: str
) -> list[StarkRankingExample] | None:
    """캐시 파일이 없거나 버전 불일치 시 None 반환."""
```

캐시 파일 레이아웃:
```
cache_dir/
└── mag_shortlists/
    └── train_k100_contriever-msmarco.pkl   ← {version, examples}
```

### 2-4. stark_mag_training.py

학습의 핵심 함수 두 가지로 구성한다.

#### `build_mag_training_examples`

```python
def build_mag_training_examples(
    train_queries: list[MagQuery],
    indexes: MagFieldIndexes,
    encoder: TextEncoderProtocol,
    *,
    shortlist_k: int,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
) -> list[StarkRankingExample]:
    """shortlist를 생성하고 선택적으로 캐싱한다."""
```

**흐름:**
1. `cache_dir`가 지정되고 캐시 파일이 있으면 `load_shortlist_cache()` → 즉시 반환
2. 없으면 각 query에 대해 `indexes.shortlist_hybrid(query_text, q_emb, shortlist_k)` 호출
3. `_example_from_shortlist()` 로 `StarkRankingExample` 변환
4. `cache_dir` 지정 시 `save_shortlist_cache()` 저장

#### `train_mfar_mag`

```python
def train_mfar_mag(
    train_queries: list[MagQuery],
    indexes: MagFieldIndexes,
    encoder: TextEncoderProtocol,
    cfg: TrainMfarConfig,
    device: torch.device,
    *,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
) -> MFARFieldAdapter:
    """MFARFieldAdapter Phase 1 학습 (head-only)."""
```

**구현 요점:**
- `f_num = len(MAG_FIELD_NAMES)` → `5`
- `num_scorers = 2` (sparse BM25, dense FAISS)
- `MFARFieldAdapter(encoder.embedding_dim, f_num, 2).to(device)`
- `TrainConfig(query_dim=encoder.embedding_dim)` → `AggregationTrainer`
- 에포크당 shuffle → `collate_stark_batch()` → `trainer.training_step()` → backward
- Phase 2 지원 여부: `cfg.phase == TrainingPhase.JOINT`이면 `_build_optimizer()` 위임 (단, Phase 2는 Phase 1 안정화 후 구현)

#### `check_gate_90_mag`

Prime용 `check_gate_90()`와 임계값 상수만 다르다. MAG 논문 수치는:

```python
GATE_RATIO_90_MAG = {"hit@1": 0.90, "recall@20": 0.90, "mrr": 0.90}
```

Prime과 동일 비율이므로 `stark_prime_training.check_gate_90()` 를 직접 재사용한다 (별도 정의 불필요).

---

## 3. Prime과 MAG 학습의 차이점 요약

| 항목 | Prime | MAG |
|------|-------|-----|
| 인덱스 소스 | `PrimeDiskIndexStore` (disk mmap) | `MagFieldIndexes` (in-memory FAISS + BM25) |
| 필드 수 | 22 | 5 |
| Shortlist 캐시 | streaming JSONL chunks (large corpus) | pickle (single file, MAG corpus 작음) |
| Query 임베딩 캐시 | `QueryEmbeddingCache` | 캐싱 불필요 (예: 소수 쿼리 세트) |
| Phase 2 인코더 | `JinaLoraQueryEncoder` 지원 | Phase 1만 먼저 구현, Phase 2는 후속 |
| `collate_stark_batch` | `stark_prime/torch_dataset.py` | 동일 함수 재사용 |

---

## 4. 파일별 변경 목록

| 동작 | 파일 | 내용 |
|------|------|------|
| 신규 | `asmr/evaluation/stark_mag_indexes.py` | `MagFieldIndexes` 이전 |
| 신규 | `asmr/datasets/stark_mag/shortlist_cache.py` | pickle 캐시 API |
| 신규 | `asmr/train/stark_mag_training.py` | `build_mag_training_examples`, `train_mfar_mag` |
| 수정 | `asmr/evaluation/stark_mag_benchmark.py` | `MagFieldIndexes` import 경로 변경 |
| 수정 | `asmr/datasets/stark_mag/__init__.py` | shortlist_cache re-export (선택) |

---

## 5. 단위 테스트 계획

**파일**: `tests/unit/asmr/train/test_stark_mag_training.py`

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_build_examples_no_cache` | `MagFieldIndexes` mock → examples 생성됨 |
| `test_build_examples_cache_hit` | 캐시 파일 존재 시 인덱스 호출 없음 |
| `test_build_examples_cache_write` | `cache_dir` 지정 시 pickle 파일 생성 |
| `test_train_mfar_mag_phase1` | 5-field stub → 학습 후 `MFARFieldAdapter` 반환, 파라미터 변경 확인 |
| `test_check_gate_90_pass` | 0.90 이상 ratio → pass |
| `test_check_gate_90_fail` | 0.89 이하 ratio → fail |

---

## 6. 구현 순서

1. `stark_mag_indexes.py` 신규 생성 + `stark_mag_benchmark.py` import 경로 수정
2. `shortlist_cache.py` 신규 생성
3. `stark_mag_training.py` 신규 생성
4. 단위 테스트 작성 및 통과 확인
5. 기존 `stark_mag_benchmark.py`의 `train_mag_adapter()` 제거 (학습을 `stark_mag_training.py`로 위임)
