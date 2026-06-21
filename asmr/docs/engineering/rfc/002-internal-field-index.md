# RFC 002: asmr/index를 benchmark 전용 sparse·dense 인덱스로 통일

- **상태**: 제안
- **작성일**: 2026-06-21
- **관련 파일**: `asmr/index/`, `asmr/evaluation/`, `asmr/encode/protocol.py`

---

## 1. 배경 및 문제

`stark_prime_benchmark.py`와 `stark_mag_benchmark.py`가 두 가지 외부/인라인 구현에 의존하고 있다.

| 역할 | 현재 구현 | 문제 |
|------|-----------|------|
| Sparse BM25 | `rank_bm25.BM25Okapi` | 외부 패키지 의존; in-memory only (대규모 corpus 부적합) |
| Dense FAISS | 인라인 numpy dot-product `dense @ q_emb` | 중복 구현; L2 정규화 수동 관리 |

`asmr/index/`는 이미 프로덕션용 CSC BM25(`BM25Indexer`, `BM25Index`)와 FAISS IndexFlatIP(`TextEncodingIndexer`, `DenseTextFieldIndex`)를 보유하고 있다. benchmark가 이를 사용하지 않는 이유는 타입 제약 때문이다.

### 핵심 타입 제약

`TextEncodingIndexer.__init__(encoder: BaseFdeEncoder, ...)` — `fde` 패키지의 추상 클래스 강요.  
benchmark에서 사용하는 `HfQueryEncoder`는 `BaseFdeEncoder`를 상속하지 않고 `nn.Module`만 상속한다.

---

## 2. 제안

### 2-1. TextEncodingIndexer encoder 타입을 TextEncoderProtocol로 교체

**파일**: `src/asmr/index/text_encoder.py`

**타입 교체 가능성 검증**

`TextEncodingIndexer`가 `encoder`에서 실제로 호출하는 메서드:

```python
# line 55: add_documents()
embeddings = self.encoder.encode_text(texts, prompt_type)          # → npt.NDArray[np.float32]

# line 90: search()
query_embedding = self.encoder.encode_text([query], prompt_type)   # → npt.NDArray[np.float32]
```

두 호출 모두 `encode_text(texts, prompt_type) -> npt.NDArray[np.float32]` 만 필요.

`TextEncoderProtocol` (asmr/encode/protocol.py):
```python
def encode_text(
    self,
    texts: str | Sequence[str],
    prompt_type: PromptType,
    *,
    batch_size: int = 32,
) -> npt.NDArray[np.float32]: ...

@property
def embedding_dim(self) -> int: ...
```

`HfQueryEncoder.encode_text` 시그니처:
```python
def encode_text(
    self,
    texts: str | Sequence[str],
    prompt_type: PromptType = PromptType.QUERY,
    *,
    batch_size: int = 32,
) -> npt.NDArray[np.float32]: ...

@property
def embedding_dim(self) -> int: ...
```

**결론**: 시그니처 완전 일치. `HfQueryEncoder`는 `TextEncoderProtocol`을 구조적으로 만족한다.  
`BaseFdeEncoder` → `TextEncoderProtocol` 교체 시 동작 변화 없음.

**변경 내용**:

```python
# 변경 전
from fde.base import BaseFdeEncoder
def __init__(self, encoder: BaseFdeEncoder, config: FieldConfig): ...

# 변경 후
from asmr.encode.protocol import TextEncoderProtocol
def __init__(self, encoder: TextEncoderProtocol, config: FieldConfig): ...
```

### 2-2. DenseTextFieldIndex encoder 타입 동기화

**파일**: `src/asmr/index/fields.py:167`

`DenseTextFieldIndex`가 `TextEncodingIndexer`에 encoder를 그대로 전달하므로 같은 타입 교체 적용.

```python
# 변경 전
from fde import base
def __init__(self, config: FieldConfig, encoder: base.BaseFdeEncoder): ...

# 변경 후
from asmr.encode.protocol import TextEncoderProtocol
def __init__(self, config: FieldConfig, encoder: TextEncoderProtocol): ...
```

`DenseImageFieldIndex`의 `encoder: base.BaseFdeEncoder`는 이미지 인코더 전용이므로 변경하지 않는다.

### 2-3. FieldConfig — IndexUsage enum으로 용도 명시

**파일**: `src/asmr/index/config.py`

현재 `__post_init__`에서 DENSE + `faiss_index_path=None` → `ValueError`. benchmark는 in-memory FAISS만 필요하므로 이 검증을 조건부로 변경한다.

인덱스 용도를 나타내는 `IndexUsage` enum을 추가하고 `FieldConfig`에 `usage` 필드를 둔다.  
`SERVING`이 기본값 — 프로덕션/서빙 모드에서는 `faiss_index_path` 필수 검증을 유지.  
`BENCHMARK` — 평가/실험 모드, 검증 생략, in-memory FAISS 허용.

```python
class IndexUsage(Enum):
    SERVING = "serving"     # 기본값: 프로덕션 서빙, faiss_index_path 필수 검증 유지
    BENCHMARK = "benchmark" # 평가 모드: in-memory FAISS, 검증 생략

@dataclass
class FieldConfig:
    name: str
    tokenizer_type: TokenizerType
    representation_type: RepresentationType
    model_path: Optional[str] = None
    faiss_index_path: Optional[str] = None
    usage: IndexUsage = IndexUsage.SERVING   # ← 추가

    def __post_init__(self):
        if self.tokenizer_type == TokenizerType.HF_AUTO and self.model_path is None:
            raise ValueError("model_path is required when tokenizer_type is HF_AUTO")
        if (
            self.representation_type == RepresentationType.DENSE
            and self.faiss_index_path is None
            and self.usage == IndexUsage.SERVING  # BENCHMARK이면 검증 생략
        ):
            raise ValueError("faiss_index_path is required when representation_type is DENSE")
```

### 2-4. SparseTextFieldIndex.add_documents() — index_dir 파라미터 추가

**파일**: `src/asmr/index/fields.py:95`

현재 `BM25Indexer.build()` 호출에 `index_dir`을 전달하지 않아 기본 경로(`./path/to/index/dir`)에 memmap 파일을 씀. 

```python
def add_documents(
    self,
    doc_ids: List[str],
    contents: List[str],
    *,
    index_dir: Path | None = None,   # ← 추가
) -> None:
    ...
    self.index: bm25.BM25Index = bm25.BM25Indexer.build(
        field_name=self.field_name,
        columnar_posting=contents,
        vocab=vocabulary,
        field_statistics=field_stats,
        doc_ids=doc_ids,
        index_dir=index_dir,          # ← 전달
    )
```

`index_dir=None`이면 기존 동작 그대로.

### 2-5. benchmark — rank_bm25 + 인라인 dense 제거

**파일**: `src/asmr/evaluation/stark_prime_benchmark.py`, `src/asmr/evaluation/stark_mag_benchmark.py`

`PrimeFieldIndexes` / `MagFieldIndexes`를 아래 패턴으로 교체한다.

```python
import tempfile
from pathlib import Path
from asmr.index.config import FieldConfig, TokenizerType, RepresentationType, IndexUsage
from asmr.index.fields import SparseTextFieldIndex, DenseTextFieldIndex

class PrimeFieldIndexes:
    field_names = list(PRIME_FIELD_NAMES)

    def __init__(self, corpus: PrimeCorpus, encoder: HfQueryEncoder, batch_size: int = 64) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._sparse: list[SparseTextFieldIndex | None] = []
        self._dense: list[DenseTextFieldIndex] = []

        for fname in self.field_names:
            texts = [field_text(doc, fname) for doc in corpus.documents]
            nonempty = any(t.strip() for t in texts)
            nonempty_texts = [t if t.strip() else fname for t in texts]

            # Sparse BM25 (디스크: tmpdir → GC 시 자동 삭제)
            sparse_cfg = FieldConfig(fname, TokenizerType.SPLIT, RepresentationType.SPARSE)
            sparse = SparseTextFieldIndex(sparse_cfg)
            if nonempty:
                idx_dir = Path(self._tmpdir.name) / f"sparse_{fname}"
                sparse.add_documents(corpus.doc_ids, texts, index_dir=idx_dir)
                self._sparse.append(sparse)
            else:
                self._sparse.append(None)

            # Dense FAISS (in-memory, save_index 미호출)
            # IndexUsage.BENCHMARK → faiss_index_path 검증 생략
            dense_cfg = FieldConfig(
                fname, TokenizerType.SPLIT, RepresentationType.DENSE,
                usage=IndexUsage.BENCHMARK,
            )
            dense = DenseTextFieldIndex(dense_cfg, encoder)
            dense.add_documents(corpus.doc_ids, nonempty_texts)
            self._dense.append(dense)

    def shortlist_hybrid(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        f_num = len(self.field_names)
        union: dict[str, np.ndarray] = {}

        for fi, fname in enumerate(self.field_names):
            if self._sparse[fi] is not None:
                for item in self._sparse[fi].search(query_text, k).items:
                    union.setdefault(item.doc_id, np.zeros((f_num, 2), dtype=np.float32))
                    union[item.doc_id][fi, 0] = max(union[item.doc_id][fi, 0], item.score)

            for item in self._dense[fi].search(query_text, k).items:
                union.setdefault(item.doc_id, np.zeros((f_num, 2), dtype=np.float32))
                union[item.doc_id][fi, 1] = max(union[item.doc_id][fi, 1], item.score)

        ...  # union → (doc_ids, scores[F,2,D], mask[F,D]) 변환 (기존 로직 유지)

    def close(self) -> None:
        self._tmpdir.cleanup()
```

`_topk()`, `rank_bm25` import, 인라인 dense numpy 배열 전부 제거.  
`rank_single_hybrid()` (single-field baseline) 도 동일 패턴으로 단순화.

---

## 3. 수정 대상 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/asmr/index/text_encoder.py` | `encoder: BaseFdeEncoder` → `encoder: TextEncoderProtocol` |
| `src/asmr/index/fields.py` | `DenseTextFieldIndex.encoder` 타입 교체; `SparseTextFieldIndex.add_documents()` `index_dir` 추가 |
| `src/asmr/index/config.py` | `IndexUsage` enum 추가; `FieldConfig.usage` 필드; `SERVING`일 때만 `faiss_index_path` 검증 |
| `src/asmr/evaluation/stark_prime_benchmark.py` | `PrimeFieldIndexes` 전면 교체 |
| `src/asmr/evaluation/stark_mag_benchmark.py` | `MagFieldIndexes` 전면 교체 |

**변경하지 않는 파일**:
- `DenseImageFieldIndex` — 이미지 인코더 전용, `BaseFdeEncoder` 유지
- `src/asmr/encode/protocol.py` — 기존 Protocol 정의 그대로
- `src/asmr/encode/implementations/hf.py` — `HfQueryEncoder` 무변경

---

## 4. 비호환 영향 범위

`TextEncodingIndexer(encoder: BaseFdeEncoder)` 호출부 전체가 영향을 받는다.

```bash
# 직접 BaseFdeEncoder 인스턴스를 TextEncodingIndexer에 넘기는 곳
grep -r "TextEncodingIndexer(" src/ --include="*.py"
grep -r "DenseTextFieldIndex(" src/ --include="*.py"
```

`BaseFdeEncoder`를 구현한 인코더는 `encode_text()` 메서드를 가지므로 `TextEncoderProtocol`도 자동 만족.  
런타임 동작 변화 없음.

---

## 5. 검증 계획

```bash
# 1. 타입·포맷 검사
cd asmr
ruff format src/ && ruff check --fix src/ && mypy src/

# 2. 단위 테스트 — synthetic corpus, 파일 I/O 없이
python -m pytest tests/unit/asmr/evaluation/test_field_indexes.py -v

# 3. 벤치마크 smoke (데이터 있을 때)
ASMR_BENCHMARK_TESTS=0 python -m pytest tests/system/test_stark_mag_mfarall_benchmark.py -v
```

단위 테스트에서 검증:
- `SparseTextFieldIndex.add_documents(index_dir=tmpdir)` BM25 top-k 결과
- `DenseTextFieldIndex(HfQueryEncoder)` in-memory FAISS search
- `shortlist_hybrid()` union 결과에 정답 doc_id 포함
