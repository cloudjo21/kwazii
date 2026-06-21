# RFC 001: STaRK-Amazon & STaRK-MAG 데이터셋 추가

**Status**: Proposed  
**Author**: cloudjo21  
**Date**: 2026-06-21  
**Branch**: feat/stark-amazon-mag-datasets

---

## 1. Motivation

`asmr` 프로젝트는 mFAR (Multi-Field Adaptive Retrieval, ICLR 2025) 논문의
벤치마크를 재현·확장하는 것을 목표로 한다. 현재 STaRK-Prime만 구현되어 있으며,
논문 Table 1의 STaRK-Amazon과 STaRK-MAG 실험을 진행하려면 두 데이터셋에 대한
`DatasetSchema` 구현이 필요하다.

참고 문서: `docs/research/compatible-benchmark-datasets.md` §2.1, §2.2

---

## 2. 데이터셋 명세

### 2.1 STaRK-Amazon

| 항목 | 내용 |
|------|------|
| 필드 수 | 8 |
| 필드 목록 | `also_buy`, `also_view`, `brand`, `description`, `feature`, `qa`, `review`, `title` |
| 코퍼스 출처 | Amazon Product Reviews (He & McAuley 2016) |
| 평가 지표 | Hit@1, Recall@20, MRR |

### 2.2 STaRK-MAG

| 항목 | 내용 |
|------|------|
| 필드 수 | 5 |
| 필드 목록 | `abstract`, `author___affiliated_with___institution`, `paper___cites___paper`, `paper___has_topic___field_of_study`, `title` |
| 코퍼스 출처 | Microsoft Academic Graph / OGBN-MAG |
| 평가 지표 | Hit@1, Recall@20, MRR |

---

## 3. 데이터 레이아웃 (data_root 기준)

```
data_root/
├── skb/
│   ├── amazon/corpus.json       # list[{id, title, brand, ...}]
│   └── mag/corpus.json          # list[{id, title, abstract, ...}]
└── qa/
    ├── amazon/
    │   ├── stark_qa/stark_qa.csv
    │   └── split/{train,val,test}.index
    └── mag/
        ├── stark_qa/stark_qa.csv
        └── split/{train,val,test}.index
```

---

## 4. 설계 결정

### 4.1 타입 정의 — Pydantic BaseModel

`python-rules.md`에 따라 Query/Corpus는 Pydantic BaseModel로 정의한다.
(stark_prime의 `@dataclass(frozen=True)`는 기존 코드를 수정하지 않고 그대로 유지)

```python
class AmazonQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_id: int
    query: str
    answer_ids: list[str]

class AmazonCorpus(BaseModel):
    model_config = ConfigDict(frozen=True)
    doc_ids: list[str]
    documents: list[dict[str, Any]]
```

MAG도 동일 패턴(`MagQuery`, `MagCorpus`).

### 4.2 field_text 재사용

`asmr.datasets.stark_prime.loader.field_text`는 str/int/float/list/dict를 모두
처리하므로 Amazon·MAG에서 그대로 import해 사용한다. 중복 구현하지 않는다.

### 4.3 auto-registration 패턴 유지

각 패키지 `__init__.py`에서 `register_dataset(STARK_*_SCHEMA)` 호출.
stark_prime과 동일.

---

## 5. 파일 구조

```
src/asmr/datasets/
├── stark_amazon/
│   ├── __init__.py
│   ├── schema.py       # StarkAmazonSchema, STARK_AMAZON_SCHEMA
│   └── loader.py       # AmazonQuery, AmazonCorpus, build/load functions
└── stark_mag/
    ├── __init__.py
    ├── schema.py        # StarkMagSchema, STARK_MAG_SCHEMA
    └── loader.py        # MagQuery, MagCorpus, build/load functions
```

---

## 6. 테스트 계획

```
tests/unit/asmr/datasets/
├── __init__.py
├── test_stark_amazon_schema.py
└── test_stark_mag_schema.py
```

각 테스트에서 파일 I/O 없이:
- `field_text()` 직렬화 검증
- `name`, `field_names` 값 검증
- `isinstance(schema, DatasetSchema)` Protocol 준수

---

## 7. 검증

```bash
uv run ruff format && uv run ruff check --fix && uv run mypy src/
uv run pytest tests/unit/asmr/datasets/ -v
python -c "
import asmr.datasets.stark_amazon
import asmr.datasets.stark_mag
from asmr.datasets import list_datasets
print(list_datasets())  # ['stark_amazon', 'stark_mag', 'stark_prime']
"
```
