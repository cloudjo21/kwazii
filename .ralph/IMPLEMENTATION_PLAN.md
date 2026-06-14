# Implementation Plan — Image Ranking Module (asmr/match)

## Architecture Decision

### New module: `asmr/src/asmr/match/`

DDD 계층 준수:
- `match/` = domain-level inference façade (retrieval infrastructure 위에 위치)
- Heavy dependencies (`torch`, `fde`, `asmr.index.*`) 는 모두 함수 내부에서 lazy import
- 인코더는 `BaseFdeEncoder` 인터페이스로 주입 (DI)

```
asmr/src/asmr/match/
├── __init__.py       # 공개 API: ImageRanker, RankResult, RankerConfig
├── config.py         # RankerConfig dataclass
├── indexer.py        # CandidateImageIndexer (ephemeral index builder)
└── ranker.py         # ImageRanker (main façade), BatchImageRanker
```

### Inference 흐름

```
text + image_urls
       ↓
CandidateImageIndexer.build()
  → load URLs (PIL)
  → DenseImageFieldIndex.add_documents() [ephemeral FAISS IndexFlatIP]
  → DenseImageFieldRetriever + QueryRouter + DocumentRetriever
       ↓
DocumentRetriever.retrieve_with_aggregation_head(query, k=N)
  → FAISS top-k by cosine similarity (text-to-image cross-modal)
  → [optional] apply_aggregation_head → logits sort
       ↓
RankResult list (url, rank, score)
```

### 모델 교체 가능성

`ImageRanker(encoder=MyEncoder())` — `BaseFdeEncoder` 구현체만 교체하면 됨.
`ranker.py` 는 `BaseFdeEncoder` 인터페이스에만 의존, 구체 모델 코드 없음.

### 배치 추론

`ImageRanker.rank_batch(queries)` → `asyncio.gather` 로 병렬 실행.
GPU 환경에서는 동시 인코딩이 실제 배치 효율 제공.

## Story → File Mapping

| Story | 파일 |
|-------|------|
| S1 | `.ralph/specs/feature.md` |
| S2 | `asmr/src/asmr/match/config.py`, `indexer.py`, `ranker.py`, `__init__.py` |
| S2 | `asmr/tests/unit/asmr/match/test_config.py`, `test_indexer.py`, `test_ranker.py` |
| S3 | `asmr/tests/integration/test_image_ranker_gpu.py` |

## Test Strategy

### Unit Tests (mypy311 conda, PYTHONPATH=asmr/src)

- 모든 외부 의존성(encoder, DocumentRetriever, FAISS, PIL 로딩) mock
- `conftest.py`에서 `kiwipiepy`, `muvfde` 모듈 mock (import chain 차단)
- Async 테스트: `@pytest.mark.asyncio`

커버리지 목표:
- `config.py`: 100% (simple dataclass)
- `indexer.py`: ≥85% (URL loading, build 경로)
- `ranker.py`: ≥85% (rank, rank_batch, error paths)

### Integration Tests (GPU, skip without GPU)

- `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` 마커
- 실제 encoder (JinaVera 또는 mock), 실제 FAISS, 실제 이미지 필요
- CI에서는 skip; GPU 서버에서만 실행

## Risk Flags

1. `FieldConfig` validation: `faiss_index_path=None`이면 ValidationError → workaround: `":ephemeral:"` 사용
2. `FieldBasedRanking.items` 이름 충돌 (Pydantic field vs method): pipeline.py가 `.items` attribute를 쓰므로 주의
3. `kiwipiepy` not installed in mypy311 → conftest.py에서 sys.modules mock 필수
4. `muvfde` not installed in mypy311 → 동일
5. torch 2.2.2 (mypy311) vs 2.6.0 (project): unit test에서 실제 torch 호출 없으므로 문제없음

## Dependency Order

S1 (spec) → S2 (impl + unit tests) → S3 (integration tests)
S2 내부: config.py → indexer.py → ranker.py (순서 의존)

PLANNING COMPLETE
