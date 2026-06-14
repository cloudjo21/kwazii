# Image Ranking Inference Module (asmr/match)

## Goal

주어진 구조화 텍스트 1개와 이미지 URL 최대 20개를 받아 top-k 이미지를 랭킹하는 추론 모듈을 제공한다. 검색 모델(인코더)을 교체해도 동일 인터페이스로 재사용 가능하며, 배치 추론을 지원한다.

## Scope

- `asmr/src/asmr/match/` 하위 모듈 신규 개발
- 텍스트 쿼리 → 이미지 URL 20개 → top-k 정렬 결과 반환
- `DocumentRetriever.retrieve_with_aggregation_head` 중심 추론 로직
- 선택적 aggregation head(학습된 re-ranking 헤드) 지원
- 단일 쿼리 + 배치 쿼리 API 제공
- URL/로컬 경로 이미지 로딩
- 모델 교체 가능 설계 (`BaseFdeEncoder` 인터페이스 기반)

## Non-Goals

- GPU 상에서의 실제 통합 테스트 실행 (준비는 하되 skip)
- 이미지 인덱스 영속화(디스크 저장/로드)
- 학습(training) 로직
- BM25/sparse retrieval (이미지 랭킹에만 집중)

## Acceptance Criteria

- [ ] AC-1: `ImageRanker.rank(text, image_urls, k)` 호출 시 `len(result) == min(k, len(image_urls))`이고 `result[i].rank == i+1` 순서로 반환된다.
- [ ] AC-2: `ImageRanker.rank_batch(queries, k)` 호출 시 각 쿼리에 대한 결과 리스트를 반환하며 길이는 `len(queries)`와 동일하다.
- [ ] AC-3: `BaseFdeEncoder`를 구현한 임의의 인코더를 `ImageRanker` 생성자에 주입할 수 있어 모델 교체 시 `ImageRanker` 코드 변경이 불필요하다.
- [ ] AC-4: `text`가 비어있거나 `image_urls`가 빈 리스트일 때 `ValueError`를 발생시킨다.
- [ ] AC-5: `aggregation_head`가 제공될 때 `retrieve_with_aggregation_head`의 head 추론 경로를 실행한다.
- [ ] AC-6: 단위 테스트 line coverage ≥ 85%, branch coverage ≥ 75%.
- [ ] AC-7: `CandidateImageIndexer.build`는 이미지 URL 리스트로부터 `DocumentRetriever`를 동적으로 생성한다.

## Test Cases

| ID | Scenario | Input | Expected Output |
|----|----------|-------|-----------------|
| TC-1 | Happy path, no head | text="shirt", urls=[20 mock URLs], k=5 | 5개 RankResult, rank 1~5, score 내림차순 |
| TC-2 | k > len(urls) | text="shirt", urls=[3 mock URLs], k=10 | 3개 RankResult |
| TC-3 | Empty text | text="", urls=[...] | ValueError |
| TC-4 | Empty urls | text="shirt", urls=[] | ValueError |
| TC-5 | Batch ranking | queries=[(text1, urls1), (text2, urls2)] | 2개 결과 리스트 |
| TC-6 | With aggregation head | text, urls, head=MockHead | head 경로 호출 확인 |
| TC-7 | URL load failure | urls=["http://bad-url"] | ValueError |
| TC-8 | Single image | text="cat", urls=["img.jpg"], k=1 | 1개 RankResult |

## Out of Scope / Future Work

- 비동기 이미지 URL 병렬 로딩 (현재는 순차 로딩)
- 이미지 캐싱 레이어
- ONNX/CoreML 추론 경로
