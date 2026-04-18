# ASMR: mFAR-style trainable retrieval 설계 (evidence composition-first)

구현 워크플로우·텐서 계약·벤치마킹 프로토콜: [trainable-retrieval-technical-detail-design.md](./trainable-retrieval-technical-detail-design.md)

이 문서는 asmr을 **학습 가능한 멀티필드 검색 시스템**으로 확장하기 위한 설계이다. 목표는 범용 단일 임베딩(BGE-M3 등)과의 **표현 품질 정면승부**가 아니라, 반정형·멀티필드 문제에 맞는 **귀납편향(inductive bias)** 과 **증거 조합(evidence composition)** 을 1급 시민으로 올리는 것이다.

---

## Terminology (용어 사전)

아래는 본 설계·[기술 상세 문서](./trainable-retrieval-technical-detail-design.md)에서 쓰는 **생소할 수 있는 용어**와 **역할 타입(role types)** 요약이다. 구현 세부(텐서 shape 등)는 기술 상세 쪽을 본다.

### 검색 파이프라인

- **Shortlist (후보 단축 목록)**  
  전체 코퍼스가 아니라, **필드별·스코어러별** 1차 검색에서 상위 **top-k**로 잘라 낸 **후보 문서 집합**이다. 문서 수가 크면 모든 `(질의, 문서)` 쌍에 대해 정밀 점수를 계산할 수 없으므로(mFAR 논문도 동일), 먼저 shortlist를 만든 뒤 그 안에서만 최종 점수·학습 헤드를 적용한다. 여러 필드/스코어러에서 각각 뽑은 후보의 **합집합**이 shortlist가 될 수 있다.

- **Corpus-wide retrieval (전 코퍼스 검색)**  
  인덱스 전체를 대상으로 하는 1차 검색. 비용·지연이 크므로 shortlist와 구분해서 말한다.

- **Aggregation / fusion (집계·융합)**  
  필드별·스코어러별 점수를 **문서 하나의 최종 점수**로 합치는 단계. 단순 합, 가중합, 또는 학습된 `G_θ`가 여기에 해당한다.

- **Re-ranking (재순위화)**  
  이미 뽑힌 후보 집합( shortlist ) 안에서 순서만 다시 매기는 단계. 무거운 cross-encoder 전 코퍼스 적용과 달리, 본 설계의 **작은 aggregation head**는 가벼운 re-rank에 가깝다.

### 멀티필드·mFAR

- **Field (필드)**  
  문서를 이루는 **이름 붙은** 텍스트(또는 값) 조각: 예) `title`, `abstract`, `brand`, `reviews`. 반정형 문서는 필드마다 성격이 다르다.

- **Scorer / scoring method (스코어러)**  
  한 필드에 대해 질의와의 관련도를 숫자로 낼 때 쓰는 **방법의 종류**. mFAR에서는 대표적으로 **lexical**(예: BM25)과 **dense**(예: 벡터 내적)를 필드마다 따로 쓰고, 기호 `M`으로 스코어러 축을 둔다.

- **Query conditioning / query-conditioned weights (쿼리 조건부 가중)**  
  필드·스코어러 가중치가 **질의 내용에 따라** 달라지는 것. “항상 title이 0.3”이 아니라, 이 질의에서는 `authors`가 더 중요해지는 식이다. mFAR에서 적응 함수 `G(q,f,m)`이 담당하며, **No query-conditioning** ablation은 이를 끈 비교 실험이다.

- **`G_θ` (composition head)**  
  단순 선형 가중합을 넘어, 필드 점수·질의 표현·(선택) 보조 특성을 받아 **문서 점수**를 내는 **작고 규제된** 학습 모듈(예: 얕은 MLP). “표현을 더 잘 만드는 거대 모델”이 아니라 **증거를 어떻게 조합할지**를 학습하는 층에 가깝다.

### 역할(role)·스키마

- **Surface field name (표면 필드명)**  
  데이터 스키마에 적힌 문자열 그대로(`summary` vs `description` 등). 도메인마다 의미가 달라질 수 있다.

- **Latent role / semantic role (잠재·의미 역할)**  
  표면 이름과 무관하게, 필드가 검색에서 맡는 **기능적 역할**을 뜻한다. 설계에서 예로 든 타입:

  - **entity-anchor**: 개체명·상품명처럼 “무엇”을 고정하는 필드  
  - **evidence-detail / topical-summary**: 서술·근거·요약 본문  
  - **constraint / filter**: 날짜·가격·카테고리처럼 **조건·필터**로 쓰이는 값  
  - **trust / provenance**: 출처·평판·메타데이터로 **신뢰**를 보조하는 필드  

  필드 하나가 여러 역할에 **부분적으로** 속할 수 있으며, 이를 **soft assignment**(연속적인 혼합 계수)로 두는 것이 설계상 목표다.

- **Schema-grounded (스키마 접지)**  
  “비슷해 보인다”가 아니라 **타입이 맞는가, 제약을 만족하는가** 같은 **스키마·제약**이 관련도 판단에 직접 들어가는 관점.

- **Neuro-symbolic (신경·상징 하이브리드)**  
  일부 필드는 신경망 유사도, 일부는 규칙·심볼릭 제약(날짜 범위 등)으로 점수화하고 **합치는** 방식.

### 일관성·보조 특성

- **Cross-field consistency (필드 간 일관성)**  
  여러 필드의 점수·텍스트가 **같은 이야기를 하는지**에 대한 신호. 초기 구현에서는 필드 점수 **분산**, **coverage**(질의 단서가 여러 필드에서 지지되는지) 등 저비용 통계로 근사할 수 있다.

- **Aux features (보조 특성)**  
  원 필드 점수 스칼라 외에 `G_θ`에 넣는 **추가 입력**(통계·마스크·coverage 등). `asmr/train/features.py`의 `build_aux_features`가 생성하는 벡터가 여기에 해당한다.

### 학습·데이터

- **Teacher / substrate (교사 / 기질)**  
  - **Teacher**: 학생 모델이 맞추려는 **더 강한** 점수·분포(예: BGE-M3의 dense+sparse+multi-vec 앙상블, cross-encoder).  
  - **Substrate**: 그 아래에서 필드 텍스트를 벡터로 바꾸는 **고정 또는 사전학습 인코더**(BGE-M3를 필드 인코더로 쓰는 경우).

- **Inductive bias (귀납편향)**  
  데이터만으로는 안 드러나는 **선험적 구조 가정**(여기서는 “문서는 필드로 나뉘고, 질의는 그중 일부 조합으로 답이 성립한다”).

- **Hard negatives (어려운 부정 예)**  
  무작위 오답이 아니라 **일부만 그럴듯한** 오답(제목만 맞고 본문은 틀림, 제약은 틀림 등). 멀티필드에서 모델을 압박하는 샘플링 전략이다.

- **Weak supervision (약한 감독)**  
  사람 정답이 아니라 규칙·BM25·LLM 등으로 만든 **노이즈 있을 수 있는** 라벨.

- **Field-level supervision (필드 단위 감독)**  
  문서 전체가 관련 있다/없다가 아니라, **어느 필드가 근거·오해·무관**인지에 대한 신호.

### 평가

- **Slice (슬라이스)**  
  전체 벤치를 한 번에 보지 않고, 질의 **부분집합**(예: 메타데이터가 중요한 질의만)으로 나눈 평가 구간.

- **Failure taxonomy (실패 유형 분류)**  
  평균 점수만이 아니라, **어떤 종류의 오답**이 나는지 나눈 범주(제목 미끼에 걸림, 필드 불일치 등).

### 데이터셋·모델 약어

- **STaRK**  
  반정형 멀티필드 검색 벤치(예: Amazon, MAG, Prime). mFAR 논문의 주요 실험 무대이다.

- **mFAR / MFAR**  
  Multi-Field Adaptive Retrieval — 필드 분해 + (쿼리 조건부) 가중 + hybrid 스코어러.

- **BGE-M3 (M3-Embedding)**  
  다국어·다기능(dense/sparse/multi-vector)·장문을 한 패밀리로 쓰는 **범용** 임베딩. 본 설계에서는 “범용 임베딩 품질만으로 멀티필드 구조 문제를 끝낸다”가 아니라, 필요 시 **teacher/substrate**로 **같이** 쓰는 것을 전제로 한다.

---

## 1. 철학: representation-first → evidence composition-first

| 축 | 범용 foundation retriever (예: BGE-M3) | 본 설계에서의 asmr |
|----|----------------------------------------|---------------------|
| 핵심 질문 | 텍스트 의미를 잘 압축했는가? | 질의가 요구하는 **증거가 어느 필드·역할·제약 조합으로 성립하는가?** |
| 승부처 | 통합 임베딩·다기능 스코어 | 필드 간 **논리적 결합**, 스키마 만족, 필드 일관성, 필드 단위 감독 |

mFAR류의 강점(필드 분해, 필드별 sparse/dense, 쿼리 조건부 가중)을 유지하되, 아래 한계를 설계로 명시적으로 극복한다.

---

## 2. 현재 asmr 기준선 (코드 매핑)

| 구성요소 | 역할 | 확장 포인트 |
|----------|------|-------------|
| `retrieve/retrievers.py` — `QueryRouter` | 필드명 → 해당 `BaseFieldRetriever` | 필드별 (sparse/dense/이미지) 후보 생성은 **고정 파이프라인**으로 유지 가능 |
| `retrieve/aggregate.py` — `aggregate_field_scores_hybrid_async` | 필드별 lexical/dense top-k를 모아 `(doc_ids, scores[F, 2, D])` 행렬 | **최종 문서 점수로 가기 직전 텐서** — trainable head의 입력; `[F, 2, D]`의 M축이 lex/dense 분리 |
| `retrieve/helpers.py` — `DocumentRetriever` | 전체/스마트 필드 선택 후 aggregate 호출 | **학습 모드**에서 감독·배치 루프 진입점 |
| `index/config.py` — `FieldConfig` | 필드명, 토크나이저, sparse/dense | **스키마 타입·역할 힌트·심볼릭 제약 채널** 메타데이터 확장 |
| 통합 테스트 `scoring_helper` | 필드 점수 합산 | **baseline** — 학습 시에는 합산 대신 `G_θ` 출력으로 대체 |

즉, “학습 가능”의 첫 단계는 **필드×문서 점수 행렬 위에 얹는 작고 규제된 aggregation·composition 모듈**이다. 무거운 cross-encoder를 1차 검색 전체에 얹기보다, **후보 집합 위의 가벼운 head**로 시작한다.

---

## 3. mFAR가 가진 한계와 asmr에서의 대응

### 한계 A: 필드를 가중합으로만 본다 → **조합 함수(composition) 학습**

단순 합:

$$
s(q,d) \approx \sum_{f,m} w_{q,f,m}\, s_{f,m}(q,d)
$$

목표 형태:

$$
s(q,d) = G_\theta\big(\{s_{f,m}\}, \phi(q), \psi(f), \chi(d), \text{aux features}\big)
$$

- `G_θ`: 작은 MLP / 저차원 bilinear / gated fusion 등 **파라미터 수 상한**을 두는 aggregator.
- 입력은 스칼라 점수만이 아니라 아래 **aux**를 포함 (과적합 방지를 위해 차원·드롭아웃·L2 규제 명시).

### 한계 B: 필드 의미가 정적 → **latent role (역할 공간)**

스키마 표면 문자열(`title`, `headline`, …)에 직접 의존하지 않고, 사전 정의 또는 학습 가능한 **역할 타입** (예: entity-anchor, evidence-detail, constraint/filter, trust/provenance)에 각 필드를 soft-assign.

- 구현: 필드 임베딩 `ψ(f)` ∈ ℝ^r, 역할 프로토타입과의 유사도로 혼합 계수 생성.
- 효과: 도메인이 바뀌어도 **역할 공간에서 정렬**되어 일반화.

### 한계 C: 필드 내부 반정형 구조 → **schema-grounded 채널 분리**

일부 필드는 “의미 유사도”가 아니라 **타입·범위·시간·카테고리 만족**이 본질이다.

- `FieldConfig` (또는 별도 스키마 레지스트리)에 `semantic | lexical | typed_constraint | provenance` 등 **채널 태그**를 둔다.
- `G_θ`는 채널별 스코어를 **같은 합으로 섞지 않고** 게이트/전용 소규모 서브헤드로 결합 (방향 C: neuro-symbolic 하이브리드의 설계적 자리).

### 한계 D: 필드 간 일관성 미반영 → **cross-field consistency 특성**

필드별 점수 외에, 후보 `(q,d)`마다 계산 가능한 **비학습 또는 저비용** 특성:

- 필드 점수 분산·최대/최소 비율
- 상위 필드 집합의 coverage (질의 토큰·엔티티가 몇 필드에서 지지되는지)
- (가능 시) 필드 텍스트 간 entailment/contradiction 스코어 — 초기에는 휴리스틱·소형 모델
- **must-have 필드 결손** 패널티 (스키마·규칙 또는 학습된 바이너리 마스크)

이들을 `G_θ` 입력 벡터에 concat.

---

## 4. 제안 통합: 데이터·학습·평가

### 4.1 감독 신호 (우선순위 1)

문서 단위 `(q, d+, d-)`만으로는 부족. 최소한 다음을 지원할 **데이터 모델**을 정의한다.

- positive / irrelevant / misleading **필드 레벨** 라벨 (weak label 허용)
- (선택) span·필드 내 핵심 구간
- teacher: cross-encoder attribution, ensemble distillation, RAG 인용 역추적 등

**손실 (다목적):**

- 문서 ranking (pairwise/listwise)
- 필드 선택 / 필드 relevance (multi-label)
- 필드 근거 distillation (teacher와의 KL 또는 MSE)
- 모순·제약 위반 샘플에 대한 **거절(rejection)** 손실

### 4.2 Hard negative 설계 (우선순위 1과 연동)

체계적으로 생성·오버샘플링:

- single-field lure (제목만 맞음)
- 스키마 호환·의미 오류
- 의미 유사·제약 불만족
- 필드 순서/역할 뒤바뀜
- 다필드 일부 강한데 핵심 필드 충돌

### 4.3 BGE-M3 (또는 동급)의 위치 (우선순위 5 — 실용)

- **적이 아니라 substrate/teacher:** dense·sparse·multi-vector 중 필요한 채널을 필드 인코더로 사용.
- asmr 고유층: **필드 역할 · 조합 · 스키마 · 일관성** — “더 큰 통합 임베딩”을 새로 만드는 것이 아니라 **구조적 의사결정 층**에 자원 집중.

### 4.4 평가 (leaderboard 평균만이 아님)

slice 예시:

- 정답 근거가 소수 필드에만 있는 질의
- conjunction이 필요한 질의
- 메타데이터/제약 중요 질의
- title bait / review noise
- 다국어·혼합 스키마

**실패 분류(failure taxonomy)** 를 정의하고, 평균 nDCG와 함께 slice별 표를 1급 지표로 둔다.

---

## 5. 아키텍처 방향 (구현 시 선택지)

다음은 논의된 방향 A–D를 asmr 모듈 경계에 대응시킨 것이다.

### 방향 A: Hierarchical Field Reasoning

1. 필드별 후보 증거 검색 (현 `QueryRouter` + 인덱스)
2. 쿼리 조건부 **역할(role) 추론** (경량 네트워크)
3. **cross-field evidence composition** — 그래프형 일관성 스코어 (초기에는 fully-connected pairwise 특성 + `G_θ`로 근사 가능)

### 방향 B: Role-conditioned Distilled mFAR (현실적 1차 목표)

- Base: BGE-M3 등으로 필드별 벡터·스코어 생산
- `FieldRoleProjector`: 필드 → latent role
- `AggregationHead`: `G_θ`
- Teacher: cross-encoder + 필드 attribution + 앙상블 distillation

### 방향 C: Schema-aware neuro-symbolic

- 일부 필드는 neural match, 일부는 symbolic/typed constraint 스코어
- `G_θ` 또는 규칙 기반 게이트로 최종 융합

### 방향 D: Multi-view counterfactual training

- 동일 문서에 대해 필드 마스킹·순열·모순 주입 등 aug → **어떤 필드가 필수였는지** 학습

---

## 6. 코드 레이아웃

기존 `retrieve` / `index`를 깨지 않고 확장한다. PyTorch는 `asmr[train]` extra 또는 루트 `kwazii` 환경에서 설치.

```
asmr/
  train/                          # 구현: src/asmr/train/
    __init__.py
    config.py                     # 학습 하이퍼파라미터, 필드 role 차원, 손실 가중치
    data.py                       # base Dataset / RankingBatch 정의
    data_stark.py                 # StarkRankingDataset, collate_ranking_batch
    features.py                   # consistency·disagreement·schema aux 계산
    aggregation.py                # G_θ, MFARFieldAdapter
    losses.py                     # ranking + field + distillation + rejection
    trainer.py                    # AggregationTrainer (학습 루프)
    train_script.py               # 진입점: 옵티마이저, 체크포인트
    __main__.py                   # python -m asmr.train 진입
    query_encoder.py              # HfQueryEncoder — query_emb [B, H] 생산
    inference.py                  # apply_aggregation_head 등 추론 유틸
  retrieve/
    aggregate.py                  # aggregate_field_scores_hybrid_async → [F, 2, D]
    helpers.py                    # retrieve_with_aggregation_head — optional head 재정렬
```

- **추론 경로:** `aggregate_field_scores_hybrid_async`로 `[F, 2, D]` 점수 행렬 획득 → `retrieve_with_aggregation_head`에 `aggregation_head` + `query_encoder` 주입 시 `G_θ`로 재정렬; 미주입 시 aggregate-only.
- **학습 경로:** 동일 텐서를 배치로 모아 손실 계산.

---

## 7. 구현 우선순위 (연구·엔지니어링 공통)

1. **필드 레벨 감독 + hard negative 파이프라인** — 구조보다 데이터·평가가 먼저.
2. **가중합 → `G_θ` composition** — `aggregate` 출력에 붙는 작은 head.
3. **Latent field role** — 필드 임베딩·soft assignment.
4. **Schema-aware / symbolic 채널** — 타입 필드 분리 스코어.
5. **BGE-M3 등 teacher/substrate** — 필드 인코더·distillation.

---

## 8. 비판적 주의 (의도적으로 문서에 명시)

- 범용 임베딩을 “따라잡으려” **거대 단일 표현기**를 asmr 안에 복제하면, BGE-M3가 이미 잘하는 게임을 중복한다.
- 본 프로젝트의 차별점은 **구조적 증거 조합·스키마·필드 일관성·필드 단위 감독·오답 구조 학습**에 있다.

---

## 9. 한 줄 요약

**asmr의 trainable 층은 “더 나은 범용 임베딩”이 아니라, “정답이 어떤 필드 증거 조합으로 성립하는가”를 학습하는 evidence composition 엔진이다.** 구현상으로는 `aggregate`가 만든 필드×문서 점수 위에 **역할·스키마·일관성**을 넣은 작은 `G_θ`와 **필드 단위 감독**을 얹는 것에서 시작한다.
