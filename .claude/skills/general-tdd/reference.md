# TDD (Technical Design Document) Reference Guide

소프트웨어 컴포넌트 TDD 작성 시 참고하는 상세 가이드입니다.
Agent 특화 내용 없이 범용적으로 사용할 수 있습니다.

---

## 1. TDD 작성 원칙

### 1.1 명확성 (Clarity)

- 모호한 표현 대신 구체적인 용어 사용
- 약어 사용 시 반드시 정의 섹션에 명시
- 다이어그램과 예시 코드를 적극 활용
- 수동태 대신 능동태로 서술

### 1.2 완전성 (Completeness)

- 모든 public interface와 상호작용을 빠짐없이 문서화
- 정상 케이스뿐 아니라 에러 케이스, 엣지 케이스 포함
- 의존성과 전제조건 명시
- 분석 불가한 항목은 `[TODO: ...]`로 명시하고 비워두지 않음

### 1.3 일관성 (Consistency)

- 동일한 개념에 동일한 용어 사용 (혼용 금지)
- 문서 전체에서 일관된 포맷 유지
- 버전 관리와 변경 이력 기록

### 1.4 추적가능성 (Traceability)

- 요구사항 ↔ 설계 ↔ 코드 간 연결고리 명시
- 코드 위치(파일 경로, 라인 번호) 정확히 기재
- 관련 문서 및 참조 자료 링크

---

## 2. 섹션별 작성 가이드

### 2.1 Document Overview

**Purpose 작성 요령:**
- "이 문서는 `[컴포넌트명]`의 ~를 위해 작성되었습니다" 형식
- 대상 독자 명확히 지정 (예: 백엔드 개발자, QA 엔지니어, DevOps)

**Scope 작성 요령:**
- 포함: 이 문서에서 설계하고 명세하는 내용
- 제외: 혼동 가능한 인접 영역 (예: "인증 서버 내부 구현은 다루지 않음")

**Definitions 예시:**
| Term | Definition |
|------|------------|
| TDD | Technical Design Document |
| NFR | Non-Functional Requirements |
| SLA | Service Level Agreement |
| P95 | 95th Percentile Latency |
| RBAC | Role-Based Access Control |

---

### 2.2 Architecture Diagram

ASCII 다이어그램 작성 시 아래 문자를 활용합니다:

```
┌──────┐   박스 상단
│      │   박스 측면
└──────┘   박스 하단
───────    수평선
│          수직선
├──        왼쪽 분기
→  ←       방향 화살표
◆          다이아몬드 (의사결정)
○          원 (시작/종료)
```

**레이어 다이어그램 예시:**
```
┌──────────────────────────────────────┐
│            Client Layer              │
│   Web Browser / Mobile / CLI / SDK   │
└──────────────────┬───────────────────┘
                   │ HTTPS
┌──────────────────▼───────────────────┐
│            API Gateway               │
│     Rate Limit / Auth / Routing      │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│          Application Layer           │
│  ┌─────────────┐  ┌───────────────┐  │
│  │   Service A │  │   Service B   │  │
│  └─────────────┘  └───────────────┘  │
└──────────────────────────────────────┘
         │                  │
┌────────▼────┐   ┌─────────▼──────┐
│  Database   │   │  External API  │
└─────────────┘   └────────────────┘
```

---

### 2.3 Data Models

**Python (Pydantic) 스키마 문서화 예시:**
```python
class CreateOrderRequest(BaseModel):
    """주문 생성 요청"""

    product_id: str = Field(
        ...,
        description="상품 ID",
        pattern=r"^[A-Z0-9]{8}$",
        examples=["PROD0001"]
    )
    quantity: int = Field(
        ...,
        ge=1,
        le=100,
        description="수량"
    )
    user_id: str = Field(..., description="사용자 ID")
```

**TypeScript (Zod) 스키마 문서화 예시:**
```typescript
const CreateOrderSchema = z.object({
  productId: z.string().regex(/^[A-Z0-9]{8}$/),
  quantity: z.number().int().min(1).max(100),
  userId: z.string().uuid(),
});
```

**스키마 → 테이블 변환:**
| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| product_id | string | Yes | - | pattern: `^[A-Z0-9]{8}$` | 상품 ID |
| quantity | integer | Yes | - | 1 ≤ x ≤ 100 | 수량 |
| user_id | string | Yes | - | UUID format | 사용자 ID |

---

### 2.4 API Specifications

**RESTful API 문서화 원칙:**
- Path는 명사, 복수형 사용 (`/users`, `/orders`)
- 동작은 HTTP 메서드로 표현 (GET, POST, PUT, DELETE, PATCH)
- 버전은 Path prefix로 관리 (`/api/v1/`)

**HTTP Status Code 가이드:**
| Code | Meaning | 사용 시점 |
|------|---------|---------|
| 200 | OK | 조회, 수정 성공 |
| 201 | Created | 리소스 생성 성공 |
| 204 | No Content | 삭제 성공, 응답 바디 없음 |
| 400 | Bad Request | 입력값 오류 |
| 401 | Unauthorized | 인증 필요 |
| 403 | Forbidden | 권한 없음 |
| 404 | Not Found | 리소스 없음 |
| 409 | Conflict | 중복, 충돌 |
| 422 | Unprocessable Entity | 검증 실패 (Pydantic 기본) |
| 429 | Too Many Requests | Rate limit 초과 |
| 500 | Internal Server Error | 서버 내부 오류 |
| 502 | Bad Gateway | 업스트림 오류 |
| 504 | Gateway Timeout | 업스트림 타임아웃 |

---

### 2.5 Error Handling

**에러 분류 체계:**
```
Error
├── Client Error (4xx) — 클라이언트 수정이 필요한 오류
│   ├── Validation Error (400)
│   ├── Authentication Error (401)
│   ├── Authorization Error (403)
│   └── Not Found (404)
└── Server Error (5xx) — 서버 측 문제
    ├── Internal Error (500)
    ├── External API Error (502)
    └── Timeout (504)
```

**재시도 전략 결정 트리:**
```
오류 발생
├── Retryable? (네트워크, 5xx)
│   └── Yes → Exponential Backoff with Jitter
│             → Max 3회, 타임아웃 30초
└── Not Retryable? (4xx, 비즈니스 오류)
    └── 즉시 에러 반환
```

**Exponential Backoff 공식:**
```
delay = min(base_delay * (2 ^ attempt) + random_jitter, max_delay)

예: base=1s, max=30s, jitter=0~1s
- 1회: 1s + jitter
- 2회: 2s + jitter
- 3회: 4s + jitter
```

---

### 2.6 Performance

**성능 지표 정의:**
| Metric | Definition | Unit |
|--------|------------|------|
| Latency P50 | 50번째 백분위 응답 시간 (중앙값) | ms |
| Latency P95 | 95번째 백분위 응답 시간 | ms |
| Latency P99 | 99번째 백분위 응답 시간 | ms |
| Throughput | 초당 처리 요청 수 | req/s |
| Error Rate | 전체 요청 중 5xx 비율 | % |
| Availability | 서비스 가용 시간 비율 | % |

**성능 목표 수립 기준:**
- SLA 기반: 계약된 SLA에서 역산
- 경험치 기반: 일반적으로 P95 < 500ms, P99 < 2s
- 부하 테스트 결과 기반: 실측값으로 설정

**캐싱 전략:**
| 전략 | 적합한 데이터 | TTL 가이드 |
|------|-------------|-----------|
| Cache-Aside | 읽기 빈도 높고 변경 적은 데이터 | 5분~1시간 |
| Write-Through | 일관성 중요 | 데이터 변경 시 즉시 갱신 |
| Cache-Around | 계산 비용 높은 결과 | 결과 유효 기간 |
| No Cache | 실시간성 필수, 개인화 | - |

---

### 2.7 Security

**보안 체크리스트:**
- [ ] 모든 외부 입력값 검증 (길이, 타입, 형식)
- [ ] SQL Injection 방어 (Parameterized Query, ORM 사용)
- [ ] XSS 방어 (출력 시 HTML 인코딩)
- [ ] 민감 정보 로깅 제외 (비밀번호, 토큰, PII)
- [ ] API 키/시크릿 환경변수 또는 Secret Manager로 관리
- [ ] HTTPS 강제 (HTTP → HTTPS 리다이렉트)
- [ ] CORS 정책 설정
- [ ] Rate Limiting 적용
- [ ] 보안 헤더 설정 (HSTS, CSP, X-Frame-Options 등)

**OWASP Top 10 대응 체크:**
| 취약점 | 대응 방법 |
|--------|---------|
| Broken Access Control | RBAC, 리소스 소유자 검증 |
| Cryptographic Failures | TLS 1.2+, AES-256 |
| Injection | Parameterized Query, Input Validation |
| Insecure Design | Threat Modeling, Secure Design Principles |
| Security Misconfiguration | 최소 권한 원칙, 보안 헤더 |
| Vulnerable Components | 의존성 취약점 스캔 (Dependabot, Snyk) |
| Auth Failures | MFA, 세션 관리, 토큰 만료 |
| Integrity Failures | 서명 검증, SBOM |
| Logging Failures | 감사 로그, 이상 탐지 |
| SSRF | 허용된 도메인 Allowlist |

---

### 2.8 Testing Strategy

**테스트 피라미드 비율:**
```
        /\
       /  \      E2E Tests (~10%)
      /    \     - 핵심 사용자 시나리오만
     /──────\
    /        \   Integration Tests (~20%)
   /          \  - 컴포넌트 간 인터페이스
  /────────────\
 /              \ Unit Tests (~70%)
/                \ - 모든 비즈니스 로직
──────────────────
```

**각 테스트 레벨 가이드:**

| Level | 목적 | 범위 | 속도 | 비용 |
|-------|------|------|------|------|
| Unit | 개별 함수/클래스 검증 | 단일 모듈 | 매우 빠름 | 낮음 |
| Integration | 컴포넌트 간 연동 검증 | 여러 모듈 | 보통 | 보통 |
| E2E | 사용자 시나리오 검증 | 전체 시스템 | 느림 | 높음 |
| Performance | 성능 요구사항 검증 | 시스템 전체 | 느림 | 높음 |

**좋은 테스트 케이스 작성 원칙 (AAA):**
```python
def test_create_order_success():
    # Arrange: 테스트 데이터와 의존성 준비
    product = create_test_product(price=10000)
    request = CreateOrderRequest(product_id=product.id, quantity=2)

    # Act: 테스트 대상 실행
    result = order_service.create(request, user_id="user-1")

    # Assert: 결과 검증
    assert result.total_price == 20000
    assert result.status == OrderStatus.PENDING
```

**테스트 케이스 명명 규칙:**
```
test_[단위]_[시나리오]_[기대결과]

예:
- test_create_order_with_valid_input_returns_order_id
- test_create_order_with_out_of_stock_raises_error
- test_get_user_with_nonexistent_id_returns_404
```

---

### 2.9 Dependencies 분석 방법

**언어별 의존성 파일:**
| Language | 파일 |
|----------|------|
| Python | `pyproject.toml`, `requirements.txt`, `Pipfile` |
| Node.js | `package.json`, `yarn.lock`, `pnpm-lock.yaml` |
| Go | `go.mod`, `go.sum` |
| Rust | `Cargo.toml`, `Cargo.lock` |
| Java | `pom.xml`, `build.gradle` |

**버전 명시 규칙 (Python):**
```
==1.2.3   # 정확한 버전 (프로덕션 권장)
~=1.2.3   # 1.2.x 허용 (패치 버전만 허용)
>=1.2.3   # 최소 버전
>=1.2,<2  # 범위 지정
```

**의존성 리스크 평가 기준:**
| 리스크 | 설명 |
|--------|------|
| High | 유지보수 중단, 알려진 취약점, 라이선스 문제 |
| Medium | 빈번한 Breaking Change, 불안정한 API |
| Low | 활발한 유지보수, 안정적인 API |

---

### 2.10 Deployment & Operations

**환경 변수 분류:**
| Category | 예시 | 관리 방법 |
|----------|------|---------|
| Required | `DATABASE_URL`, `API_KEY` | 환경별 설정 필수 |
| Optional | `LOG_LEVEL=INFO`, `TIMEOUT=30` | 기본값 제공 |
| Secret | `DB_PASSWORD`, `JWT_SECRET` | Secret Manager 사용 |
| Feature Flag | `FEATURE_NEW_UI=true` | 코드 내 기본값 또는 Feature Flag 서비스 |

**로깅 가이드:**
```python
# 좋은 예: 구조화된 로그
logger.info("Order created", extra={
    "order_id": order.id,
    "user_id": user.id,
    "total": order.total_price,
    "request_id": ctx.request_id,
})

# 나쁜 예: 비구조화 로그
print(f"Order {order.id} created for user {user.id}")
```

**Observability 3대 요소:**
| 요소 | 도구 예시 | 목적 |
|------|---------|------|
| Metrics | Prometheus, Datadog | 시스템 상태 수치화 |
| Logs | ELK Stack, CloudWatch | 이벤트 기록 |
| Traces | Jaeger, Zipkin, OpenTelemetry | 요청 흐름 추적 |

---

## 3. 문서 품질 체크리스트

### 3.1 작성 완료 후 Self-Review

- [ ] 모든 섹션이 채워졌는가? (또는 `[TODO]`로 명시되었는가?)
- [ ] 코드 파일 경로가 실제로 존재하는가?
- [ ] 다이어그램이 현재 구현을 정확히 반영하는가?
- [ ] 에러 케이스가 빠짐없이 문서화되었는가?
- [ ] 용어가 문서 전체에서 일관성 있게 사용되었는가?
- [ ] 외부 참조 링크가 유효한가?
- [ ] 성능 목표가 측정 가능한 형태로 명시되었는가?
- [ ] 보안 요구사항이 구체적으로 기술되었는가?
- [ ] Open Questions 섹션에 미결 사항이 기록되었는가?

### 3.2 리뷰어 체크 항목

- [ ] 요구사항과 설계가 일치하는가?
- [ ] 아키텍처 결정에 합리적인 근거가 있는가?
- [ ] 보안 요구사항이 충족되었는가?
- [ ] 성능 목표가 현실적이고 달성 가능한가?
- [ ] 테스트 전략이 충분한 커버리지를 보장하는가?
- [ ] 운영 고려사항(모니터링, 알림, 배포)이 충분한가?
- [ ] 의존성 리스크가 식별되고 대응 방안이 있는가?

---

## 4. 언어/프레임워크별 특화 가이드

### 4.1 Python / FastAPI

**디렉토리 구조 예시:**
```
src/
├── {component}/
│   ├── __init__.py
│   ├── router.py          # HTTP 라우터 (FastAPI)
│   ├── service.py         # 비즈니스 로직
│   ├── repository.py      # 데이터 접근
│   ├── schemas.py         # Pydantic 모델
│   ├── models.py          # DB 모델 (SQLAlchemy)
│   ├── dependencies.py    # DI 의존성
│   └── exceptions.py      # 커스텀 예외
└── tests/
    ├── unit/
    └── integration/
```

**FastAPI 엔드포인트 분석 포인트:**
- `@router.get/post/put/delete` 데코레이터
- `response_model` 파라미터
- `Depends()` 의존성 주입
- `status_code` 기본값

### 4.2 Node.js / TypeScript

**디렉토리 구조 예시:**
```
src/
├── {component}/
│   ├── index.ts           # 진입점, 모듈 export
│   ├── controller.ts      # HTTP 컨트롤러
│   ├── service.ts         # 비즈니스 로직
│   ├── repository.ts      # 데이터 접근
│   ├── dto/               # Data Transfer Objects
│   │   ├── request.dto.ts
│   │   └── response.dto.ts
│   └── types.ts           # 타입 정의
└── __tests__/
    ├── unit/
    └── integration/
```

### 4.3 Go

**디렉토리 구조 예시:**
```
internal/
└── {component}/
    ├── handler.go         # HTTP 핸들러
    ├── service.go         # 비즈니스 로직
    ├── repository.go      # 데이터 접근
    ├── model.go           # 도메인 모델
    └── handler_test.go    # 테스트
```

---

## 5. 참고 자료

### 5.1 표준 / 방법론

- [RFC 2119](https://tools.ietf.org/html/rfc2119) — 요구사항 수준 표기 (MUST, SHOULD, MAY)
- [IEEE 830](https://standards.ieee.org/ieee/SRS/1921/) — Software Requirements Specification
- [ISO/IEC/IEEE 26512](https://www.iso.org/standard/80585.html) — System Documentation
- [C4 Model](https://c4model.com/) — 소프트웨어 아키텍처 다이어그램 방법론
- [ADR (Architecture Decision Records)](https://adr.github.io/) — 아키텍처 결정 기록

### 5.2 외부 참고 자료

- [Google Technical Writing](https://developers.google.com/tech-writing) — 기술 문서 작성 가이드
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) — 웹 애플리케이션 보안
- [OpenAPI Specification](https://swagger.io/specification/) — API 문서화 표준
- [SRE Book](https://sre.google/sre-book/table-of-contents/) — SLI/SLO/SLA 개념
- [The Twelve-Factor App](https://12factor.net/) — 클라우드 네이티브 앱 설계 원칙
