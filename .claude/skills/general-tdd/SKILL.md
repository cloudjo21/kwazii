---
name: general-tdd
description: 소프트웨어 컴포넌트(서비스, 모듈, API, 라이브러리 등)의 TDD(Technical Design Document) 문서를 작성합니다. 컴포넌트 이름을 인자로 받아 코드베이스를 분석하고 체계적인 기술 설계 문서를 생성합니다.
argument-hint: "[component-name] [--output path/to/output.md]"
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Write, Edit, Bash
---

# General Technical Design Document Generator

$ARGUMENTS 로 전달받은 소프트웨어 컴포넌트의 TDD(Technical Design Document)를 작성합니다.

## 인자 파싱

- `$ARGUMENTS` 첫 번째 토큰: 컴포넌트 이름 (필수)
- `--output <path>`: 출력 파일 경로 (선택, 기본값: `docs/tdd/<component-name>-tdd.md`)

## 사전 조건

1. `$ARGUMENTS`로 컴포넌트 이름이 전달되어야 합니다
2. 현재 작업 디렉토리에서 분석 가능한 코드가 존재해야 합니다

---

## 실행 절차

### Step 1: 컴포넌트 탐색

1. Glob으로 컴포넌트 관련 디렉토리/파일 탐색
   - `**/{component-name}/**`, `**/*{component-name}*` 패턴으로 검색
2. 프로젝트 루트의 패키지 설정 파일 확인
   - `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `pom.xml` 등
3. 진입점(entry point) 파악
   - `main.py`, `index.ts`, `main.go`, `App.java` 등

### Step 2: 코드 구조 분석

1. 디렉토리 트리 파악 (최대 3 depth)
2. 핵심 파일 읽기
   - 인터페이스/추상 클래스 정의
   - 주요 클래스/함수
   - 설정 파일
   - 스키마/모델 정의
3. 의존성 분석
   - import 문, require 문, use 구문 등
4. API/인터페이스 파악
   - HTTP 엔드포인트, gRPC 서비스, CLI 인터페이스 등
5. 테스트 코드 분석
   - 테스트 파일 위치, 테스트 전략 파악

### Step 3: TDD 문서 생성

아래 **TDD 문서 템플릿**에 따라 실제 코드를 기반으로 문서를 작성합니다.
분석한 내용으로 채울 수 없는 섹션은 `[TODO: ...]` 로 표시합니다.

---

## TDD 문서 템플릿

```markdown
# [Component Name] Technical Design Document

## Document Information

| Item | Description |
|------|-------------|
| Document Version | 1.0 |
| Created Date | YYYY-MM-DD |
| Last Updated | YYYY-MM-DD |
| Author | Claude Code |
| Status | Draft |
| Component | [component-name] |
| Language / Runtime | [e.g., Python 3.13, Node.js 22, Go 1.23] |

---

## 1. Document Overview

### 1.1 Purpose

[이 문서의 목적과 대상 독자를 기술합니다]
- 예: "이 문서는 `[컴포넌트명]`의 내부 설계와 인터페이스를 설명하며, 개발자와 리뷰어를 주요 독자로 합니다."

### 1.2 Scope

**포함 범위:**
- [이 문서에서 다루는 내용]

**제외 범위:**
- [이 문서에서 명시적으로 다루지 않는 내용]

### 1.3 Definitions & Abbreviations

| Term | Definition |
|------|------------|
| | |

---

## 2. System Overview

### 2.1 Component Description

[컴포넌트의 핵심 기능과 비즈니스/기술적 가치를 설명합니다]

**핵심 역할:**
- [역할 1]
- [역할 2]

**주요 사용자/소비자:**
- [이 컴포넌트를 사용하는 시스템/팀/사용자]

### 2.2 Architecture Diagram

[컴포넌트의 내부 구조와 외부 연동을 ASCII 다이어그램으로 표현합니다]

```
┌─────────────────────────────────────────────────────────┐
│                  [Component Name]                        │
├───────────────────────┬─────────────────────────────────┤
│     Core Module       │        Supporting Modules        │
│  ┌─────────────────┐  │  ┌──────────┐  ┌─────────────┐  │
│  │  Business Logic │  │  │ Config   │  │  Utilities  │  │
│  └─────────────────┘  │  └──────────┘  └─────────────┘  │
└───────────────────────┴─────────────────────────────────┘
         │                          │
         ▼                          ▼
  [External API]            [Database / Cache]
```

### 2.3 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | | | |
| Framework | | | |
| Database | | | |
| Cache | | | |
| Message Queue | | | |
| External APIs | | | |

### 2.4 Non-Functional Requirements Summary

| Category | Requirement |
|----------|------------|
| Availability | [예: 99.9%] |
| Latency | [예: P95 < 500ms] |
| Scalability | [예: 수평 확장 지원] |
| Security | [예: OAuth2, mTLS] |

---

## 3. Core Components

### 3.1 Module Structure

```
[component-root]/
├── [entry-point]          # 진입점
├── [core-module]/         # 핵심 로직
├── [interface-module]/    # 인터페이스 정의
├── [config-module]/       # 설정
├── [util-module]/         # 유틸리티
└── tests/                 # 테스트
```

### 3.2 Key Classes / Functions

각 핵심 클래스/함수에 대해 아래 형식으로 기술합니다:

#### 3.2.1 [ClassName / FunctionName]

| Property | Value |
|----------|-------|
| Name | |
| File Path | |
| Type | Class / Function / Interface |
| Purpose | |
| Public Interface | |

**Responsibilities:**
- [책임 1]
- [책임 2]

**Key Methods / Parameters:**
| Name | Type | Description |
|------|------|-------------|
| | | |

**Dependencies:**
- [의존하는 모듈/클래스/서비스]

### 3.3 Configuration

| Key | Type | Default | Required | Description |
|-----|------|---------|----------|-------------|
| | | | | |

---

## 4. Data Models & Schemas

### 4.1 Input / Request Models

#### 4.1.1 [Model Name]

```python
# 또는 TypeScript, Go struct, JSON Schema 등 해당 언어 형식
class ExampleRequest(BaseModel):
    field: str
```

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| | | | | | |

### 4.2 Output / Response Models

#### 4.2.1 [Model Name]

```python
class ExampleResponse(BaseModel):
    field: str
```

| Field | Type | Description |
|-------|------|-------------|
| | | |

### 4.3 Internal / Domain Models

[컴포넌트 내부에서만 사용되는 데이터 모델을 기술합니다]

### 4.4 Database Schema (해당 시)

```sql
-- 테이블 정의
CREATE TABLE example (
  id SERIAL PRIMARY KEY,
  ...
);
```

---

## 5. Interface Specifications

### 5.1 HTTP API (해당 시)

#### 5.1.1 [Endpoint Name]

| Property | Value |
|----------|-------|
| Method | GET / POST / PUT / DELETE / PATCH |
| Path | /api/v1/... |
| Auth | Bearer Token / API Key / None |
| Description | |

**Request:**
```json
{
  "field": "value"
}
```

**Response (200 OK):**
```json
{
  "field": "value"
}
```

**Error Responses:**
| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | INVALID_REQUEST | 요청 형식 오류 |
| 401 | UNAUTHORIZED | 인증 실패 |
| 403 | FORBIDDEN | 권한 없음 |
| 404 | NOT_FOUND | 리소스 없음 |
| 500 | INTERNAL_ERROR | 서버 내부 오류 |

### 5.2 Event / Message Interface (해당 시)

| Topic / Queue | Direction | Schema | Description |
|---------------|-----------|--------|-------------|
| | Publish / Subscribe | | |

### 5.3 CLI Interface (해당 시)

```
[component] [command] [options]

Commands:
  start     Start the service
  stop      Stop the service

Options:
  --config  Path to config file
  --port    Port to listen on
```

### 5.4 Library / SDK Interface (해당 시)

```python
# Public API 예시
from component import Client

client = Client(config)
result = client.do_something(input)
```

---

## 6. Processing Flow

### 6.1 Main Flow

[주요 처리 흐름을 단계별로 설명합니다]

```
1. 요청 수신
   └─ 2. 입력 검증
         └─ 3. 비즈니스 로직 실행
               ├─ 3a. 외부 서비스 호출 (선택)
               └─ 3b. 데이터 저장/조회
                     └─ 4. 응답 반환
```

### 6.2 Key Scenarios

#### Scenario 1: [정상 케이스]

```
Client → [Component] → [External Service]
       ←──────────────────────────────────
```

1. [단계 설명]
2. [단계 설명]

#### Scenario 2: [오류 케이스]

[오류 발생 시 처리 흐름]

---

## 7. Error Handling Requirements

### 7.1 Error Categories

| Category | Description | HTTP Status | Handling Strategy |
|----------|-------------|-------------|-------------------|
| Validation Error | 입력 데이터 유효성 검증 실패 | 400 | 즉시 반환, 재시도 불필요 |
| Authentication Error | 인증 실패 | 401 | 즉시 반환 |
| Authorization Error | 권한 없음 | 403 | 즉시 반환 |
| Not Found | 리소스 없음 | 404 | 즉시 반환 |
| External API Error | 외부 API 호출 실패 | 502 | 재시도 또는 Fallback |
| Timeout | 처리 시간 초과 | 504 | 재시도 또는 Fallback |
| Internal Error | 내부 로직 오류 | 500 | 로깅 후 반환 |

### 7.2 Standard Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "additional context"
    },
    "request_id": "uuid"
  }
}
```

### 7.3 Retry Strategy

| Error Type | Retryable | Max Retries | Backoff | Timeout |
|------------|-----------|-------------|---------|---------|
| Network Timeout | Yes | 3 | Exponential (1s, 2s, 4s) | 30s |
| External API 5xx | Yes | 3 | Exponential + Jitter | 30s |
| Validation Error | No | - | - | - |
| Auth Error | No | - | - | - |

### 7.4 Fallback Mechanisms

[오류 발생 시 대체 동작을 기술합니다]
- [예: 캐시에서 이전 결과 반환]
- [예: 기본값 반환]
- [예: 서킷 브레이커 Open 시 즉시 오류 반환]

---

## 8. Performance Requirements

### 8.1 Response Time Targets

| Operation | P50 | P95 | P99 | Max |
|-----------|-----|-----|-----|-----|
| | ms | ms | ms | ms |

### 8.2 Throughput

| Metric | Target | Peak |
|--------|--------|------|
| Requests per second | | |
| Concurrent connections | | |
| Batch size | | |

### 8.3 Resource Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Memory | | |
| CPU | | |
| Disk I/O | | |
| Network I/O | | |
| DB connections | | |

### 8.4 Caching Strategy

| Data | Cache Layer | TTL | Invalidation Strategy |
|------|-------------|-----|-----------------------|
| | | | |

---

## 9. Security Considerations

### 9.1 Authentication & Authorization

| Mechanism | Description |
|-----------|-------------|
| AuthN | [인증 방식: JWT, OAuth2, API Key 등] |
| AuthZ | [권한 관리 방식: RBAC, ABAC 등] |

### 9.2 Data Protection

| Data Type | Classification | Protection Method |
|-----------|---------------|-------------------|
| PII | Sensitive | 암호화, 마스킹, 최소 수집 |
| Credentials | Secret | 환경변수, Secret Manager |
| API Keys | Secret | Vault, Secret Manager |
| User Data | Confidential | 암호화 저장 |

### 9.3 Input Validation

- [ ] 길이 제한 적용
- [ ] 타입 검증
- [ ] 특수문자 sanitization
- [ ] SQL Injection 방어
- [ ] XSS 방어
- [ ] Path Traversal 방어

### 9.4 Network Security

- [ ] TLS 1.2+ 강제
- [ ] CORS 정책 설정
- [ ] Rate Limiting 적용
- [ ] 방화벽 / Security Group 설정

### 9.5 Secrets Management

| Secret | Storage | Rotation Policy |
|--------|---------|-----------------|
| | | |

---

## 10. Testing Strategy

### 10.1 Test Pyramid

```
       /\
      /  \      E2E / System Tests
     /----\
    /      \    Integration Tests
   /--------\
  /          \  Unit Tests
 --------------
```

### 10.2 Unit Tests

| Test Target | File Path | Description | Coverage Target |
|-------------|-----------|-------------|-----------------|
| | | | |

### 10.3 Integration Tests

| Test Scenario | File Path | External Dependencies | Description |
|---------------|-----------|----------------------|-------------|
| | | | |

### 10.4 End-to-End Tests

| Test Scenario | Description | Preconditions |
|---------------|-------------|---------------|
| | | |

### 10.5 Performance / Load Tests

| Test Type | Tool | Target Metric | Pass Criteria |
|-----------|------|---------------|---------------|
| Load Test | | RPS | |
| Stress Test | | Max RPS | No errors |
| Soak Test | | Duration | Memory stable |

### 10.6 Test Data Strategy

| Category | Description |
|----------|-------------|
| Fixtures | [테스트 픽스처 관리 방법] |
| Mocking | [외부 의존성 Mock 전략] |
| Test DB | [테스트 데이터베이스 전략] |

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Module / Service | Version | Purpose | Interface |
|-----------------|---------|---------|-----------|
| | | | |

### 11.2 External Dependencies (Libraries)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| | | | |

### 11.3 Infrastructure Dependencies

| Service | Provider | Purpose | SLA |
|---------|----------|---------|-----|
| | | | |

### 11.4 Dependency Risk Assessment

| Dependency | Risk Level | Mitigation |
|------------|------------|------------|
| | Low/Medium/High | |

---

## 12. Deployment & Operations

### 12.1 Environment Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| | | Yes/No | | |

### 12.2 Deployment Architecture

[배포 구조를 기술합니다]
- 컨테이너화 여부 (Docker, OCI)
- 오케스트레이션 (Kubernetes, ECS 등)
- 배포 전략 (Blue/Green, Canary, Rolling)

### 12.3 Health Checks

| Endpoint | Type | Success Criteria |
|----------|------|-----------------|
| `/health` | Liveness | HTTP 200 |
| `/ready` | Readiness | HTTP 200 + DB connected |

### 12.4 Monitoring & Observability

**Metrics:**
| Metric Name | Type | Description | Alert Threshold |
|-------------|------|-------------|-----------------|
| | Counter/Gauge/Histogram | | |

**Logging:**
| Level | Usage |
|-------|-------|
| DEBUG | 개발 시 상세 디버깅 |
| INFO | 주요 이벤트 (요청 시작/종료) |
| WARNING | 잠재적 문제 (재시도, 느린 쿼리) |
| ERROR | 오류 발생 (복구 가능) |
| CRITICAL | 서비스 중단 수준 오류 |

**Distributed Tracing:**
- Trace ID 전파 방식: [W3C TraceContext, B3 등]
- Sampling Rate: [예: 10%]

### 12.5 Alerting Rules

| Alert Name | Condition | Severity | Action |
|------------|-----------|----------|--------|
| High Error Rate | error_rate > 1% for 5min | P1 | PagerDuty |
| High Latency | P95 > 1s for 5min | P2 | Slack |
| Low Availability | availability < 99.9% | P1 | PagerDuty |

---

## 13. References

### 13.1 Related Documents

- [관련 PRD / 기획서]
- [관련 Architecture Decision Records (ADR)]
- [관련 API 문서]
- [관련 Runbook]

### 13.2 External References

- [사용 기술 공식 문서]
- [관련 RFC / 표준]

### 13.3 TDD 작성 참고 항목

아래 항목들을 TDD 작성 시 모두 검토하세요:

1. **Document Overview**: 목적, 범위, 용어 정의
2. **System Overview**: 컴포넌트 설명, 아키텍처 다이어그램, 기술 스택, NFR 요약
3. **Core Components**: 모듈 구조, 핵심 클래스/함수, 설정
4. **Data Models & Schemas**: 입출력 모델, 내부 모델, DB 스키마
5. **Interface Specifications**: HTTP API, 이벤트, CLI, 라이브러리 인터페이스
6. **Processing Flow**: 주요 처리 흐름, 핵심 시나리오
7. **Error Handling**: 에러 분류, 표준 에러 포맷, 재시도 전략, Fallback
8. **Performance**: 응답 시간 목표, 처리량, 리소스 한계, 캐싱
9. **Security**: 인증/인가, 데이터 보호, 입력 검증, 네트워크 보안
10. **Testing Strategy**: 단위/통합/E2E/성능 테스트, 테스트 데이터
11. **Dependencies**: 내부/외부 의존성, 인프라, 리스크
12. **Deployment & Operations**: 환경 변수, 헬스체크, 모니터링, 알림

---

## Appendix

### A. Change History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | YYYY-MM-DD | Claude Code | Initial draft |

### B. Open Questions

| # | Question | Owner | Due Date | Status |
|---|----------|-------|----------|--------|
| 1 | | | | Open |

### C. Approval History

| Role | Name | Date |
|------|------|------|
| Author | | |
| Tech Lead | | |
| Reviewer | | |
| Approver | | |
```

---

## 출력 위치

생성된 TDD 문서는 다음 경로에 저장합니다:
- 기본: `docs/tdd/[component-name]-tdd.md`
- `--output` 옵션 지정 시: 지정된 경로

## 추가 리소스

TDD 작성 상세 가이드는 [reference.md](reference.md)를 참조하세요.
