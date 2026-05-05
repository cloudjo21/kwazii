# Python 유닛 테스트 가이드

> Clean Code 원칙과 프로젝트 실제 코드 기반의 Python pytest 유닛 테스트 작성 가이드입니다.

## 목차

1. [핵심 원칙](#1-핵심-원칙)
2. [테스트 구조 - 클래스 기반 그룹화](#2-테스트-구조---클래스-기반-그룹화)
3. [Fixture - 테스트 데이터 관리](#3-fixture---테스트-데이터-관리)
4. [Mocking - `@mock.patch` vs `monkeypatch`](#4-mocking---mockpatch-vs-monkeypatch)
5. [테스트 명명 규칙](#5-테스트-명명-규칙)
6. [Parametrize - 다중 입력 테스트](#6-parametrize---다중-입력-테스트)
7. [비동기 테스트](#7-비동기-테스트)
8. [예외 테스트](#8-예외-테스트)
9. [Mock 활용 패턴](#9-mock-활용-패턴)
10. [동시성 테스트](#10-동시성-테스트)
11. [파일 전체 예시 비교](#11-파일-전체-예시-비교)

---

## 1. 핵심 원칙

Clean Code의 테스트 원칙인 **F.I.R.S.T**를 기본으로 합니다.

| 원칙 | 설명 |
|------|------|
| **Fast** | 테스트는 빠르게 실행되어야 한다 |
| **Independent** | 테스트는 서로 독립적이어야 한다 |
| **Repeatable** | 어떤 환경에서도 동일한 결과가 나와야 한다 |
| **Self-validating** | 테스트는 스스로 성공/실패를 판단해야 한다 |
| **Timely** | 테스트는 프로덕션 코드와 함께 작성되어야 한다 |

### 테스트당 하나의 개념

```python
# Bad: 하나의 테스트에서 여러 개념을 검증
def test_client() -> None:
    client = create_client()
    assert client.domain_name == "accommodation"      # 속성 검증
    result = asyncio.run(client.search(request))       # 기능 검증
    assert result is not None

# Good: 각 개념을 별도 테스트로 분리
def test_client_attributes(self, ...) -> None:
    """Tests that client attributes are set correctly."""
    client = AccommodationSearchGrpcClient(grpc_client_instance)
    assert client.domain_name == "accommodation"

async def test_search_success(self, ...) -> None:
    """Tests successful search operation."""
    result = await client.search(sample_request_message)
    assert result is mock_response
```

---

## 2. 테스트 구조 - 클래스 기반 그룹화

관련된 테스트는 **반드시 클래스로 묶습니다**. 클래스는 테스트 대상을 명확히 하고, fixture 공유 범위를 제한하며, 테스트 리포트의 가독성을 높입니다.

### Bad: 함수 레벨 나열

```python
# test_simple_adk_agent.py - 개선 전 패턴
def test_invoke_missing_model_returns_validation_error() -> None:
    strategy = _build_strategy()
    result = list(strategy._invoke({"query": "hello"}))
    assert result == [{"type": "text", "text": MISSING_MODEL_MESSAGE}]

def test_invoke_missing_query_returns_validation_error() -> None:
    strategy = _build_strategy()
    result = list(strategy._invoke({"model": _default_model(), "query": "   "}))
    assert result == [{"type": "text", "text": MISSING_QUERY_MESSAGE}]
```

### Good: 클래스로 그룹화

```python
# test_grpc_clients.py - 권장 패턴
class TestAccommodationCidSearchClient:
    """Tests for AccommodationCidSearchClient class."""

    def test_client_attributes(self, mock_json_grpc_client: mock.Mock) -> None:
        """Tests that client attributes are set correctly."""
        ...

    async def test_search_success(self, ...) -> None:
        """Tests successful search operation."""
        ...

    async def test_search_with_parse_error(self, ...) -> None:
        """Tests search operation when ParseError occurs."""
        ...
```

**클래스명 규칙**: `Test{테스트_대상_클래스명}`

---

## 3. Fixture - 테스트 데이터 관리

`@pytest.fixture`를 사용하여 테스트 데이터를 관리합니다. 클래스 내부에 정의된 fixture는 해당 클래스의 테스트 메서드에서만 사용됩니다.

### Bad: 헬퍼 함수로 테스트 데이터 관리

```python
# test_simple_adk_agent.py - 개선 전 패턴
def _default_model() -> dict[str, Any]:
    return {"provider": "google", "name": "gemini-2.0-flash"}

def _build_strategy() -> MockSimpleAdkAgentStrategy:
    return MockSimpleAdkAgentStrategy.__new__(MockSimpleAdkAgentStrategy)

def test_invoke_missing_query_returns_validation_error() -> None:
    strategy = _build_strategy()  # 매번 직접 호출
    result = list(strategy._invoke({"model": _default_model(), "query": "   "}))
```

### Good: `@pytest.fixture`로 테스트 데이터 선언

```python
# test_grpc_clients.py - 권장 패턴
class TestAccommodationCidSearchClient:

    @pytest.fixture
    def base_url(self) -> str:
        return "test-search-host.example.com:443"

    @pytest.fixture
    def sample_request_message(self) -> service_pb.LocalPropertyCidSearchRequest:
        return service_pb.LocalPropertyCidSearchRequest(
            filters=service_pb.LocalPropertySearchCriteria(
                must_have=service_pb.LocalPropertyFilterGroup(
                    filters=[
                        service_pb.LocalPropertyFilter(
                            field=service_pb.LocalPropertyFieldFilter(
                                field="LOCAL_PROPERTY_FIELD_NAME",
                                string_values={"values": ["강남 호텔"]},
                            )
                        )
                    ]
                ),
            ),
            limit=10,
        )

    @pytest.fixture
    def sample_response(self) -> dict:
        return {
            "results": [
                {
                    "id": "12345",
                    "name": "테스트 호텔",
                    "address": "서울시 강남구",
                }
            ]
        }

    async def test_search_success(
        self,
        mock_json_grpc_client: mock.Mock,
        base_url: str,                                     # fixture 주입
        sample_request_message: service_pb.LocalPropertyCidSearchRequest,  # fixture 주입
    ) -> None:
        """Tests successful search operation."""
        ...
```

**fixture 작성 규칙:**
- 반환 타입을 명시한다 (`-> str`, `-> dict`, `-> SomeClass`)
- fixture 이름은 역할을 명확히 설명하는 명사형으로 짓는다 (`base_url`, `sample_request_message`)
- 공유 범위가 필요할 때만 `scope` 파라미터를 사용한다 (기본값 `"function"` 권장)

---

## 4. Mocking - `@mock.patch` vs `monkeypatch`

`unittest.mock`의 **`@mock.patch` 데코레이터**를 기본으로 사용합니다. `monkeypatch`는 pytest 전용이며 패치 범위와 복원 방식이 덜 명시적입니다.

### Bad: `monkeypatch` 사용

```python
# test_simple_adk_agent.py - 개선 전 패턴
def test_invoke_success_returns_normalized_text(monkeypatch: Any) -> None:
    strategy = _build_strategy()
    monkeypatch.setattr(
        strategy,
        "_create_adk_runner",
        lambda app_name, model_config, system_instruction: object(),
    )
    monkeypatch.setattr(strategy, "_run_adk_query", lambda runner, query: {"answer": "hello from adk"})
```

### Good: `@mock.patch` 데코레이터 사용

```python
# test_grpc_clients.py - 권장 패턴
@mock.patch("core.api_client.grpc_client.GrpcStubClient")
@pytest.mark.asyncio
async def test_search_success(
    self,
    mock_json_grpc_client: mock.Mock,   # @mock.patch 인자는 함수 시그니처에 역순으로 추가
    base_url: str,
    sample_request_message: service_pb.LocalPropertyCidSearchRequest,
) -> None:
    """Tests successful search operation."""
    grpc_client_instance = mock.Mock()
    mock_response = mock.Mock(spec=service_pb.LocalPropertyCidSearchResponse)
    grpc_client_instance.stub.Search = mock.AsyncMock(return_value=mock_response)
    mock_json_grpc_client.create = mock.AsyncMock(return_value=grpc_client_instance)

    client = await AccommodationSearchGrpcClient.create(base_url)
    result = await client.search(sample_request_message)

    assert result is mock_response
```

**`@mock.patch` 적용 규칙:**

- `@mock.patch.object` 대신 **`@mock.patch("패키지.경로.ClassName.method")`** 형식의 문자열을 사용한다
  ```python
  # Good: 패키지 경로 문자열로 대상 지정
  @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy._run_adk_query")

  # Bad: mock.patch.object 사용
  @mock.patch.object(SimpleAdkAgentStrategy, "_run_adk_query")
  ```
- `side_effect`, `return_value`는 **데코레이터 인자로 넘기지 않고** 테스트 본문에서 설정한다
  ```python
  # Good: 테스트 본문에서 동작 설정
  @mock.patch("module.ClassName.method")
  def test_something(self, mock_method: mock.Mock, ...) -> None:
      mock_method.return_value = "expected"       # 본문에서 설정
      mock_method.side_effect = RuntimeError()    # 본문에서 설정

  # Bad: 데코레이터에 return_value / side_effect 인라인
  @mock.patch("module.ClassName.method", return_value="expected")
  def test_something(self, mock_method: mock.Mock, ...) -> None: ...
  ```
- 패치 경로는 **사용되는 위치의 모듈 경로**를 사용한다 (정의된 위치가 아닌 `import`된 위치)
  ```python
  # grpc_clients.py에서 import된 위치 기준
  @mock.patch("infrastructures.nol.cid_search.grpc_clients.protobuf_utils.validate_parse_dict")
  # 정의된 위치(X): @mock.patch("core.utils.protobuf_utils.validate_parse_dict")
  ```
- 여러 `@mock.patch`를 쌓을 때, 함수 인자는 **아래에서 위** 순서로 매핑된다
  ```python
  @mock.patch("module.ClassB")   # → mock_b (두 번째 인자)
  @mock.patch("module.ClassA")   # → mock_a (첫 번째 인자)
  async def test_something(self, mock_a: mock.Mock, mock_b: mock.Mock, ...) -> None:
  ```

---

## 5. 테스트 명명 규칙

테스트 이름은 **`test_{대상}_{시나리오}`** 패턴을 사용합니다. 이름만으로 어떤 상황을 테스트하는지 파악할 수 있어야 합니다.

```python
# Good: 무엇을, 어떤 조건에서 테스트하는지 명확
test_client_create_and_init          # 생성 및 초기화 검증
test_search_success                  # 정상 동작 검증
test_search_with_parse_error         # ParseError 발생 시 동작 검증
test_search_with_non_parse_error     # 일반 예외 발생 시 동작 검증
test_search_with_various_requests    # 다양한 요청 형식 검증
test_concurrent_search               # 동시 요청 검증

# Bad: 구현 세부사항이 드러나거나 의미가 불명확
test_1
test_search_ok
test_it_works
test_search_grpc_stub_mock_call      # 구현 세부사항 노출
```

**모든 테스트 메서드에 docstring을 작성합니다:**

```python
async def test_search_with_parse_error(self, ...) -> None:
    """Tests search operation when ParseError occurs."""   # 무엇을 검증하는지 한 줄로 설명
    ...
```

---

## 6. Parametrize - 다중 입력 테스트

동일한 로직을 여러 입력값으로 검증할 때는 `@pytest.mark.parametrize`를 사용합니다. 테스트 함수를 반복 작성하지 않습니다.

### Bad: 유사 테스트 반복 작성

```python
def test_invoke_missing_model() -> None:
    ...

def test_invoke_missing_query() -> None:
    ...

def test_invoke_blank_query() -> None:
    ...
```

### Good: `@pytest.mark.parametrize` 활용

```python
# test_grpc_clients.py 패턴
@mock.patch("core.api_client.grpc_client.GrpcStubClient")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_message",
    [
        service_pb.LocalPropertyCidSearchRequest(limit=1),
        service_pb.LocalPropertyCidSearchRequest(limit=5),
        service_pb.LocalPropertyCidSearchRequest(
            filters=service_pb.LocalPropertySearchCriteria(
                must_have=service_pb.LocalPropertyFilterGroup(
                    filters=[
                        service_pb.LocalPropertyFilter(
                            field=service_pb.LocalPropertyFieldFilter(
                                field="LOCAL_PROPERTY_FIELD_NAME",
                                string_values={"values": ["호텔"]},
                            )
                        ),
                    ]
                ),
            ),
            limit=20,
        ),
    ],
)
async def test_search_with_various_requests(
    self,
    mock_json_grpc_client: mock.Mock,
    base_url: str,
    request_message: service_pb.LocalPropertyCidSearchRequest,
) -> None:
    """Tests search operation with various request message formats."""
    ...
    result = await client.search(request_message)
    assert result is mock_response
```

**파라미터 ID 명시 (선택):** 테스트 리포트에서 케이스를 구분하려면 `ids`를 사용합니다.

```python
@pytest.mark.parametrize(
    "status_code, expected",
    [
        (200, True),
        (404, False),
        (500, False),
    ],
    ids=["success", "not_found", "server_error"],
)
def test_is_success(self, status_code: int, expected: bool) -> None:
    ...
```

---

## 7. 비동기 테스트

`async` 함수를 테스트할 때는 `@pytest.mark.asyncio`를 사용합니다.

```python
# test_grpc_clients.py 패턴
@mock.patch("core.api_client.grpc_client.GrpcStubClient")
@pytest.mark.asyncio                               # 비동기 테스트 마커
async def test_client_create_and_init(
    self,
    mock_json_grpc_client: mock.Mock,
    base_url: str,
) -> None:
    """Tests async create and __init__."""
    mock_instance = mock_json_grpc_client.create.return_value
    mock_json_grpc_client.create = mock.AsyncMock(return_value=mock_instance)

    client = await AccommodationSearchGrpcClient.create(base_url)  # await 사용
    assert isinstance(client, AccommodationSearchGrpcClient)
```

**`mock.AsyncMock` 사용:** async 메서드를 mock할 때는 반드시 `mock.AsyncMock`을 사용합니다.

```python
# async 메서드 mock
grpc_client_instance.stub.Search = mock.AsyncMock(return_value=mock_response)
mock_json_grpc_client.create = mock.AsyncMock(return_value=grpc_client_instance)

# async mock 호출 검증
grpc_client_instance.stub.Search.assert_awaited_once_with(
    sample_request_message, timeout=mock.ANY, metadata=mock.ANY
)
```

---

## 8. 예외 테스트

예외 발생을 검증할 때는 `pytest.raises()`를 사용합니다.

```python
# test_grpc_clients.py 패턴
@mock.patch("core.api_client.grpc_client.GrpcStubClient")
@pytest.mark.asyncio
async def test_search_with_non_parse_error(
    self,
    mock_json_grpc_client: mock.Mock,
    base_url: str,
    sample_request_message: service_pb.LocalPropertyCidSearchRequest,
) -> None:
    """Tests search operation when non-ParseError exception occurs."""
    grpc_client_instance = mock.Mock()
    grpc_error = Exception("gRPC connection failed")
    grpc_client_instance.stub.Search = mock.AsyncMock(side_effect=grpc_error)
    mock_json_grpc_client.create = mock.AsyncMock(return_value=grpc_client_instance)

    client = await AccommodationSearchGrpcClient.create(base_url)
    with pytest.raises(Exception, match="gRPC connection failed"):   # 예외 타입과 메시지 검증
        await client.search(sample_request_message)
```

**`side_effect`로 예외 발생 설정:**

```python
# 예외 인스턴스 전달
mock.stub.method = mock.AsyncMock(side_effect=ValueError("invalid input"))

# 예외 클래스 전달 (인자 없이)
mock.stub.method = mock.Mock(side_effect=ConnectionError)

# 함수로 조건부 예외 발생
def raise_on_first_call(call_count=[0]):
    call_count[0] += 1
    if call_count[0] == 1:
        raise TimeoutError("timeout")
    return "success"

mock.stub.method = mock.Mock(side_effect=raise_on_first_call)
```

---

## 9. Mock 활용 패턴

### `mock.ANY` - 유연한 assertion

검증이 불필요한 인자에는 `mock.ANY`를 사용합니다.

```python
# test_grpc_clients.py 패턴
grpc_client_instance.stub.Search.assert_awaited_once_with(
    sample_request_message,    # 정확히 검증
    timeout=mock.ANY,          # 값은 상관없으나 전달 여부 확인
    metadata=mock.ANY,
    wait_for_ready=mock.ANY,
)
```

### `mock.Mock(spec=...)` - 안전한 Mock

`spec`을 지정하면 실제 객체에 없는 속성/메서드 접근 시 `AttributeError`가 발생하여 잘못된 Mock 사용을 방지합니다.

```python
# Bad: spec 없는 Mock은 어떤 속성도 허용
mock_response = mock.Mock()
mock_response.nonexistent_field  # AttributeError 없음 → 잘못된 테스트 가능

# Good: spec 지정으로 실제 인터페이스 강제
mock_response = mock.Mock(spec=service_pb.LocalPropertyCidSearchResponse)
mock_response.nonexistent_field  # AttributeError 발생 → 잘못된 접근 즉시 탐지
```

### `assert_awaited_once_with` vs `assert_called_once_with`

| 메서드 | 사용 시점 |
|--------|----------|
| `assert_called_once_with(...)` | 동기 mock 호출 검증 |
| `assert_awaited_once_with(...)` | `AsyncMock` 호출 검증 |
| `assert_called_once()` | 인자 검증 없이 한 번 호출됨만 확인 |
| `assert_not_called()` | 호출되지 않았음을 확인 |
| `call_count` | 호출 횟수 확인 |

```python
# 동기
mock_func.assert_called_once_with(arg1, key=value)

# 비동기
async_mock.assert_awaited_once_with(arg1, key=value)

# 호출 횟수
assert grpc_client_instance.stub.Search.call_count == 3
```

---

## 10. 동시성 테스트

비동기 클라이언트의 동시 요청 처리를 검증합니다.

```python
# test_grpc_clients.py 패턴
@mock.patch("core.api_client.grpc_client.GrpcStubClient")
@pytest.mark.asyncio
async def test_concurrent_search(
    self,
    mock_json_grpc_client: mock.Mock,
    base_url: str,
    sample_request_message: service_pb.LocalPropertyCidSearchRequest,
) -> None:
    """Tests concurrent search requests."""
    grpc_client_instance = mock.Mock()
    mock_response = mock.Mock(spec=service_pb.LocalPropertyCidSearchResponse)
    grpc_client_instance.stub.Search = mock.AsyncMock(return_value=mock_response)
    mock_json_grpc_client.create = mock.AsyncMock(return_value=grpc_client_instance)

    client = await AccommodationSearchGrpcClient.create(base_url)
    tasks = [
        client.search(sample_request_message),
        client.search(sample_request_message),
        client.search(sample_request_message),
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for result in results:
        assert result is mock_response
    assert grpc_client_instance.stub.Search.call_count == 3
```

---

## 11. 파일 전체 예시 비교

### Before: 개선 전 패턴 (`test_simple_adk_agent.py` 기반)

```python
"""Unit tests for SimpleAdkAgentStrategy."""

from typing import Any
from nol_agents.strategies.simple_adk_agent import (
    MISSING_MODEL_MESSAGE,
    SimpleAdkAgentStrategy,
)


# 문제: 클래스 없이 모듈 수준 헬퍼 함수 사용
def _default_model() -> dict[str, Any]:
    return {"provider": "google", "name": "gemini-2.0-flash"}


def _build_strategy() -> SimpleAdkAgentStrategy:
    # 문제: __new__로 비관용적 인스턴스 생성
    return SimpleAdkAgentStrategy.__new__(SimpleAdkAgentStrategy)


# 문제: 클래스 그룹화 없음, docstring 없음
def test_invoke_missing_model_returns_validation_error() -> None:
    strategy = _build_strategy()
    result = list(strategy._invoke({"query": "hello"}))
    assert result == [{"type": "text", "text": MISSING_MODEL_MESSAGE}]


# 문제: monkeypatch 사용
def test_invoke_success_returns_normalized_text(monkeypatch: Any) -> None:
    strategy = _build_strategy()
    monkeypatch.setattr(
        strategy,
        "_create_adk_runner",
        lambda app_name, model_config, system_instruction: object(),
    )
    monkeypatch.setattr(
        strategy, "_run_adk_query", lambda runner, query: {"answer": "hello"}
    )
    result = list(strategy._invoke({"model": _default_model(), "query": "hello"}))
    assert result == [{"type": "text", "text": "hello"}]
```

### After: 권장 패턴 (`test_grpc_clients.py` 기준)

```python
"""Unit tests for SimpleAdkAgentStrategy."""

from typing import Any
from unittest import mock

import pytest

from nol_agents.strategies.simple_adk_agent import (
    MISSING_MODEL_MESSAGE,
    MISSING_QUERY_MESSAGE,
    EMPTY_RESPONSE_FALLBACK,
    EXECUTION_FAILURE_MESSAGE,
    SimpleAdkAgentStrategy,
)


class TestSimpleAdkAgentStrategy:
    """Tests for SimpleAdkAgentStrategy class."""

    # fixture로 공유 테스트 데이터 관리
    @pytest.fixture
    def model_config(self) -> dict[str, Any]:
        return {"provider": "google", "name": "gemini-2.0-flash"}

    @pytest.fixture
    def strategy(self) -> SimpleAdkAgentStrategy:
        return SimpleAdkAgentStrategy.__new__(SimpleAdkAgentStrategy)

    # 파라미터화로 유사 케이스 통합
    @pytest.mark.parametrize(
        "params, expected_message",
        [
            ({"query": "hello"}, MISSING_MODEL_MESSAGE),
            ({"model": {"provider": "google", "name": "gemini-2.0-flash"}, "query": "   "}, MISSING_QUERY_MESSAGE),
        ],
        ids=["missing_model", "blank_query"],
    )
    def test_invoke_validation_error(
        self,
        strategy: SimpleAdkAgentStrategy,
        params: dict[str, Any],
        expected_message: str,
    ) -> None:
        """Tests that validation errors return appropriate messages."""
        result = list(strategy._invoke(params))  # noqa: SLF001
        assert result[0]["text"] == expected_message

    # @mock.patch으로 명시적 mocking (패키지 경로 문자열 사용)
    @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy._run_adk_query")
    @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy._create_adk_runner")
    @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy.create_text_message")
    def test_invoke_success(
        self,
        mock_create_text: mock.Mock,   # 아래에서 위 순서: create_text_message
        mock_create_runner: mock.Mock, # _create_adk_runner
        mock_run_query: mock.Mock,     # _run_adk_query
        strategy: SimpleAdkAgentStrategy,
        model_config: dict[str, Any],
    ) -> None:
        """Tests successful invocation returns normalized text."""
        # side_effect / return_value는 테스트 본문에서 설정
        mock_create_text.side_effect = lambda text: {"type": "text", "text": text}
        mock_create_runner.return_value = mock.Mock()
        mock_run_query.return_value = "hello from adk"

        result = list(strategy._invoke({"model": model_config, "query": "hello"}))  # noqa: SLF001

        mock_run_query.assert_called_once()
        mock_create_text.assert_called_once_with(text="hello from adk")
        assert result == [{"type": "text", "text": "hello from adk"}]

    @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy._run_adk_query")
    @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy._create_adk_runner")
    @mock.patch("nol_agents.strategies.simple_adk_agent.SimpleAdkAgentStrategy.create_text_message")
    def test_invoke_execution_failure(
        self,
        mock_create_text: mock.Mock,
        mock_create_runner: mock.Mock,
        mock_run_query: mock.Mock,
        strategy: SimpleAdkAgentStrategy,
        model_config: dict[str, Any],
    ) -> None:
        """Tests execution failure returns friendly error message."""
        mock_create_text.side_effect = lambda text: {"type": "text", "text": text}
        mock_create_runner.return_value = mock.Mock()
        mock_run_query.side_effect = RuntimeError("execution failed")

        result = list(strategy._invoke({"model": model_config, "query": "hello"}))  # noqa: SLF001

        mock_create_text.assert_called_once_with(text=EXECUTION_FAILURE_MESSAGE)
        assert result == [{"type": "text", "text": EXECUTION_FAILURE_MESSAGE}]
```

---

## 체크리스트

테스트 작성 전후 다음 항목을 확인합니다.

### 구조
- [ ] 관련 테스트가 `class Test{ClassName}:` 으로 그룹화되어 있는가
- [ ] 모든 테스트 메서드에 docstring이 있는가
- [ ] 테스트 이름이 `test_{대상}_{시나리오}` 패턴을 따르는가

### 데이터 관리
- [ ] 공유 테스트 데이터가 `@pytest.fixture`로 관리되는가
- [ ] fixture에 반환 타입 힌트가 있는가

### Mocking
- [ ] `monkeypatch` 대신 `@mock.patch`를 사용하고 있는가
- [ ] async 메서드 mock에 `mock.AsyncMock`을 사용하고 있는가
- [ ] mock 검증에 `mock.ANY`와 `spec=`을 적절히 활용하고 있는가
- [ ] 패치 경로가 정의된 위치가 아닌 **사용되는 위치** 기준인가

### 커버리지
- [ ] 정상 케이스(happy path)가 테스트되어 있는가
- [ ] 예외/오류 케이스가 테스트되어 있는가
- [ ] 경계값(빈 문자열, None, 빈 리스트 등)이 테스트되어 있는가
- [ ] 유사 케이스가 `@pytest.mark.parametrize`로 통합되어 있는가

### 비동기
- [ ] async 테스트에 `@pytest.mark.asyncio`가 있는가
- [ ] 동시 요청 시나리오가 `asyncio.gather()`로 검증되어 있는가
