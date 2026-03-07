---
name: python-unittest
description: Clean Code 원칙(F.I.R.S.T.)에 기반하여 Python pytest 유닛 테스트를 작성하고 실행하는 스킬. 사용자가 "테스트 작성", "unittest 만들어줘", "pytest 코드 작성", "테스트 추가해줘", "테스트 실행", "테스트 커버리지 확인" 등을 요청하거나 특정 모듈/클래스/함수에 대한 테스트가 필요할 때 사용한다.
---

# Python Unittest (Clean Code 기반)

## 핵심 가이드 참조

**반드시** 테스트 작성 전에 프로젝트 가이드를 읽는다:

- [docs/unittest-guides/python/README.md](../../../docs/unittest-guides/python/README.md) — 전체 패턴, Bad/Good 예시, 체크리스트 포함

## 워크플로우

### 1. 대상 코드 분석

- 테스트 대상 클래스/함수를 읽는다
- public 메서드, 예외 경로, 비동기 여부를 파악한다
- 기존 테스트 파일이 있으면 먼저 읽어 패턴을 확인한다

### 2. 테스트 파일 위치 결정

- 변경된 파일이 속한 프로젝트 내 `tests/` 아래에 테스트 파일을 위치시킨다

```
tests/
├── unit/          # 단위 테스트 (mock 사용)
├── integration/   # 통합 테스트 (실제 의존성)
└── conftest.py    # 공유 fixture
```

파일명: `test_{모듈명}.py` (대상 파일명에 `test_` 접두사)

### 3. 테스트 작성 규칙 (요약)

| 항목 | 규칙 |
|------|------|
| 구조 | `class Test{ClassName}:` 으로 그룹화 |
| 명명 | `test_{대상}_{시나리오}` |
| Docstring | 모든 테스트 메서드에 한 줄 필수 |
| Fixture | `@pytest.fixture` 로 테스트 데이터 관리 |
| Mock | `@mock.patch("패키지.경로.Class.method")` 사용 |
| 비동기 | `@pytest.mark.asyncio` + `mock.AsyncMock` |
| 다중 입력 | `@pytest.mark.parametrize` 로 통합 |
| 예외 | `pytest.raises()` 사용 |

> 상세 패턴과 Bad/Good 예시: [docs/unittest-guides/python/README.md](../../../docs/unittest-guides/python/README.md)

### 4. 테스트 실행

```bash
# 특정 파일 실행
pytest tests/unit/test_my_module.py -v

# 특정 클래스만 실행
pytest tests/unit/test_my_module.py::TestMyClass -v

# 특정 테스트만 실행
pytest tests/unit/test_my_module.py::TestMyClass::test_search_success -v

# 비동기 포함 전체 실행
pytest tests/ -v

# 커버리지 측정
pytest tests/ --cov=nol_agents --cov-report=term-missing

# 실패한 테스트만 재실행
pytest tests/ --lf -v
```

### 5. 작성 후 체크리스트

- [ ] `class Test{ClassName}:` 로 그룹화
- [ ] 모든 테스트 메서드에 docstring
- [ ] 공유 데이터는 `@pytest.fixture`
- [ ] `monkeypatch` 대신 `@mock.patch` 사용
- [ ] async 테스트에 `@pytest.mark.asyncio`
- [ ] mock 경로가 **정의된 위치**가 아닌 **사용되는 위치** 기준
- [ ] Happy path + 예외 케이스 + 경계값 커버
- [ ] 유사 케이스 `@pytest.mark.parametrize` 로 통합
