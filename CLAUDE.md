# Project Constitution

## Project

- **Main branch**: `main`
- **Language**: Python (uv 환경)

## Skills & Commands


### Python 코드 작성/수정 시

→ **`python-style-checker`** skill 자동 적용
- `.py` 파일 생성·수정·리뷰·리팩터링 시 Google Python Style Guide 준수
- 완료 후 수정한 파일들에 한해서 `uv run ruff format` → `uv run ruff check --fix` → `uv run mypy` 순서로 실행

### 테스트 작성/실행 시

→ **`python-unittest`** skill 적용
- "테스트 작성", "unittest", "pytest 코드", "테스트 추가/실행/커버리지" 요청 시
- Clean Code F.I.R.S.T. 원칙 + 프로젝트 가이드(`docs/unittest-guides/python/README.md`) 참조
- 테스트 파일은 변경된 파일이 속한 프로젝트 내 `tests/` 하위에 위치

### Session Handoff

- command 위치: @.claude/commands/handoff.md

다음 상황에서 진행 상황을 정리한다:
- 세션 종료 전
- 작업 흐름상 컨텍스트가 무거워졌다고 느껴질 때
- Claude가 컨텍스트 한계 경고를 보낼 때

Handoff 작업이 끝나면 claude session을 /clear 처리한다.
