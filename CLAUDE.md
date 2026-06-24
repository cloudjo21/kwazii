# nol-agents Project Constitution

## Project

- **Main branch**: `main`
- **Language**: Python (uv 환경)


## Skills & Commands

### Python 코드 작성/수정 시

→ **`python-style-checker`** skill 자동 적용
- `.py` 파일 생성·수정·리뷰·리팩터링 시 Google Python Style Guide 준수
- 완료 후 `uv run ruff format` → `uv run ruff check --fix` → `uv run mypy` 순서로 실행

### 테스트 작성/실행 시

→ **`python-unittest`** skill 적용
- "테스트 작성", "unittest", "pytest 코드", "테스트 추가/실행/커버리지" 요청 시
- Clean Code F.I.R.S.T. 원칙 + 프로젝트 가이드(`docs/unittest-guides/python/README.md`) 참조
- 테스트 파일은 변경된 파일이 속한 프로젝트 내 `tests/` 하위에 위치

### Session Handoff

- **트리거**: `/context-free-space` 결과 `free space ≤ 20%`일 때 handoff 권장
  > 예: `context usage free: 19% (used: 81%)` → handoff 권장 구간
- **절차**: `/handoff` 실행 → 핸드오프 문서 생성 완료 후 → `/clear` 실행
- Claude는 위 절차를 순서대로 자동 수행할 것

~/.claude/guides/ 아래 문서들 읽고 참고해