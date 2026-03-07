Review the entire conversation history and create a handoff document for the next session.

Generate a structured summary with the following sections:

## Goal
State the primary objective we're working on.

## Current Status
- What's working
- What's broken or incomplete

## Completed
- [x] List finished tasks

## Not Yet Done
- [ ] List remaining tasks

## Failed Approaches (Don't Repeat These)
Document what was tried but didn't work, why it failed, and include actual error messages if any.

## Key Decisions
| Decision | Rationale |
|----------|-----------|
| What we chose | Why we chose it |

## Relevant Files
List key files with brief descriptions. Include specific line numbers if relevant.

## Next Steps
Provide clear, actionable instructions for resuming work. Include expected outcomes for verification.

---

Before saving, derive a session title from the Goal section: lowercase English letters and hyphens only, 20 characters or less (e.g. `fix-search-index`).

Then run the following shell command to get the current timestamp:
```
date +%y%m%d_%H%M
```

Save this to `.claude/handoffs/<session-title>/<yymmdd_hhmm>/SUMMARY.md` using the session title you derived and the timestamp output.
