"""
Ralph spec checker — validates that .ralph/specs/feature.md has the required structure.

Customize REQUIRED_SECTIONS and MIN_AC_ITEMS for your project's spec format.
Run: python3 scripts/spec_check.py .ralph/specs/feature.md
Output: JSON to stdout, exit 1 if check fails.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    "## Goal",
    "## Scope",
    "## Non-Goals",
    "## Acceptance Criteria",
    "## Test Cases",
]
MIN_AC_ITEMS = 3


def main(path: str) -> int:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        result = {"pass": False, "reason": f"file not found: {path}"}
        print(json.dumps(result))
        return 1

    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    ac_items = re.findall(r"^- \[ \]", text, flags=re.MULTILINE)

    passed = not missing and len(ac_items) >= MIN_AC_ITEMS
    result = {
        "pass": passed,
        "missing_sections": missing,
        "ac_items": len(ac_items),
        "reason": (
            "ok"
            if passed
            else f"missing={missing}, ac_items={len(ac_items)} (need {MIN_AC_ITEMS})"
        ),
    }
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ".ralph/specs/feature.md"))
