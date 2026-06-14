"""
Ralph code quality scorer — composite 10-point deduction-based score.

Reads sub-reports written by verify.sh (.ralph/reports/radon_cc.json, .ralph/reports/jscpd/,
.ralph/reports/architecture.json) and thresholds from .ralph/QUALITY_GATE.json.

Customize deduction amounts below for your project's quality standards.
Architecture violations are always hard-fail regardless of numeric score.

Run: python3 scripts/score.py
Output: .ralph/reports/score.json + JSON to stdout.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, p.stdout + p.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def load_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> None:
    gate = load_json(".ralph/QUALITY_GATE.json")
    score = 10.0
    deductions: list[dict] = []
    hard_fail = False
    hard_fail_reasons: list[str] = []

    # --- Style / type safety: up to -2.0 ---
    lint_cmd = gate.get("lintCommand", "")
    if lint_cmd:
        code, _ = run(lint_cmd.split())
        if code != 0:
            score -= 2.0
            deductions.append({"rule": "lint", "amount": 2.0, "reason": "lint/type check failed"})

    # --- Complexity (radon CC report, if available) ---
    cc_report = load_json(".ralph/reports/radon_cc.json")
    warn_cc = gate.get("complexity", {}).get("warn", 10)
    fail_cc = gate.get("complexity", {}).get("fail", 15)
    warn_count = 0
    for blocks in cc_report.values():
        for block in blocks:
            cc = int(block.get("complexity", 0))
            if cc >= fail_cc:
                hard_fail = True
                hard_fail_reasons.append(
                    f"CC {cc} in {block.get('name')} ({block.get('file', '?')})"
                )
            elif cc > warn_cc:
                warn_count += 1
    if warn_count:
        amount = min(2.0, warn_count * 0.3)
        score -= amount
        deductions.append({
            "rule": "complexity",
            "amount": round(amount, 2),
            "reason": f"{warn_count} function(s) exceed CC {warn_cc}",
        })

    # --- Duplication (jscpd report, if available) ---
    jscpd_path = Path(".ralph/reports/jscpd/jscpd-report.json")
    if jscpd_path.exists():
        jscpd = load_json(str(jscpd_path))
        dup_pct = float(
            jscpd.get("statistics", {}).get("total", {}).get("percentage", 0.0)
        )
        fail_dup = gate.get("duplication", {}).get("fail", 5.0)
        warn_dup = gate.get("duplication", {}).get("warn", 3.0)
        if dup_pct > fail_dup:
            score -= 1.5
            deductions.append({
                "rule": "duplication",
                "amount": 1.5,
                "reason": f"duplication {dup_pct:.2f}% > {fail_dup}%",
            })
        elif dup_pct > warn_dup:
            score -= 0.5
            deductions.append({
                "rule": "duplication",
                "amount": 0.5,
                "reason": f"duplication {dup_pct:.2f}% > {warn_dup}%",
            })

    # --- Architecture (lint-imports report, if available) ---
    arch_report = load_json(".ralph/reports/architecture.json")
    if arch_report and not arch_report.get("pass", True):
        hard_fail = True
        hard_fail_reasons.append("architecture import contracts violated")

    final_score = round(max(score, 0.0), 2)
    min_score = gate.get("score", {}).get("min", 8.5)
    passed = (final_score >= min_score) and not hard_fail

    result = {
        "pass": passed,
        "score": final_score,
        "min_required": min_score,
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "deductions": deductions,
    }
    Path(".ralph/reports/score.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
