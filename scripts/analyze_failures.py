"""Generate failure analysis from real evaluation failure cases."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.failure_analysis import render_failure_report, summarize_failure_cases, validate_failure_cases  # noqa: E402


def main() -> int:
    path = ROOT / "evaluation" / "failure_cases.csv"
    if not path.exists():
        print("BLOCKED: provide evaluation/failure_cases.csv from real evaluation errors first.")
        return 2
    try:
        cases = validate_failure_cases(pd.read_csv(path))
    except ValueError as error:
        print(f"BLOCKED: {error}")
        return 2
    summary = summarize_failure_cases(cases)
    summary.to_csv(ROOT / "evaluation" / "failure_summary.csv", index=False)
    (ROOT / "docs" / "failure_analysis.md").write_text(render_failure_report(cases), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())