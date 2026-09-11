"""Summarize Phase 15 reply-quality ablation scores."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ablations import summarize_ablation_scores  # noqa: E402


def main() -> int:
    path = ROOT / "evaluation" / "ablation_scores.csv"
    if not path.exists():
        print("BLOCKED: provide evaluation/ablation_scores.csv with real judge scores first.")
        return 2
    try:
        summary = summarize_ablation_scores(pd.read_csv(path))
    except ValueError as error:
        print(f"BLOCKED: {error}")
        return 2
    output = ROOT / "evaluation" / "ablation_results.csv"
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())