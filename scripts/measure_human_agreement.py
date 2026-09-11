"""Measure agreement between saved LLM-judge and human ratings."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.human_agreement import agreement_metrics  # noqa: E402


def main() -> int:
    evaluation_dir = ROOT / "evaluation"
    judge_path = evaluation_dir / "judge_scores.csv"
    human_path = evaluation_dir / "human_ratings.csv"
    if not judge_path.exists() or not human_path.exists():
        print("BLOCKED: provide evaluation/judge_scores.csv and evaluation/human_ratings.csv first.")
        return 2
    try:
        metrics = agreement_metrics(pd.read_csv(judge_path), pd.read_csv(human_path))
    except ValueError as error:
        print(f"BLOCKED: {error}")
        return 2
    output = evaluation_dir / "human_agreement_results.csv"
    metrics.to_csv(output, index=False)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())