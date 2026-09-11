"""Evaluate saved predictions against manually labeled golden data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_escalation, evaluate_intent, require_complete_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "data" / "golden_set.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    if not args.golden.exists():
        print(f"BLOCKED: golden file does not exist: {args.golden}")
        return 2
    golden = pd.read_csv(args.golden)
    predictions = pd.read_csv(args.predictions)
    require_complete_labels(golden, ["id", "gold_intent", "gold_decision"])
    required_predictions = {"id", "predicted_intent", "predicted_decision"}
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        print(f"BLOCKED: predictions are missing columns: {missing}")
        return 2
    merged = golden[["id", "gold_intent", "gold_decision"]].merge(
        predictions[["id", "predicted_intent", "predicted_decision"]], on="id", how="inner"
    )
    if len(merged) != len(golden):
        print("BLOCKED: predictions do not cover every golden example.")
        return 2
    intent_metrics, confusion = evaluate_intent(merged["gold_intent"], merged["predicted_intent"])
    escalation_metrics = evaluate_escalation(merged["gold_decision"], merged["predicted_decision"])
    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(output_dir / "intent_confusion_matrix.csv")
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps({"intent": intent_metrics, "escalation": escalation_metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"intent": intent_metrics, "escalation": escalation_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())