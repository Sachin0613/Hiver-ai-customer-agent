"""Run Phase 6 majority baseline after development labels are reviewed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.majority_baseline import classification_metrics, fit_majority, require_labels  # noqa: E402


def main() -> int:
    path = ROOT / "data" / "processed" / "intent_annotation_queue.csv"
    frame = pd.read_csv(path)
    try:
        labeled = require_labels(frame)
    except ValueError as error:
        print(f"BLOCKED: {error}")
        return 2
    model = fit_majority(labeled)
    predictions = model.predict(labeled["clean_text"])
    metrics = classification_metrics(labeled["intent_label"], predictions)
    metrics["baseline"] = "majority_class"
    metrics["majority_intent"] = model.intent
    output = ROOT / "evaluation" / "results" / "majority_baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())