"""Run Phase 7 TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_config  # noqa: E402
from src.tfidf_baseline import evaluate_tfidf, fit_tfidf_baseline, split_labeled_data  # noqa: E402


def main() -> int:
    config = load_config()
    seed = int(config["inspection"]["random_seed"])
    path = ROOT / "data" / "processed" / "intent_annotation_queue.csv"
    frame = pd.read_csv(path)
    try:
        train, validation = split_labeled_data(frame, random_seed=seed)
    except ValueError as error:
        print(f"BLOCKED: {error}")
        print("Manually fill intent_label in data/processed/intent_annotation_queue.csv first.")
        return 2
    model = fit_tfidf_baseline(train, random_seed=seed)
    metrics, matrix, predictions = evaluate_tfidf(model, validation)
    metrics.update({
        "baseline": "tfidf_logistic_regression",
        "random_seed": seed,
        "training_examples": len(train),
        "classes": model.labels,
    })
    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tfidf_baseline.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix.to_csv(output_dir / "tfidf_confusion_matrix.csv")
    predictions.to_csv(output_dir / "tfidf_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())