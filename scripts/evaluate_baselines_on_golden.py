"""Evaluate majority and TF-IDF intent baselines on the locked golden set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_intent, require_complete_labels  # noqa: E402
from src.majority_baseline import fit_majority  # noqa: E402
from src.tfidf_baseline import fit_tfidf_baseline  # noqa: E402


def main() -> int:
    development = pd.read_csv(ROOT / "data" / "processed" / "intent_annotation_queue.csv")
    golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
    require_complete_labels(golden, ["id", "gold_intent", "gold_decision"])
    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    majority = fit_majority(development)
    majority_predicted = majority.predict(golden["customer_message"])
    majority_metrics, majority_matrix = evaluate_intent(golden["gold_intent"], majority_predicted)
    majority_metrics.update({"model": "majority_class", "majority_intent": majority.intent})
    majority_matrix.to_csv(output_dir / "majority_golden_confusion_matrix.csv")

    tfidf = fit_tfidf_baseline(development, random_seed=42)
    tfidf_predicted = tfidf.predict(golden["customer_message"])
    tfidf_metrics, tfidf_matrix = evaluate_intent(golden["gold_intent"], tfidf_predicted)
    tfidf_metrics.update({"model": "tfidf_logistic_regression", "training_examples": len(development)})
    tfidf_matrix.to_csv(output_dir / "tfidf_golden_confusion_matrix.csv")

    predictions = golden[["id", "conversation_id", "customer_message", "gold_intent", "gold_decision"]].copy()
    predictions["majority_predicted_intent"] = majority_predicted.to_numpy()
    predictions["tfidf_predicted_intent"] = tfidf_predicted.to_numpy()
    predictions.to_csv(output_dir / "baseline_golden_predictions.csv", index=False)
    (output_dir / "baseline_golden_metrics.json").write_text(
        json.dumps({"majority": majority_metrics, "tfidf": tfidf_metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"majority": majority_metrics, "tfidf": tfidf_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())