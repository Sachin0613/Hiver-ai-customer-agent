"""Majority-class intent baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MajorityModel:
    intent: str

    def predict(self, messages: pd.Series) -> pd.Series:
        return pd.Series(self.intent, index=messages.index, name="predicted_intent")


def require_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"clean_text", "intent_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    labeled = frame[frame["intent_label"].fillna("").astype(str).str.strip().ne("")].copy()
    if labeled.empty:
        raise ValueError(
            "No intent labels found. Fill intent_label in "
            "data/processed/intent_annotation_queue.csv before evaluating."
        )
    return labeled


def fit_majority(frame: pd.DataFrame) -> MajorityModel:
    labeled = require_labels(frame)
    counts = Counter(labeled["intent_label"].astype(str).str.strip())
    return MajorityModel(intent=counts.most_common(1)[0][0])


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    actual_values = actual.astype(str).to_numpy()
    predicted_values = predicted.astype(str).to_numpy()
    accuracy = float((actual_values == predicted_values).mean()) if len(actual_values) else 0.0
    labels = sorted(set(actual_values) | set(predicted_values))
    f1_values: list[float] = []
    for label in labels:
        true_positive = int(((actual_values == label) & (predicted_values == label)).sum())
        false_positive = int(((actual_values != label) & (predicted_values == label)).sum())
        false_negative = int(((actual_values == label) & (predicted_values != label)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "examples": len(actual_values),
    }