"""Evaluation metrics for intent and escalation predictions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def require_complete_labels(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing evaluation columns: {missing}")
    incomplete = frame[list(columns)].isna().any(axis=1)
    for column in columns:
        incomplete |= frame[column].astype(str).str.strip().eq("")
    if incomplete.any():
        raise ValueError(f"Evaluation labels are incomplete in {int(incomplete.sum())} rows.")
    return frame


def evaluate_intent(actual: pd.Series, predicted: pd.Series) -> tuple[dict, pd.DataFrame]:
    labels = sorted(set(actual.astype(str)) | set(predicted.astype(str)))
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "examples": int(len(actual)),
        "per_intent": report,
    }
    matrix = pd.DataFrame(confusion_matrix(actual, predicted, labels=labels), index=labels, columns=labels)
    return metrics, matrix


def evaluate_escalation(actual: pd.Series, predicted: pd.Series) -> dict:
    labels = ["AUTO_HANDLE", "ESCALATE"]
    actual_values = actual.astype(str)
    predicted_values = predicted.astype(str)
    report = classification_report(actual_values, predicted_values, labels=labels, output_dict=True, zero_division=0)
    should_escalate = actual_values.eq("ESCALATE")
    auto_handled = predicted_values.eq("AUTO_HANDLE")
    false_auto_count = int((should_escalate & auto_handled).sum())
    return {
        "accuracy": float(accuracy_score(actual_values, predicted_values)),
        "precision": float(precision_score(actual_values, predicted_values, pos_label="ESCALATE", zero_division=0)),
        "recall": float(recall_score(actual_values, predicted_values, pos_label="ESCALATE", zero_division=0)),
        "f1": float(f1_score(actual_values, predicted_values, pos_label="ESCALATE", zero_division=0)),
        "false_auto_handling_count": false_auto_count,
        "false_auto_handling_rate": false_auto_count / int(should_escalate.sum()) if should_escalate.sum() else 0.0,
        "examples": int(len(actual_values)),
        "per_decision": report,
    }