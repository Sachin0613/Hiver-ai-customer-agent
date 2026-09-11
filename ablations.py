"""Ablation-result validation and aggregation."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


VARIANTS = (
    "llm_without_retrieval",
    "llm_with_retrieval",
    "retrieval_without_intent_filter",
    "retrieval_with_intent_filter",
)
SCORE_COLUMNS = ("relevance", "groundedness", "helpfulness", "tone", "unsupported_claims")


def validate_ablation_scores(frame: pd.DataFrame, required_variants: Iterable[str] = VARIANTS) -> None:
    required = {"id", "variant", *SCORE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ablation scores are missing columns: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError("Ablation IDs must be unique.")
    unknown = set(frame["variant"]) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown ablation variants: {sorted(unknown)}")
    absent = set(required_variants) - set(frame["variant"])
    if absent:
        raise ValueError(f"Missing ablation variants: {sorted(absent)}")
    for column in SCORE_COLUMNS:
        scores = pd.to_numeric(frame[column], errors="coerce")
        if scores.isna().any() or (~scores.between(1, 5)).any():
            raise ValueError(f"{column} must contain scores from 1 to 5.")


def summarize_ablation_scores(frame: pd.DataFrame) -> pd.DataFrame:
    validate_ablation_scores(frame)
    summary = frame.groupby("variant", sort=True)[list(SCORE_COLUMNS)].agg(["mean", "count"])
    summary.columns = [f"{criterion}_{stat}" for criterion, stat in summary.columns]
    return summary.reset_index()