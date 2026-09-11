"""Compare LLM-judge scores with independent human ratings."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


CRITERIA = ("relevance", "groundedness", "helpfulness", "tone", "unsupported_claims")


def validate_ratings(judge: pd.DataFrame, human: pd.DataFrame) -> None:
    required_judge = {"id", *[f"judge_{criterion}" for criterion in CRITERIA]}
    required_human = {"id", *[f"human_{criterion}" for criterion in CRITERIA]}
    missing_judge = sorted(required_judge - set(judge.columns))
    missing_human = sorted(required_human - set(human.columns))
    if missing_judge or missing_human:
        raise ValueError(f"Missing judge columns: {missing_judge}; missing human columns: {missing_human}")
    if judge["id"].duplicated().any() or human["id"].duplicated().any():
        raise ValueError("Rating IDs must be unique within each file.")
    if set(judge["id"]) != set(human["id"]):
        raise ValueError("Judge and human ratings must cover the same IDs.")
    if len(human) < 40:
        raise ValueError("At least 40 independently human-rated examples are required.")
    for frame, prefix in ((judge, "judge_"), (human, "human_")):
        for criterion in CRITERIA:
            values = pd.to_numeric(frame[f"{prefix}{criterion}"], errors="coerce")
            if values.isna().any() or (~values.between(1, 5)).any():
                raise ValueError(f"{prefix}{criterion} must contain only scores from 1 to 5.")


def agreement_metrics(judge: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    validate_ratings(judge, human)
    human_indexed = human.set_index("id")
    rows = []
    for criterion in CRITERIA:
        judge_scores = pd.to_numeric(judge.set_index("id")[f"judge_{criterion}"])
        human_scores = pd.to_numeric(human_indexed[f"human_{criterion}"]).reindex(judge_scores.index)
        differences = (judge_scores - human_scores).abs()
        rows.append(
            {
                "criterion": criterion,
                "examples": len(judge_scores),
                "exact_agreement": float((differences == 0).mean()),
                "within_one_point_agreement": float((differences <= 1).mean()),
                "mean_absolute_difference": float(differences.mean()),
                "pearson_correlation": float(judge_scores.corr(human_scores, method="pearson")),
                "spearman_correlation": float(judge_scores.corr(human_scores, method="spearman")),
            }
        )
    return pd.DataFrame(rows)