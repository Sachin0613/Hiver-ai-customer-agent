"""Validation and reporting for observed evaluation failures."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = (
    "id", "customer_message", "expected_output", "actual_output",
    "failure_category", "why_failed", "hypothesis", "potential_fix",
)


def validate_failure_cases(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Failure cases are missing columns: {missing}")
    if frame.empty:
        raise ValueError("At least one real failure case is required.")
    if frame["id"].duplicated().any():
        raise ValueError("Failure case IDs must be unique.")
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Failure case column must be complete: {column}")
    return frame.copy()


def summarize_failure_cases(frame: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    cases = validate_failure_cases(frame)
    counts = cases["failure_category"].value_counts().head(top_k)
    return counts.rename_axis("failure_category").reset_index(name="cases")


def render_failure_report(frame: pd.DataFrame, top_k: int = 5) -> str:
    cases = validate_failure_cases(frame)
    summary = summarize_failure_cases(cases, top_k)
    lines = [
        "# Failure Analysis",
        "",
        "This report contains only observed failures supplied in the evaluation case file.",
        "",
        "## Top Failure Modes",
        "",
        "| Failure mode | Cases |",
        "| --- | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.failure_category} | {row.cases} |")
    lines.extend(["", "## Cases", ""])
    for row in cases.itertuples(index=False):
        lines.extend(
            [
                f"### {row.id} — {row.failure_category}",
                "",
                f"**Customer message:** {row.customer_message}",
                f"**Expected output:** {row.expected_output}",
                f"**Actual output:** {row.actual_output}",
                f"**Why it failed:** {row.why_failed}",
                f"**Hypothesis:** {row.hypothesis}",
                f"**Potential fix:** {row.potential_fix}",
                "",
            ]
        )
    return "\n".join(lines)