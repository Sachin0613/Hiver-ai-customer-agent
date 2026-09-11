"""Reproducible conversation-level splitting and leakage checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SPLIT_NAMES = ("development", "retrieval", "golden")
SPLIT_FRACTIONS = {"development": 0.70, "retrieval": 0.15, "golden": 0.15}
TEXT_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SplitResult:
    assignments: pd.DataFrame
    stats: pd.DataFrame


def normalized_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    return TEXT_RE.sub(" ", text).strip()


def stable_split(conversation_ids: Iterable[object], seed: int) -> pd.DataFrame:
    """Assign each conversation exactly once using a deterministic RNG."""
    ids = pd.Series(list(conversation_ids), name="conversation_id").drop_duplicates()
    ids = ids.sort_values(kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    labels = np.empty(len(ids), dtype=object)
    n_development = int(len(ids) * SPLIT_FRACTIONS["development"])
    n_retrieval = int(len(ids) * SPLIT_FRACTIONS["retrieval"])
    labels[order[:n_development]] = "development"
    labels[order[n_development:n_development + n_retrieval]] = "retrieval"
    labels[order[n_development + n_retrieval:]] = "golden"
    return pd.DataFrame({"conversation_id": ids, "split": labels})


def assert_no_overlap(assignments: pd.DataFrame) -> None:
    duplicated = assignments["conversation_id"].duplicated(keep=False)
    if duplicated.any():
        raise AssertionError("A conversation was assigned to multiple splits.")
    if set(assignments["split"]) != set(SPLIT_NAMES):
        raise AssertionError("All three split labels must be present.")


def assert_required_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AssertionError("Missing required columns: " + ", ".join(missing))


def exact_customer_overlap(
    development: pd.DataFrame,
    retrieval: pd.DataFrame,
    golden: pd.DataFrame,
    text_column: str = "customer_message",
) -> pd.DataFrame:
    frames = []
    for name, frame in (("development", development), ("retrieval", retrieval), ("golden", golden)):
        if text_column not in frame:
            continue
        values = frame[text_column].map(normalized_text)
        frames.append(pd.DataFrame({"text": values, "split": name}))
    if not frames:
        return pd.DataFrame(columns=["text", "splits"])
    all_text = pd.concat(frames, ignore_index=True)
    overlap = all_text.groupby("text")["split"].agg(lambda values: sorted(set(values)))
    overlap = overlap[overlap.map(len) > 1]
    return overlap.rename("splits").reset_index()


def text_signature(value: object, n: int = 3) -> frozenset[str]:
    text = normalized_text(value)
    return frozenset(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def near_duplicate_summary(frames: dict[str, pd.DataFrame], text_column: str) -> dict[str, int]:
    """Report exact normalized matches and cheap signature matches across splits."""
    signatures: dict[str, set[frozenset[str]]] = {}
    for name, frame in frames.items():
        if text_column in frame:
            signatures[name] = {text_signature(value) for value in frame[text_column].dropna() if normalized_text(value)}
    pairs: dict[str, int] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            if left not in signatures or right not in signatures:
                continue
            pairs[f"{left}_vs_{right}_signature_overlap"] = len(signatures[left] & signatures[right])
    return pairs


def dataframe_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].sort_values(columns).astype(str).to_csv(index=False).encode()
    return hashlib.sha256(values).hexdigest()