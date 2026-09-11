"""Deterministic golden-set candidate sampling from the isolated golden pool."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.intent_discovery import broad_sampling_bucket


def sample_golden_candidates(pool: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    required = {"conversation_id", "tweet_id", "clean_text", "raw_text", "timestamp"}
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"Golden pool is missing columns: {missing}")
    candidates = pool.copy()
    candidates = candidates.sort_values(["conversation_id", "tweet_id"], kind="mergesort")
    # One example per conversation prevents multi-turn context from dominating the set.
    candidates = candidates.drop_duplicates("conversation_id", keep="first").reset_index(drop=True)
    candidates["sampling_bucket"] = candidates["clean_text"].map(broad_sampling_bucket)
    candidates["text_length"] = candidates["clean_text"].fillna("").str.len()
    candidates["writing_style"] = pd.cut(
        candidates["text_length"],
        bins=[-1, 40, 160, float("inf")],
        labels=["short", "medium", "long"],
    ).astype(str)
    rng = np.random.default_rng(seed)
    groups = list(candidates.groupby(["sampling_bucket", "writing_style"], observed=True, sort=True))
    selected_parts: list[pd.DataFrame] = []
    remaining = target_size
    for index, (_, group) in enumerate(groups):
        groups_left = len(groups) - index
        take = min(len(group), max(1, remaining // groups_left))
        chosen_indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
        selected_parts.append(group.loc[chosen_indices])
        remaining -= take
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.head(0)
    if len(selected) < target_size:
        selected_ids = set(selected["conversation_id"])
        remainder = candidates[~candidates["conversation_id"].isin(selected_ids)]
        take = min(target_size - len(selected), len(remainder))
        if take:
            selected = pd.concat([selected, remainder.iloc[rng.choice(len(remainder), size=take, replace=False)]], ignore_index=True)
    selected = selected.head(target_size).sort_values(["sampling_bucket", "writing_style", "conversation_id"]).reset_index(drop=True)
    result = pd.DataFrame(
        {
            "id": [f"gold_{index:04d}" for index in range(1, len(selected) + 1)],
            "conversation_id": selected["conversation_id"].to_numpy(),
            "source_tweet_id": selected["tweet_id"].to_numpy(),
            "customer_message": selected["clean_text"].to_numpy(),
            "raw_text": selected["raw_text"].to_numpy(),
            "timestamp": selected["timestamp"].to_numpy(),
            "sampling_bucket": selected["sampling_bucket"].to_numpy(),
            "writing_style": selected["writing_style"].to_numpy(),
            "gold_intent": "",
            "gold_decision": "",
            "gold_response_characteristics": "",
            "notes": "",
        }
    )
    return result