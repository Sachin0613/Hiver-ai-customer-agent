"""Create Phase 4 conversation-level development, retrieval, and golden-pool data."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_split import (  # noqa: E402
    SPLIT_NAMES,
    assert_no_overlap,
    assert_required_columns,
    exact_customer_overlap,
    near_duplicate_summary,
    normalized_text,
    stable_split,
)
from src.data_loader import load_config, resolve_path  # noqa: E402


def main() -> int:
    config = load_config()
    paths = config["paths"]
    split_config = config.get("data_split", {})
    seed = int(config["inspection"]["random_seed"])
    processed = resolve_path(paths["processed_dir"])
    output_dir = ROOT / "data" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    conversations_path = processed / "conversations.csv"
    pairs_path = processed / "conversation_pairs.csv"
    messages_path = processed / "selected_brand_messages.csv"
    conversations = pd.read_csv(conversations_path)
    pairs = pd.read_csv(pairs_path)
    assert_required_columns(conversations, ["conversation_id", "brand", "message_count", "is_resolved_candidate"])
    assert_required_columns(pairs, ["conversation_id", "customer_message", "company_response", "brand"])
    selected_brand = str(split_config.get("selected_brand", conversations["brand"].dropna().iloc[0]))
    if set(conversations["brand"].dropna()) != {selected_brand} or set(pairs["brand"].dropna()) != {selected_brand}:
        raise ValueError("Phase 4 inputs contain a brand different from configured selected_brand.")

    assignments = stable_split(conversations["conversation_id"], seed)
    assert_no_overlap(assignments)
    assignments.to_csv(output_dir / "conversation_assignments.csv", index=False)
    split_by_id = assignments.set_index("conversation_id")["split"]
    conversations["split"] = conversations["conversation_id"].map(split_by_id)
    pairs["split"] = pairs["conversation_id"].map(split_by_id)

    # Stream messages so the largest Phase 3 file is never duplicated in RAM.
    message_columns = ["conversation_id", "tweet_id", "timestamp", "clean_text", "raw_text", "speaker", "message_index", "is_low_quality", "brand", "parent_id"]
    development_path = output_dir / "development.csv"
    golden_path = output_dir / "golden_pool.csv"
    golden_conversations = conversations[conversations["split"] == "golden"].copy()
    golden_metadata = golden_conversations[["conversation_id", "message_count", "customer_message_count", "company_message_count", "is_resolved_candidate"]]
    golden_written = False
    development_hashes: set[str] = set()
    for index, chunk in enumerate(pd.read_csv(messages_path, usecols=message_columns, chunksize=100_000)):
        chunk["split"] = chunk["conversation_id"].map(split_by_id)
        dev_chunk = chunk[chunk["split"] == "development"].drop(columns="split")
        dev_chunk.to_csv(development_path, mode="w" if index == 0 else "a", header=index == 0, index=False)
        dev_customers = dev_chunk[dev_chunk["speaker"] == "CUSTOMER"]
        development_hashes.update(dev_customers["clean_text"].map(normalized_text))
        golden_chunk = chunk[(chunk["split"] == "golden") & (chunk["speaker"] == "CUSTOMER")].drop(columns="split")
        if not golden_chunk.empty:
            golden_chunk = golden_chunk.merge(golden_metadata, on="conversation_id", how="left")
            golden_chunk.to_csv(golden_path, mode="w" if not golden_written else "a", header=not golden_written, index=False)
            golden_written = True

    pairs[pairs["split"] == "retrieval"].drop(columns="split").to_csv(output_dir / "retrieval_corpus.csv", index=False)

    golden_pool = pd.read_csv(golden_path) if golden_written else pd.DataFrame()

    retrieval_hashes = set(pairs.loc[pairs["split"] == "retrieval", "customer_message"].map(normalized_text))
    golden_hashes = set(golden_pool["clean_text"].map(normalized_text))
    overlaps = pd.DataFrame(
        {"text": sorted((development_hashes & retrieval_hashes) | (development_hashes & golden_hashes) | (retrieval_hashes & golden_hashes))}
    )
    overlap_report = pd.DataFrame(
        [
            ("normalized_customer_text_overlap_count", len(overlaps)),
            ("development_customer_unique_texts", len(development_hashes)),
            ("retrieval_customer_unique_texts", len(retrieval_hashes)),
            ("golden_customer_unique_texts", len(golden_hashes)),
        ],
        columns=["metric", "value"],
    )
    overlap_report.to_csv(output_dir / "duplicate_diagnostics.csv", index=False)
    golden_ids = set(golden_pool["conversation_id"])
    retrieval_ids = set(pairs.loc[pairs["split"] == "retrieval", "conversation_id"])
    development_ids = set(conversations.loc[conversations["split"] == "development", "conversation_id"])
    if golden_ids & retrieval_ids or golden_ids & development_ids or retrieval_ids & development_ids:
        raise AssertionError("Conversation leakage detected across split outputs.")

    distribution = []
    for name in SPLIT_NAMES:
        ids = set(assignments.loc[assignments["split"] == name, "conversation_id"])
        conv = conversations[conversations["conversation_id"].isin(ids)]
        pair_count = int(pairs["conversation_id"].isin(ids).sum())
        distribution.append({
            "split": name,
            "conversations": len(ids),
            "customer_messages": int(conv["customer_message_count"].sum()),
            "company_responses": int(conv["company_message_count"].sum()),
            "retrieval_pairs": pair_count,
            "short_conversations": int((conv["message_count"] <= 2).sum()),
            "long_conversations": int((conv["message_count"] >= 7).sum()),
        })
    distribution_frame = pd.DataFrame(distribution)
    distribution_frame.to_csv(output_dir / "split_statistics.csv", index=False)

    first_times = pd.to_datetime(conversations["first_message_time"], utc=True, errors="coerce")
    temporal_cutoff = first_times.quantile(0.85)
    temporal = conversations.assign(first_time=first_times)
    temporal_holdout = temporal[temporal["first_time"] >= temporal_cutoff]
    temporal_summary = pd.DataFrame([{
        "analysis": "newest_15_percent_conversation_holdout",
        "cutoff": temporal_cutoff.isoformat(),
        "conversations": len(temporal_holdout),
        "customer_messages": int(temporal_holdout["customer_message_count"].sum()),
        "company_responses": int(temporal_holdout["company_message_count"].sum()),
        "created_as_output": False,
        "reason": "Random split remains the main evaluation design; this is a robustness comparison only.",
    }])
    temporal_summary.to_csv(output_dir / "temporal_analysis.csv", index=False)

    manifest = {
        "random_seed": seed,
        "selected_brand": selected_brand,
        "development_conversations": int((assignments["split"] == "development").sum()),
        "retrieval_conversations": int((assignments["split"] == "retrieval").sum()),
        "golden_pool_conversations": int((assignments["split"] == "golden").sum()),
        "golden_pool_target_examples": 200,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "split_method": "conversation_level_random_permutation",
        "fractions": {"development": 0.70, "retrieval": 0.15, "golden": 0.15},
        "temporal_holdout_created": False,
        "conversation_overlap": False,
        "golden_retrieval_leakage": False,
        "golden_development_leakage": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(distribution_frame.to_string(index=False))
    print(f"Golden pool examples: {len(golden_pool):,}")
    print(f"Exact normalized customer-message overlaps: {len(overlaps):,}")
    print(f"Wrote Phase 4 outputs to {output_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())