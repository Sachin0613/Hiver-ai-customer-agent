"""Prepare clean AmazonHelp conversation data for later project phases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    choose_main_csv,
    column_map,
    discover_data_files,
    load_config,
    resolve_path,
)
from src.preprocessing import (  # noqa: E402
    build_conversation_summary,
    build_customer_company_pairs,
    collect_selected_messages,
    build_relationship_index,
    selected_relationship_table,
    selected_roots,
)


def read_mapping(csv_path: Path) -> dict[str, str]:
    return column_map(list(pd.read_csv(csv_path, nrows=0).columns))


def main() -> int:
    config = load_config()
    paths = config["paths"]
    prep_config = config.get("preprocessing", {})
    selected_brand = str(prep_config["selected_brand"])
    chunksize = int(prep_config.get("chunksize", 200_000))
    raw_dir = resolve_path(paths["raw_dir"])
    processed_dir = resolve_path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = choose_main_csv(discover_data_files(raw_dir))
    if csv_path is None:
        print("No CSV found under data/raw/. See data/raw/README.md")
        return 1
    mapping = read_mapping(csv_path)
    print(f"Dataset: {csv_path.relative_to(ROOT)}")
    print(f"Selected brand: {selected_brand}")
    print(f"Schema fields used: {', '.join(mapping[name] for name in mapping)}")

    index_path = processed_dir / ".phase3_relationships.sqlite"
    input_quality = build_relationship_index(csv_path, mapping, index_path, chunksize)
    raw_record_count = input_quality["raw_record_count"]
    roots = selected_roots(index_path, selected_brand)
    relationships = selected_relationship_table(index_path, roots)
    selected_conversations = set(relationships["conversation_id"].astype(int))
    print(f"Raw records: {raw_record_count:,}")
    print(f"Selected conversations: {len(selected_conversations):,}")

    messages, missing_text = collect_selected_messages(
        csv_path,
        mapping,
        relationships,
        selected_conversations,
        selected_brand,
        chunksize,
    )
    if messages.empty:
        print("No messages found for the selected brand.")
        return 1
    summary = build_conversation_summary(messages, selected_brand)
    pairs = build_customer_company_pairs(messages, selected_brand)

    messages_path = processed_dir / "selected_brand_messages.csv"
    summary_path = processed_dir / "conversations.csv"
    pairs_path = processed_dir / "conversation_pairs.csv"
    messages.to_csv(messages_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    summary.to_csv(summary_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    pairs.to_csv(pairs_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    customer_mask = messages["speaker"].eq("CUSTOMER")
    company_mask = messages["speaker"].eq("COMPANY")
    report = pd.DataFrame(
        [
            ("raw_record_count", raw_record_count),
            ("selected_brand_record_count", len(messages)),
            ("customer_message_count", int(customer_mask.sum())),
            ("company_response_count", int(company_mask.sum())),
            ("conversation_count", len(summary)),
            ("resolved_candidate_count", int(summary["is_resolved_candidate"].sum())),
            ("missing_text_count_selected_brand", missing_text),
            ("duplicate_tweet_id_count", 0),
            ("invalid_tweet_id_count", input_quality["invalid_tweet_id_count"]),
            ("low_quality_message_count", int(messages["is_low_quality"].sum())),
            ("average_conversation_length", float(summary["message_count"].mean())),
            ("median_conversation_length", float(summary["message_count"].median())),
            ("conversation_pair_count", len(pairs)),
        ],
        columns=["metric", "value"],
    )
    report.to_csv(processed_dir / "data_quality_report.csv", index=False)

    metadata = {
        "dataset": str(csv_path.relative_to(ROOT)),
        "selected_brand": selected_brand,
        "schema": mapping,
        "conversation_id_method": "root tweet_id reached by following in_response_to_tweet_id; orphan roots use their own tweet_id",
        "golden_evaluation_set_created": False,
    }
    (processed_dir / "phase3_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    index_path.unlink(missing_ok=True)
    print(f"Wrote {messages_path.relative_to(ROOT)} ({len(messages):,} rows)")
    print(f"Wrote {summary_path.relative_to(ROOT)} ({len(summary):,} conversations)")
    print(f"Wrote {pairs_path.relative_to(ROOT)} ({len(pairs):,} pairs)")
    print(f"Resolved candidates: {int(summary['is_resolved_candidate'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())