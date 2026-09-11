"""Discover provisional AmazonHelp intents from development data only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_config, resolve_path  # noqa: E402
from src.intent_discovery import (  # noqa: E402
    INTENT_DEFINITIONS,
    build_annotation_queue,
    topic_counts,
)


def main() -> int:
    config = load_config()
    processed = resolve_path(config["paths"]["processed_dir"])
    development_path = ROOT / "data" / "splits" / "development.csv"
    output_path = processed / "intent_annotation_queue.csv"
    development = pd.read_csv(development_path)
    customers = development[development["speaker"] == "CUSTOMER"]
    seed = int(config["inspection"]["random_seed"])
    discovery = config.get("intent_discovery", {})
    queue = build_annotation_queue(
        development,
        sample_per_bucket=int(discovery.get("sample_per_bucket_style", 20)),
        sample_seed=seed,
    )
    queue.to_csv(output_path, index=False)
    counts = topic_counts(customers["clean_text"])
    counts.to_csv(processed / "intent_topic_counts.csv", index=False)
    payload = {
        "source": "data/splits/development.csv",
        "golden_pool_used": False,
        "customer_messages_analyzed": int(len(customers)),
        "conversations_analyzed": int(customers["conversation_id"].nunique()),
        "taxonomy_status": "provisional_manual_review_required",
        "intent_names": [name for name, *_ in INTENT_DEFINITIONS],
        "annotation_queue_rows": int(len(queue)),
        "seed": seed,
    }
    (processed / "intent_discovery_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Development customer messages analyzed: {len(customers):,}")
    print(f"Provisional intents: {len(INTENT_DEFINITIONS)}")
    print(f"Annotation queue: {len(queue):,} rows")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())