"""Create the locked Phase 11 golden-set annotation file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_config  # noqa: E402
from src.golden_set import sample_golden_candidates  # noqa: E402


def main() -> int:
    config = load_config()
    seed = int(config["inspection"]["random_seed"])
    golden_config = config.get("golden_set", {})
    target = int(golden_config.get("target_size", 200))
    splits = ROOT / "data" / "splits"
    pool = pd.read_csv(splits / "golden_pool.csv")
    result = sample_golden_candidates(pool, target, seed)
    output = ROOT / "data" / "golden_set.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {
        "source": "data/splits/golden_pool.csv",
        "target_size": target,
        "actual_size": len(result),
        "random_seed": seed,
        "unique_conversations": int(result["conversation_id"].nunique()),
        "labels_completed": False,
        "manual_review_required": True,
        "retrieval_corpus_used": False,
        "development_data_used_for_sampling": False,
    }
    (ROOT / "data" / "golden_set_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {output.relative_to(ROOT)} with {len(result)} unlabeled examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())