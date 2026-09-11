"""Build the Phase 8 historical retrieval index."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_config  # noqa: E402
from src.retrieval import (  # noqa: E402
    DEFAULT_MODEL,
    add_text_batches_to_index,
    load_retrieval_corpus,
    save_index,
)


def main() -> int:
    config = load_config()
    model_name = config.get("retrieval", {}).get("model_name", DEFAULT_MODEL)
    splits = ROOT / "data" / "splits"
    corpus = load_retrieval_corpus(
        splits / "retrieval_corpus.csv",
        splits / "golden_pool.csv",
    )
    batch_size = int(config.get("retrieval", {}).get("batch_size", 32))
    print(f"Encoding {len(corpus):,} retrieval pairs with {model_name} in batches of {batch_size}...", flush=True)
    index, encoded_rows = add_text_batches_to_index(
        (
            f"CUSTOMER: {customer}\nBRAND: {response}"
            for customer, response in zip(corpus["customer_message"], corpus["company_response"])
        ),
        model_name,
        batch_size=batch_size,
    )
    if encoded_rows != len(corpus):
        raise RuntimeError(f"Encoded {encoded_rows} rows but metadata contains {len(corpus)} rows.")
    output_dir = ROOT / "data" / "indexes"
    print("Saving FAISS index", flush=True)
    save_index(index, corpus, output_dir, model_name)
    print("Saving metadata", flush=True)
    print(f"Wrote {index.ntotal:,} vectors to {output_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())