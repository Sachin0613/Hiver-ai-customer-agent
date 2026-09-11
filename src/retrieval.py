"""Historical customer-support retrieval using Sentence Transformers and FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_retrieval_corpus(path: Path, golden_path: Path | None = None) -> pd.DataFrame:
    corpus = pd.read_csv(path)
    required = {
        "conversation_id", "customer_tweet_id", "customer_message",
        "company_tweet_id", "company_response", "timestamp", "brand",
    }
    missing = sorted(required - set(corpus.columns))
    if missing:
        raise ValueError(f"Retrieval corpus is missing columns: {missing}")
    if corpus["conversation_id"].duplicated().any():
        # Multiple pairs per conversation are valid; tweet pairs must be unique.
        if corpus[["customer_tweet_id", "company_tweet_id"]].duplicated().any():
            raise ValueError("Retrieval corpus contains duplicate customer/company pairs.")
    if golden_path is not None and golden_path.exists():
        golden = pd.read_csv(golden_path, usecols=["conversation_id"])
        overlap = set(corpus["conversation_id"]) & set(golden["conversation_id"])
        if overlap:
            raise ValueError(f"Golden-pool leakage detected for {len(overlap)} conversations.")
    return corpus.reset_index(drop=True)


def retrieval_texts(corpus: pd.DataFrame) -> pd.Series:
    return (
        "CUSTOMER: " + corpus["customer_message"].fillna("").astype(str)
        + "\nBRAND: " + corpus["company_response"].fillna("").astype(str)
    )


def iter_text_batches(texts: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for text in texts:
        batch.append(str(text))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_texts(
    texts: Iterable[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    progress: bool = False,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required to build the retrieval index. "
            "Install requirements.txt first."
        ) from error
    model = SentenceTransformer(model_name)
    batches = iter_text_batches(texts, batch_size)
    encoded_batches: list[np.ndarray] = []
    for batch_number, batch in enumerate(batches, start=1):
        if progress:
            print(f"Encoding batch {batch_number}", flush=True)
        encoded = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        encoded_batches.append(np.asarray(encoded, dtype="float32"))
    if not encoded_batches:
        raise ValueError("No texts were provided for encoding.")
    return np.vstack(encoded_batches).astype("float32", copy=False)


def add_text_batches_to_index(
    texts: Iterable[str],
    model_name: str,
    batch_size: int = 32,
    progress: bool = True,
):
    """Encode and add batches directly to FAISS, keeping only one batch in RAM."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required to build the retrieval index.") from error
    model = SentenceTransformer(model_name)
    index = None
    total = 0
    for batch_number, batch in enumerate(iter_text_batches(texts, batch_size), start=1):
        if progress:
            print(f"Encoding batch {batch_number}", flush=True)
        encoded = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.ascontiguousarray(encoded, dtype="float32")
        if index is None:
            import faiss
            index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        total += len(batch)
    if index is None or total == 0:
        raise ValueError("No texts were provided for encoding.")
    return index, total


def build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "faiss-cpu is required to build the retrieval index. Install requirements.txt first."
        ) from error
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional array.")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    return index


def save_index(index, metadata: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "retrieval.faiss"))
    metadata.to_csv(output_dir / "retrieval_metadata.csv", index=False)
    (output_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "index_type": "IndexFlatIP",
                "normalized_embeddings": True,
                "rows": int(index.ntotal),
                "metadata_file": "retrieval_metadata.csv",
                "golden_pool_included": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class HistoricalRetriever:
    def __init__(self, index, metadata: pd.DataFrame, model_name: str = DEFAULT_MODEL):
        self.index = index
        self.metadata = metadata.reset_index(drop=True)
        self.model_name = model_name

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        vector = encode_texts([query], self.model_name)
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            result = self.metadata.iloc[int(index)].to_dict()
            result["similarity"] = float(score)
            results.append(result)
        return results