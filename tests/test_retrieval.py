import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.retrieval import build_faiss_index, iter_text_batches, load_retrieval_corpus, retrieval_texts


class RetrievalTests(unittest.TestCase):
    def corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "conversation_id": [1, 2],
                "customer_tweet_id": [10, 20],
                "customer_message": ["delivery is late", "payment failed"],
                "company_tweet_id": [11, 21],
                "company_response": ["check tracking", "check payment method"],
                "timestamp": ["2020-01-01", "2020-01-02"],
                "brand": ["AmazonHelp", "AmazonHelp"],
            }
        )

    def test_golden_conversations_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.csv"
            golden_path = root / "golden.csv"
            self.corpus().to_csv(corpus_path, index=False)
            pd.DataFrame({"conversation_id": [2]}).to_csv(golden_path, index=False)
            with self.assertRaises(ValueError):
                load_retrieval_corpus(corpus_path, golden_path)

    def test_faiss_index_search_returns_metadata_order(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        index = build_faiss_index(embeddings)
        scores, indices = index.search(np.asarray([[1.0, 0.0]], dtype="float32"), 1)
        self.assertEqual(int(indices[0][0]), 0)
        self.assertAlmostEqual(float(scores[0][0]), 1.0)

    def test_retrieval_text_contains_both_sides(self) -> None:
        text = retrieval_texts(self.corpus()).iloc[0]
        self.assertIn("delivery is late", text)
        self.assertIn("check tracking", text)

    def test_text_batches_cover_all_rows_without_dropping_order(self) -> None:
        batches = list(iter_text_batches(["a", "b", "c", "d", "e"], batch_size=2))
        self.assertEqual(batches, [["a", "b"], ["c", "d"], ["e"]])


if __name__ == "__main__":
    unittest.main()