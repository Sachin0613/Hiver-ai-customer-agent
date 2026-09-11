import unittest

import pandas as pd

from src.golden_set import sample_golden_candidates


class GoldenSetTests(unittest.TestCase):
    def pool(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "conversation_id": list(range(1, 31)),
                "tweet_id": list(range(101, 131)),
                "clean_text": ["delivery is late" if i % 2 else "refund requested" for i in range(30)],
                "raw_text": ["delivery is late" if i % 2 else "refund requested" for i in range(30)],
                "timestamp": ["2020-01-01"] * 30,
            }
        )

    def test_sampling_is_reproducible_and_conversation_unique(self) -> None:
        first = sample_golden_candidates(self.pool(), 10, 42)
        second = sample_golden_candidates(self.pool(), 10, 42)
        self.assertEqual(first["conversation_id"].tolist(), second["conversation_id"].tolist())
        self.assertEqual(first["conversation_id"].nunique(), 10)

    def test_gold_fields_are_blank(self) -> None:
        result = sample_golden_candidates(self.pool(), 10, 42)
        self.assertTrue(result["gold_intent"].eq("").all())
        self.assertTrue(result["gold_decision"].eq("").all())


if __name__ == "__main__":
    unittest.main()