import unittest

import pandas as pd

from src.brand_analysis import conversation_labels, pick_candidates


class BrandAnalysisTests(unittest.TestCase):
    def test_conversation_labels_follow_parent_chain(self) -> None:
        tweet_ids = pd.Series([10, 11, 12, 20])
        parent_ids = pd.Series([pd.NA, 10, 11, pd.NA], dtype="Int64")

        labels = conversation_labels(tweet_ids, parent_ids)

        self.assertEqual(len(set(labels[:3])), 1)
        self.assertNotEqual(labels[0], labels[3])

    def test_candidates_balance_volume_and_response_rate(self) -> None:
        stats = pd.DataFrame(
            {
                "brand": ["large", "covered"],
                "resolved_conversations": [100, 90],
                "response_rate": [0.50, 0.90],
            }
        )

        self.assertEqual(pick_candidates(stats, n=1, min_resolved=1), ["covered"])


if __name__ == "__main__":
    unittest.main()