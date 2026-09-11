import unittest

import pandas as pd

from src.preprocessing import (
    build_conversation_summary,
    build_customer_company_pairs,
    clean_text,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = pd.DataFrame(
            {
                "conversation_id": [1, 1],
                "tweet_id": [10, 11],
                "parent_id": [pd.NA, 10],
                "timestamp": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-01T00:01:00Z"], utc=True),
                "speaker": ["CUSTOMER", "COMPANY"],
                "clean_text": ["payment failed", "Please send a DM"],
            }
        )

    def test_clean_text_preserves_content_and_normalizes_spacing(self) -> None:
        self.assertEqual(clean_text("  order\n  123  "), "order 123")

    def test_summary_marks_response_after_customer_as_candidate(self) -> None:
        summary = build_conversation_summary(self.messages, "AmazonHelp")
        self.assertTrue(bool(summary.iloc[0]["is_resolved_candidate"]))

    def test_pairs_reference_existing_tweets(self) -> None:
        pairs = build_customer_company_pairs(self.messages, "AmazonHelp")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["customer_tweet_id"], 10)
        self.assertEqual(pairs.iloc[0]["company_tweet_id"], 11)


if __name__ == "__main__":
    unittest.main()