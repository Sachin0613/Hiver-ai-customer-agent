import unittest

import pandas as pd

from src.intent_discovery import INTENT_DEFINITIONS, broad_sampling_bucket, build_annotation_queue


class IntentDiscoveryTests(unittest.TestCase):
    def test_taxonomy_is_small_and_named(self) -> None:
        self.assertGreaterEqual(len(INTENT_DEFINITIONS), 6)
        self.assertLessEqual(len(INTENT_DEFINITIONS), 12)
        self.assertEqual(len({name for name, *_ in INTENT_DEFINITIONS}), len(INTENT_DEFINITIONS))

    def test_sampling_bucket_is_not_a_gold_label(self) -> None:
        self.assertEqual(broad_sampling_bucket("My delivery is late"), "delivery_tracking")

    def test_annotation_queue_is_blank_and_development_only(self) -> None:
        frame = pd.DataFrame(
            {
                "conversation_id": [1, 2],
                "tweet_id": [10, 20],
                "speaker": ["CUSTOMER", "CUSTOMER"],
                "clean_text": ["My delivery is late", "I need a refund"],
                "raw_text": ["My delivery is late", "I need a refund"],
                "timestamp": ["2020-01-01", "2020-01-02"],
                "message_index": [0, 0],
            }
        )
        queue = build_annotation_queue(frame, sample_per_bucket=2, sample_seed=42)
        self.assertTrue((queue["intent_label"] == "").all())
        self.assertTrue((queue["annotation_status"] == "unlabeled").all())


if __name__ == "__main__":
    unittest.main()