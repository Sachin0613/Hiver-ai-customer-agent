import unittest

import pandas as pd

from src.majority_baseline import classification_metrics, fit_majority, require_labels


class MajorityBaselineTests(unittest.TestCase):
    def test_majority_model_predicts_one_class(self) -> None:
        frame = pd.DataFrame(
            {
                "clean_text": ["a", "b", "c"],
                "intent_label": ["delivery_tracking", "delivery_tracking", "payment_billing"],
            }
        )
        model = fit_majority(frame)
        self.assertEqual(model.intent, "delivery_tracking")
        self.assertEqual(set(model.predict(frame["clean_text"])), {"delivery_tracking"})

    def test_unlabeled_data_is_rejected(self) -> None:
        frame = pd.DataFrame({"clean_text": ["a"], "intent_label": [""]})
        with self.assertRaises(ValueError):
            require_labels(frame)

    def test_metrics_are_computed_without_external_models(self) -> None:
        metrics = classification_metrics(
            pd.Series(["a", "a", "b"]),
            pd.Series(["a", "b", "b"]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertIn("macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()