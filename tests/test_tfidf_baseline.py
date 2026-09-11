import unittest

import pandas as pd

from src.tfidf_baseline import evaluate_tfidf, fit_tfidf_baseline, split_labeled_data


def labeled_fixture() -> pd.DataFrame:
    rows = []
    for index in range(20):
        intent = "delivery_tracking" if index % 2 == 0 else "payment_billing"
        text = "delivery package late" if intent == "delivery_tracking" else "credit card payment failed"
        rows.append(
            {
                "annotation_id": f"x{index}",
                "conversation_id": index,
                "tweet_id": index,
                "clean_text": text,
                "intent_label": intent,
            }
        )
    return pd.DataFrame(rows)


class TfidfBaselineTests(unittest.TestCase):
    def test_stratified_split_is_reproducible(self) -> None:
        first_train, first_validation = split_labeled_data(labeled_fixture(), 42)
        second_train, second_validation = split_labeled_data(labeled_fixture(), 42)
        self.assertEqual(first_train["annotation_id"].tolist(), second_train["annotation_id"].tolist())
        self.assertEqual(first_validation["annotation_id"].tolist(), second_validation["annotation_id"].tolist())

    def test_model_returns_metrics_and_predictions(self) -> None:
        train, validation = split_labeled_data(labeled_fixture(), 42)
        model = fit_tfidf_baseline(train, 42)
        metrics, matrix, predictions = evaluate_tfidf(model, validation)
        self.assertIn("accuracy", metrics)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(len(predictions), len(validation))


if __name__ == "__main__":
    unittest.main()