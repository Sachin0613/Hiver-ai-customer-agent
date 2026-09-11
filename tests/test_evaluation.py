import unittest

import pandas as pd

from src.evaluation import evaluate_escalation, evaluate_intent, require_complete_labels


class EvaluationTests(unittest.TestCase):
    def test_intent_metrics_include_macro_f1_and_matrix(self) -> None:
        metrics, matrix = evaluate_intent(
            pd.Series(["a", "a", "b"]),
            pd.Series(["a", "b", "b"]),
        )
        self.assertIn("macro_f1", metrics)
        self.assertEqual(matrix.shape, (2, 2))

    def test_false_auto_handling_is_counted(self) -> None:
        metrics = evaluate_escalation(
            pd.Series(["ESCALATE", "AUTO_HANDLE"]),
            pd.Series(["AUTO_HANDLE", "AUTO_HANDLE"]),
        )
        self.assertEqual(metrics["false_auto_handling_count"], 1)
        self.assertEqual(metrics["false_auto_handling_rate"], 1.0)

    def test_incomplete_golden_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_complete_labels(pd.DataFrame({"gold_intent": [""]}), ["gold_intent"])


if __name__ == "__main__":
    unittest.main()