import unittest

import pandas as pd

from src.failure_analysis import render_failure_report, summarize_failure_cases, validate_failure_cases


def cases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2"],
            "customer_message": ["Where is it?", "I was charged twice"],
            "expected_output": ["Ask for tracking details", "Escalate payment risk"],
            "actual_output": ["Your package arrived", "Please wait"],
            "failure_category": ["poor_retrieval", "incorrect_escalation"],
            "why_failed": ["Evidence was unrelated", "Rule missed risk"],
            "hypothesis": ["Embedding ambiguity", "Missing keyword"],
            "potential_fix": ["Improve retrieval", "Add rule"],
        }
    )


class FailureAnalysisTests(unittest.TestCase):
    def test_summary_counts_real_categories(self) -> None:
        summary = summarize_failure_cases(cases())
        self.assertEqual(summary["cases"].sum(), 2)

    def test_report_contains_observed_example(self) -> None:
        report = render_failure_report(cases())
        self.assertIn("Where is it?", report)
        self.assertIn("poor_retrieval", report)

    def test_incomplete_case_is_rejected(self) -> None:
        frame = cases()
        frame.loc[0, "hypothesis"] = ""
        with self.assertRaises(ValueError):
            validate_failure_cases(frame)


if __name__ == "__main__":
    unittest.main()