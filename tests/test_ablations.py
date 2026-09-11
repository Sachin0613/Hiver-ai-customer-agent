import unittest

import pandas as pd

from evaluation.ablations import summarize_ablation_scores, validate_ablation_scores


def scores() -> pd.DataFrame:
    rows = []
    variants = [
        "llm_without_retrieval",
        "llm_with_retrieval",
        "retrieval_without_intent_filter",
        "retrieval_with_intent_filter",
    ]
    for index, variant in enumerate(variants):
        rows.append({"id": str(index), "variant": variant, "relevance": 4, "groundedness": 4, "helpfulness": 3, "tone": 5, "unsupported_claims": 5})
    return pd.DataFrame(rows)


class AblationTests(unittest.TestCase):
    def test_all_variants_are_summarized(self) -> None:
        result = summarize_ablation_scores(scores())
        self.assertEqual(len(result), 4)
        self.assertIn("groundedness_mean", result.columns)

    def test_missing_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_ablation_scores(scores().iloc[:2])

    def test_invalid_score_is_rejected(self) -> None:
        frame = scores()
        frame.loc[0, "tone"] = 6
        with self.assertRaises(ValueError):
            validate_ablation_scores(frame)


if __name__ == "__main__":
    unittest.main()