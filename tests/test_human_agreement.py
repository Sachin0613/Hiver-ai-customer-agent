import unittest

import pandas as pd

from evaluation.human_agreement import agreement_metrics, validate_ratings


def ratings(size: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"x{index}" for index in range(size)]
    judge = pd.DataFrame({"id": ids})
    human = pd.DataFrame({"id": ids})
    for criterion in ("relevance", "groundedness", "helpfulness", "tone", "unsupported_claims"):
        judge[f"judge_{criterion}"] = 5
        human[f"human_{criterion}"] = 4
    return judge, human


class HumanAgreementTests(unittest.TestCase):
    def test_agreement_metrics_include_exact_and_within_one(self) -> None:
        judge, human = ratings()
        result = agreement_metrics(judge, human)
        self.assertEqual(len(result), 5)
        self.assertTrue((result["exact_agreement"] == 0).all())
        self.assertTrue((result["within_one_point_agreement"] == 1).all())

    def test_fewer_than_forty_examples_are_rejected(self) -> None:
        judge, human = ratings(39)
        with self.assertRaises(ValueError):
            validate_ratings(judge, human)


if __name__ == "__main__":
    unittest.main()