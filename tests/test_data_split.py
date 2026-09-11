import unittest

import pandas as pd

from src.data_split import assert_no_overlap, stable_split


class DataSplitTests(unittest.TestCase):
    def test_conversations_are_disjoint(self) -> None:
        assignments = stable_split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], seed=42)
        assert_no_overlap(assignments)
        self.assertEqual(len(assignments), 10)

    def test_same_seed_is_reproducible(self) -> None:
        first = stable_split([1, 2, 3, 4, 5], seed=42)
        second = stable_split([5, 4, 3, 2, 1], seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seed_can_change_assignment(self) -> None:
        first = stable_split(range(100), seed=42)
        second = stable_split(range(100), seed=7)
        self.assertFalse(first["split"].equals(second["split"]))


if __name__ == "__main__":
    unittest.main()