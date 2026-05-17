from __future__ import annotations

import unittest

from tooling.data_analysis.stats import SampleStats


class SampleStatsTests(unittest.TestCase):
    def test_empty_stats_return_none_for_numeric_properties(self) -> None:
        stats = SampleStats.from_values([None, None])

        self.assertEqual(stats.values, ())
        self.assertEqual(stats.count, 0)
        self.assertIsNone(stats.median)
        self.assertIsNone(stats.min)
        self.assertIsNone(stats.max)
        self.assertIsNone(stats.total)

    def test_stats_filter_none_and_convert_to_float(self) -> None:
        stats = SampleStats.from_values([3, None, 1.0, 2])

        self.assertEqual(stats.values, (3.0, 1.0, 2.0))
        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.median, 2.0)
        self.assertEqual(stats.min, 1.0)
        self.assertEqual(stats.max, 3.0)
        self.assertEqual(stats.total, 6.0)

    def test_even_count_median(self) -> None:
        stats = SampleStats.from_values([10, 2, 4, 8])

        self.assertEqual(stats.median, 6.0)


if __name__ == "__main__":
    unittest.main()
