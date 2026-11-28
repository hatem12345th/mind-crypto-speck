"""
Unit tests for statistical reporting helpers (no PyTorch required).
"""

import unittest

from experiments.reporting import accuracy_significance, annotate_significance


class TestAccuracySignificance(unittest.TestCase):

    def test_chance_accuracy_is_not_significant(self):
        sig = accuracy_significance(0.5, 20_000)
        self.assertAlmostEqual(sig['p_vs_chance'], 1.0)
        self.assertLess(sig['ci95_low'], 0.5)
        self.assertGreater(sig['ci95_high'], 0.5)

    def test_ci_width_matches_binomial_standard_error(self):
        # Half-width at p=0.5, n=10,000 is ~1.96 * 0.005 = 0.0098
        sig = accuracy_significance(0.5, 10_000)
        self.assertAlmostEqual(sig['ci95_high'] - sig['ci95_low'], 0.0196, places=3)

    def test_strong_distinguisher_is_significant(self):
        sig = accuracy_significance(0.6617, 10_000)
        self.assertLess(sig['p_vs_chance'], 1e-10)
        self.assertGreater(sig['ci95_low'], 0.65)

    def test_ci_stays_in_unit_interval(self):
        sig = accuracy_significance(1.0, 10_000)
        self.assertLessEqual(sig['ci95_high'], 1.0)
        self.assertGreater(sig['ci95_low'], 0.999)

    def test_annotate_adds_fields_to_every_row(self):
        rows = [{'accuracy': 0.49}, {'accuracy': 0.51}]
        annotate_significance(rows, 20_000)
        for row in rows:
            for key in ('n_test', 'ci95_low', 'ci95_high', 'p_vs_chance'):
                self.assertIn(key, row)

    def test_rejects_empty_test_set(self):
        with self.assertRaises(ValueError):
            accuracy_significance(0.5, 0)


if __name__ == '__main__':
    unittest.main()
