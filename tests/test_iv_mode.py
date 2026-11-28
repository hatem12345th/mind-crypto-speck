"""
Tests for the IV-visible CBC setting (adversary sees IV || C) and seed handling.
"""

import os
import tempfile
import unittest
import numpy as np

from crypto.cbc_generator import CBCDatasetGenerator, convert_from_binary_32
from crypto.speck import ror16
from experiments.config import ExperimentConfig
from experiments.dataset_builder import ensure_all_datasets, load_dataset_splits
from experiments.exp_a_deep_analyst import run_experiment_a
from experiments.reporting import summarize_over_seeds


class TestIVVisibleDataset(unittest.TestCase):

    def test_shape_and_layout(self):
        gen = CBCDatasetGenerator(rounds=3, seed=1)
        d = gen.generate_dataset(500, include_iv=True, return_raw_words=True)
        self.assertEqual(d['X'].shape, (1000, 64))
        ivx, ivy = convert_from_binary_32(d['X'][:, :32])
        cx, cy = convert_from_binary_32(d['X'][:, 32:])
        np.testing.assert_array_equal(ivx, d['iv_x'])
        np.testing.assert_array_equal(ivy, d['iv_y'])
        np.testing.assert_array_equal(cx, d['cx'])
        np.testing.assert_array_equal(cy, d['cy'])

    def test_ciphertext_only_mode_unchanged(self):
        gen = CBCDatasetGenerator(rounds=3, seed=1)
        self.assertEqual(gen.generate_dataset(100)['X'].shape, (200, 32))

    def test_one_round_label_is_recoverable_from_iv_and_c(self):
        """After 1 round, y_in = ror2(x_out ^ y_out); y_in = P_y ^ IV_y reveals P's bit 0."""
        gen = CBCDatasetGenerator(rounds=1, seed=3)
        d = gen.generate_dataset(2000, include_iv=True, return_raw_words=True)
        my = ror16(d['cy'] ^ d['cx'], 2)
        recovered = ((my ^ d['iv_y']) & 1).astype(np.uint8)
        np.testing.assert_array_equal(recovered, d['Y'])

    def test_ciphertext_alone_carries_no_label_information(self):
        """With a uniform IV, C is uniform for both plaintexts: per-bit means match."""
        gen = CBCDatasetGenerator(rounds=1, seed=5)
        d = gen.generate_dataset(50_000)
        diff = np.abs(d['X'][d['Y'] == 0].mean(axis=0) - d['X'][d['Y'] == 1].mean(axis=0))
        self.assertLess(diff.max(), 0.02)


class TestConfigVariants(unittest.TestCase):

    def test_paths_differ_by_mode_and_seed(self):
        base = ExperimentConfig()
        iv = ExperimentConfig(include_iv=True)
        other_seed = ExperimentConfig(seed=7)
        paths = {c.dataset_path(5, 'K1') for c in (base, iv, other_seed)}
        self.assertEqual(len(paths), 3)
        self.assertEqual(base.input_channels, 2)
        self.assertEqual(iv.input_channels, 4)


class TestSeedSummary(unittest.TestCase):

    def test_groups_by_setting_and_aggregates_accuracy(self):
        rows = [
            {'rounds': 1, 'n_test': 10, 'accuracy': 0.9, 'roc_auc': 0.95, 'seed': 1},
            {'rounds': 1, 'n_test': 10, 'accuracy': 0.7, 'roc_auc': 0.85, 'seed': 2},
            {'rounds': 2, 'n_test': 10, 'accuracy': 0.5, 'roc_auc': 0.5, 'seed': 1},
        ]
        summary = {r['rounds']: r for r in summarize_over_seeds(rows)}
        self.assertEqual(summary[1]['n_seeds'], 2)
        self.assertAlmostEqual(summary[1]['acc_mean'], 0.8)
        self.assertAlmostEqual(summary[1]['acc_std'], 0.1414, places=4)
        self.assertAlmostEqual(summary[1]['auc_mean'], 0.9)
        self.assertEqual(summary[2]['acc_std'], 0.0)


class TestIVPipeline(unittest.TestCase):

    def test_experiment_a_runs_with_iv_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                data_dir=os.path.join(tmpdir, 'data'),
                checkpoint_dir=os.path.join(tmpdir, 'checkpoints'),
                results_dir=os.path.join(tmpdir, 'results'),
                include_iv=True,
                rounds=[1, 2],
                transitions=[(1, 2)],
                dl_train_per_class=200,
                dl_val_per_class=50,
                dl_test_per_class=50,
                tl_sample_sizes=[50],
                tl_test_per_class=50,
                num_filters=8, d1=16, d2=16, epochs=1, batch_size=64,
            )
            ensure_all_datasets(cfg, verbose=False)
            splits = load_dataset_splits(cfg, rounds=1, key_name='K1', train_per_class=100)
            self.assertEqual(splits['X_train'].shape[1], 64)
            rows = run_experiment_a(cfg, verbose=False)
            self.assertEqual(len(rows), 6)  # Table 1: 4, Table 2: 1, Table 3: 1


if __name__ == '__main__':
    unittest.main()
