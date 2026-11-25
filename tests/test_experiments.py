"""
Unit and integration tests for MIND-Crypt experiment pipelines:
- Gohr cyclic LR, model checkpointing save/load
- XGBoost Appendix D hyperparameter tuning
- Experiment A & B orchestration and reporting
- Experiment C Gohr differential generator and round sweep
"""

import os
import tempfile
import unittest
import numpy as np
import torch

from models.dataset import create_dataloaders
from models.resnet_distinguisher import (
    ResNetDistinguisher,
    cyclic_lr_factor,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from transfer.xgboost_eval import train_and_eval_xgboost
from crypto.gohr_generator import generate_gohr_differential_dataset
from experiments.config import ExperimentConfig
from experiments.dataset_builder import ensure_all_datasets, load_dataset_splits
from experiments.exp_a_deep_analyst import run_experiment_a
from experiments.exp_b_transfer_analyst import run_experiment_b
from experiments.exp_c_gohr_sweep import run_experiment_c_gohr


class TestModelUpgrades(unittest.TestCase):

    def test_cyclic_lr_bounds(self):
        high, low, cycle = 0.002, 0.0001, 10
        self.assertAlmostEqual(cyclic_lr_factor(0, cycle, high, low), 1.0)
        self.assertAlmostEqual(cyclic_lr_factor(9, cycle, high, low), low / high)
        self.assertAlmostEqual(cyclic_lr_factor(10, cycle, high, low), 1.0)

    def test_checkpoint_save_and_load(self):
        model = ResNetDistinguisher(num_filters=16, depth=1, d1=32, d2=32)
        model.eval()
        x = torch.randint(0, 2, (8, 32), dtype=torch.float32)
        with torch.no_grad():
            expected_out = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, 'test.pt')
            save_checkpoint(model, ckpt, metadata={'epoch': 3})
            loaded, meta = load_checkpoint(ckpt)
            self.assertEqual(meta['epoch'], 3)
            with torch.no_grad():
                actual_out = loaded(x)
            torch.testing.assert_close(actual_out, expected_out)

    def test_train_with_cyclic_and_checkpoint(self):
        rng = np.random.default_rng(7)
        splits = {
            'X_train': rng.integers(0, 2, (200, 32), dtype=np.uint8),
            'y_train': np.tile([0, 1], 100).astype(np.uint8),
            'X_val': rng.integers(0, 2, (100, 32), dtype=np.uint8),
            'y_val': np.tile([0, 1], 50).astype(np.uint8),
        }
        loaders = create_dataloaders(splits, batch_size=64)
        model = ResNetDistinguisher(num_filters=8, depth=1, d1=16, d2=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, 'best.pt')
            hist = train_model(
                model,
                loaders['train'],
                loaders['val'],
                epochs=3,
                scheduler_type='cyclic',
                checkpoint_path=ckpt,
                verbose=False,
            )
            self.assertTrue(os.path.exists(ckpt))
            self.assertEqual(len(hist['lr']), 3)
            self.assertGreaterEqual(hist['best_epoch'], 1)


class TestXGBoostTuning(unittest.TestCase):

    def test_randomized_search_tuning(self):
        rng = np.random.default_rng(11)
        x_train = rng.standard_normal((120, 32)).astype(np.float32)
        y_train = np.tile([0, 1], 60).astype(np.uint8)
        x_test = rng.standard_normal((40, 32)).astype(np.float32)
        y_test = np.tile([0, 1], 20).astype(np.uint8)

        res = train_and_eval_xgboost(
            x_train, y_train, x_test, y_test,
            tune=True, tune_n_iter=2, tune_cv=2,
            xgb_params={'n_estimators': 10},
        )
        self.assertIn('best_params', res)
        self.assertIn('cv_best_score', res)
        self.assertIn('accuracy', res['metrics'])


class TestExperimentPipeline(unittest.TestCase):

    def test_end_to_end_mini_experiments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                data_dir=os.path.join(tmpdir, 'data'),
                checkpoint_dir=os.path.join(tmpdir, 'checkpoints'),
                results_dir=os.path.join(tmpdir, 'results'),
                rounds=[5, 6],
                transitions=[(5, 6), (6, 5)],
                dl_train_per_class=200,
                dl_val_per_class=50,
                dl_test_per_class=50,
                tl_sample_sizes=[50, 100],
                tl_test_per_class=50,
                num_filters=8,
                d1=16,
                d2=16,
                epochs=2,
                batch_size=64,
                xgb_n_estimators=10,
                xgb_max_depth=3,
            )

            paths = ensure_all_datasets(cfg, verbose=False)
            self.assertEqual(len(paths), 4)

            loaded = load_dataset_splits(cfg, rounds=5, key_name='K1', train_per_class=100, val_per_class=50, test_per_class=50)
            self.assertEqual(loaded['X_train'].shape, (200, 32))
            self.assertEqual(np.sum(loaded['y_train'] == 0), 100)
            self.assertEqual(np.sum(loaded['y_train'] == 1), 100)

            exp_a_rows = run_experiment_a(cfg, verbose=False)
            # Table 1 (2*2=4) + Table 2 (2) + Table 3 (2) = 8 rows
            self.assertEqual(len(exp_a_rows), 8)
            self.assertTrue(os.path.exists(os.path.join(cfg.results_dir, 'exp_a_results.csv')))
            self.assertTrue(os.path.exists(os.path.join(cfg.results_dir, 'exp_a_results.json')))

            exp_b_rows = run_experiment_b(cfg, verbose=False)
            # 2 sample sizes * (Table 4: 2 + Table 5: 2 + Table 6: 2) = 12 rows
            self.assertEqual(len(exp_b_rows), 12)
            self.assertTrue(os.path.exists(os.path.join(cfg.results_dir, 'exp_b_results.csv')))
            self.assertTrue(os.path.exists(os.path.join(cfg.results_dir, 'exp_b_results.json')))


class TestGohrSweep(unittest.TestCase):

    def test_gohr_generator_and_sweep(self):
        d = generate_gohr_differential_dataset(n_samples=200, rounds=5, seed=42)
        self.assertEqual(d['X'].shape, (200, 64))
        self.assertEqual(d['Y'].shape, (200,))
        self.assertEqual(int(np.sum(d['Y'] == 0)), 100)
        self.assertEqual(int(np.sum(d['Y'] == 1)), 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                results_dir=os.path.join(tmpdir, 'results'),
                results_tag='gohr_test',
                rounds=[1, 5],
                num_filters=8,
                d1=16,
                d2=16,
                batch_size=64,
            )
            rows = run_experiment_c_gohr(
                cfg, n_train=200, n_val=50, n_test=50, epochs=2, verbose=False
            )
            self.assertEqual(len(rows), 2)
            self.assertIn('roc_auc', rows[0])
            self.assertTrue(os.path.exists(os.path.join(cfg.results_dir, 'exp_c_gohr_gohr_test.csv')))


if __name__ == '__main__':
    unittest.main()
