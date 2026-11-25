"""
Unit tests for Transfer Learning (feature extraction + XGBoost).
"""

import unittest
import numpy as np
import torch
from models.resnet_distinguisher import ResNetDistinguisher
from transfer.feature_extractor import extract_latent_features
from transfer.xgboost_eval import train_and_eval_xgboost


class TestTransferLearning(unittest.TestCase):

    def setUp(self):
        self.model = ResNetDistinguisher(num_filters=16, depth=1, d1=32, d2=32)

    def test_feature_extraction(self):
        n_samples = 100
        x = np.random.randint(0, 2, size=(n_samples, 32), dtype=np.uint8)
        feats = extract_latent_features(self.model, x, batch_size=32)
        # 16 filters * 16 bits = 256
        self.assertEqual(feats.shape, (n_samples, 256))

    def test_xgboost_fit_predict(self):
        n_train = 200
        n_test = 50
        n_features = 64

        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
        y_train = rng.integers(0, 2, size=n_train, dtype=np.uint8)

        x_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
        y_test = rng.integers(0, 2, size=n_test, dtype=np.uint8)

        res = train_and_eval_xgboost(
            x_train, y_train, x_test, y_test,
            xgb_params={'n_estimators': 10, 'max_depth': 3}
        )

        self.assertIn('metrics', res)
        self.assertIn('accuracy', res['metrics'])
        self.assertEqual(len(res['preds']), n_test)


if __name__ == '__main__':
    unittest.main()
