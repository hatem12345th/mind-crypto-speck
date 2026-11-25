"""
Unit tests for PyTorch ResNet distinguisher model and training loop.
"""

import unittest
import torch
import numpy as np
from models.resnet_distinguisher import ResNetDistinguisher, compute_metrics, train_model, evaluate_model
from models.dataset import CiphertextDataset, create_dataloaders
from crypto.cbc_generator import CBCDatasetGenerator


class TestResNetDistinguisher(unittest.TestCase):

    def setUp(self):
        self.model = ResNetDistinguisher(num_filters=32, depth=1, d1=64, d2=64)

    def test_forward_pass_shape(self):
        batch_size = 64
        x = torch.zeros((batch_size, 32), dtype=torch.float32)
        out = self.model(x)
        self.assertEqual(out.shape, (batch_size, 1))

    def test_extract_features_shape(self):
        batch_size = 32
        x = torch.zeros((batch_size, 32), dtype=torch.float32)
        feats = self.model.extract_features(x, feature_level='flatten')
        # 32 filters * 16 bits = 512 features
        self.assertEqual(feats.shape, (batch_size, 512))

    def test_compute_metrics(self):
        y_true = np.array([1, 1, 0, 0], dtype=np.uint8)
        y_pred = np.array([1, 0, 0, 1], dtype=np.uint8)
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m['accuracy'], 0.5)
        self.assertAlmostEqual(m['tpr'], 0.5)
        self.assertAlmostEqual(m['tnr'], 0.5)

    def test_quick_train_and_eval(self):
        gen = CBCDatasetGenerator(rounds=5, seed=42)
        splits = gen.generate_split_dataset(
            train_samples_per_class=500,
            test_samples_per_class=200,
            val_samples_per_class=200
        )
        loaders = create_dataloaders(splits, batch_size=256)

        history = train_model(
            self.model,
            loaders['train'],
            loaders['val'],
            epochs=2,
            lr=0.005,
            verbose=False
        )

        self.assertEqual(len(history['train_loss']), 2)
        eval_res = evaluate_model(self.model, loaders['test'])
        self.assertIn('accuracy', eval_res)
        self.assertIn('tpr', eval_res)
        self.assertIn('tnr', eval_res)


if __name__ == '__main__':
    unittest.main()
