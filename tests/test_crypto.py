"""
Unit and integration tests for SPECK32/64 and CBC dataset generation.
"""

import unittest
import numpy as np
from crypto.speck import Speck32_64, check_speck_testvector, expand_key, PAPER_KEY_1_BYTES, PAPER_KEY_2_BYTES
from crypto.cbc_generator import CBCDatasetGenerator, convert_to_binary_32, convert_from_binary_32


class TestSpeck32(unittest.TestCase):

    def test_nsa_test_vector(self):
        """Verify standard NSA test vector."""
        self.assertTrue(check_speck_testvector(), "SPECK32/64 NSA test vector failed!")

    def test_round_keys_length(self):
        """Test key schedule for various rounds."""
        for r in [5, 6, 21, 22]:
            keys = expand_key(PAPER_KEY_1_BYTES, rounds=r)
            self.assertEqual(len(keys), r)

    def test_vectorized_matches_scalar(self):
        """Verify vectorized batch encryption matches single-block encryption."""
        cipher = Speck32_64(key=PAPER_KEY_1_BYTES, rounds=6)
        n = 100
        rng = np.random.default_rng(42)
        x = rng.integers(0, 0x10000, size=n, dtype=np.uint16)
        y = rng.integers(0, 0x10000, size=n, dtype=np.uint16)

        # Vectorized
        vx, vy = cipher.encrypt_batch(x, y)

        # Scalar
        for i in range(n):
            sx, sy = cipher.encrypt_block(int(x[i]), int(y[i]))
            self.assertEqual(vx[i], sx)
            self.assertEqual(vy[i], sy)

    def test_encryption_decryption_invertibility(self):
        """Verify that decryption is exact inverse of encryption."""
        cipher = Speck32_64(key=PAPER_KEY_2_BYTES, rounds=22)
        n = 1000
        rng = np.random.default_rng(123)
        x = rng.integers(0, 0x10000, size=n, dtype=np.uint16)
        y = rng.integers(0, 0x10000, size=n, dtype=np.uint16)

        cx, cy = cipher.encrypt_batch(x, y)
        dx, dy = cipher.decrypt_batch(cx, cy)

        np.testing.assert_array_equal(dx, x)
        np.testing.assert_array_equal(dy, y)


class TestCBCGenerator(unittest.TestCase):

    def test_binary_conversion_roundtrip(self):
        """Verify convert_to_binary_32 and convert_from_binary_32 roundtrip."""
        rng = np.random.default_rng(999)
        n = 500
        cx = rng.integers(0, 0x10000, size=n, dtype=np.uint16)
        cy = rng.integers(0, 0x10000, size=n, dtype=np.uint16)

        bits = convert_to_binary_32(cx, cy)
        self.assertEqual(bits.shape, (n, 32))
        self.assertTrue(np.all((bits == 0) | (bits == 1)))

        rcx, rcy = convert_from_binary_32(bits)
        np.testing.assert_array_equal(rcx, cx)
        np.testing.assert_array_equal(rcy, cy)

    def test_dataset_generation_shape_and_balance(self):
        """Test dataset generation shape, balance, and labels."""
        gen = CBCDatasetGenerator(rounds=5, seed=42)
        n_per_class = 5000
        data = gen.generate_dataset(samples_per_class=n_per_class)

        x = data['X']
        y = data['Y']

        self.assertEqual(x.shape, (2 * n_per_class, 32))
        self.assertEqual(y.shape, (2 * n_per_class,))
        self.assertEqual(np.sum(y == 0), n_per_class)
        self.assertEqual(np.sum(y == 1), n_per_class)
        self.assertTrue(np.all((x == 0) | (x == 1)))

    def test_split_dataset(self):
        """Test train/val/test splitting."""
        gen = CBCDatasetGenerator(rounds=6, seed=101)
        splits = gen.generate_split_dataset(
            train_samples_per_class=1000,
            test_samples_per_class=500,
            val_samples_per_class=200
        )

        self.assertEqual(splits['X_train'].shape, (2000, 32))
        self.assertEqual(splits['y_train'].shape, (2000,))
        self.assertEqual(splits['X_test'].shape, (1000, 32))
        self.assertEqual(splits['y_test'].shape, (1000,))
        self.assertEqual(splits['X_val'].shape, (400, 32))
        self.assertEqual(splits['y_val'].shape, (400,))


if __name__ == '__main__':
    unittest.main()
