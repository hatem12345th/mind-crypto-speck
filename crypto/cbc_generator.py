"""
Dataset Generator for MIND-Crypt (arXiv:2405.19683).

Generates datasets of SPECK32/64 ciphertexts encrypted in CBC mode
for assessing cryptographic indistinguishability between two plaintexts
P1 and P2 differing by a single bit.
"""

from typing import Tuple, Optional, Dict, Union, List
import os
import numpy as np
from .speck import Speck32_64, WORD_SIZE, PAPER_KEY_1_BYTES, PAPER_KEY_2_BYTES


def convert_to_binary_32(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """
    Convert N pairs of 16-bit ciphertext words (cx, cy) to an (N, 32) binary matrix.
    Follows Christoph Gohr's MSB-first bit extraction convention:
    Indices 0..15: bits of cx (MSB down to LSB)
    Indices 16..31: bits of cy (MSB down to LSB)
    """
    n = len(cx)
    out = np.zeros((32, n), dtype=np.uint8)
    words = [cx, cy]
    for i in range(32):
        word_idx = i // WORD_SIZE
        bit_offset = WORD_SIZE - 1 - (i % WORD_SIZE)
        out[i] = (words[word_idx] >> bit_offset) & 1
    # Transpose to (N, 32)
    return out.T


def convert_from_binary_32(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of convert_to_binary_32. Reconstructs (cx, cy) from (N, 32) binary matrix."""
    n = bits.shape[0]
    cx = np.zeros(n, dtype=np.uint16)
    cy = np.zeros(n, dtype=np.uint16)
    for i in range(16):
        shift = 15 - i
        cx |= (bits[:, i].astype(np.uint16) << shift)
        cy |= (bits[:, 16 + i].astype(np.uint16) << shift)
    return cx, cy


class CBCDatasetGenerator:
    """
    Generator for MIND-Crypt indistinguishability datasets.
    """

    def __init__(
        self,
        key: Union[bytes, Tuple[int, int, int, int], List[int], np.ndarray] = PAPER_KEY_1_BYTES,
        rounds: int = 22,
        p1: Tuple[int, int] = (0x0000, 0x0000),
        p2: Tuple[int, int] = (0x0000, 0x0001),
        seed: Optional[int] = None
    ):
        """
        key: 64-bit master key (defaults to K1 from Table 7)
        rounds: number of SPECK rounds (e.g. 5, 6, 21, 22)
        p1: 32-bit plaintext for class 0 (two 16-bit words)
        p2: 32-bit plaintext for class 1 (differs by single bit, e.g. bit 0)
        """
        self.rounds = rounds
        self.p1 = p1
        self.p2 = p2
        self.cipher = Speck32_64(key=key, rounds=rounds)
        self.rng = np.random.default_rng(seed)

    def set_key(self, key: Union[bytes, Tuple[int, int, int, int], List[int], np.ndarray], rounds: Optional[int] = None):
        """Update cipher key and rounds."""
        if rounds is not None:
            self.rounds = rounds
        self.cipher.set_key(key, self.rounds)

    def generate_ciphertexts(
        self,
        pt: Tuple[int, int],
        n_samples: int,
        iv_mode: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate n_samples of CBC ciphertexts for a given plaintext.
        Returns:
            cx, cy: Ciphertext 16-bit word arrays (shape: (n_samples,))
            iv_x, iv_y: Initialization vector arrays (shape: (n_samples,))
        """
        if iv_mode == 'random':
            iv_x = self.rng.integers(0, 0x10000, size=n_samples, dtype=np.uint16)
            iv_y = self.rng.integers(0, 0x10000, size=n_samples, dtype=np.uint16)
        elif iv_mode == 'fixed':
            iv_x = np.zeros(n_samples, dtype=np.uint16)
            iv_y = np.zeros(n_samples, dtype=np.uint16)
        elif iv_mode == 'counter':
            indices = np.arange(n_samples, dtype=np.uint32)
            iv_x = (indices >> 16).astype(np.uint16)
            iv_y = (indices & 0xFFFF).astype(np.uint16)
        else:
            raise ValueError(f"Unknown iv_mode: {iv_mode}")

        # CBC: M = P XOR IV
        mx = (pt[0] ^ iv_x).astype(np.uint16)
        my = (pt[1] ^ iv_y).astype(np.uint16)

        # C = Encrypt(M)
        cx, cy = self.cipher.encrypt_batch(mx, my, rounds=self.rounds)
        return cx, cy, iv_x, iv_y

    def generate_dataset(
        self,
        samples_per_class: int,
        iv_mode: str = 'random',
        shuffle: bool = True,
        return_raw_words: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate balanced dataset for MIND-Crypt:
        samples_per_class from P1 (label 0)
        samples_per_class from P2 (label 1)

        Returns dictionary:
            'X': (2*N, 32) binary array (uint8 or float32)
            'Y': (2*N,) labels array (uint8: 0 or 1)
            (Optional raw words 'cx', 'cy', 'iv_x', 'iv_y')
        """
        c1x, c1y, iv1x, iv1y = self.generate_ciphertexts(self.p1, samples_per_class, iv_mode)
        c2x, c2y, iv2x, iv2y = self.generate_ciphertexts(self.p2, samples_per_class, iv_mode)

        cx = np.concatenate([c1x, c2x])
        cy = np.concatenate([c1y, c2y])
        y = np.concatenate([
            np.zeros(samples_per_class, dtype=np.uint8),
            np.ones(samples_per_class, dtype=np.uint8)
        ])

        x_bits = convert_to_binary_32(cx, cy)

        if shuffle:
            perm = self.rng.permutation(len(y))
            x_bits = x_bits[perm]
            y = y[perm]
            cx = cx[perm]
            cy = cy[perm]

        result = {
            'X': x_bits,
            'Y': y
        }
        if return_raw_words:
            result['cx'] = cx
            result['cy'] = cy

        return result

    def generate_split_dataset(
        self,
        train_samples_per_class: int,
        test_samples_per_class: int,
        val_samples_per_class: int = 0,
        iv_mode: str = 'random'
    ) -> Dict[str, np.ndarray]:
        """Generate train, val, and test partitions."""
        train_data = self.generate_dataset(train_samples_per_class, iv_mode=iv_mode, shuffle=True)
        test_data = self.generate_dataset(test_samples_per_class, iv_mode=iv_mode, shuffle=True)

        res = {
            'X_train': train_data['X'],
            'y_train': train_data['Y'],
            'X_test': test_data['X'],
            'y_test': test_data['Y'],
        }

        if val_samples_per_class > 0:
            val_data = self.generate_dataset(val_samples_per_class, iv_mode=iv_mode, shuffle=True)
            res['X_val'] = val_data['X']
            res['y_val'] = val_data['Y']

        return res

    def save_dataset_to_disk(self, dataset: Dict[str, np.ndarray], output_path: str):
        """Save dataset to a compressed npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(output_path, **dataset)
        print(f"Dataset saved successfully to: {output_path}")
