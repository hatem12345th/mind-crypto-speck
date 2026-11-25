"""
Gohr (CRYPTO 2019) Differential Cryptanalysis Dataset Generator for SPECK32/64.

Generates 64-bit binary ciphertext pair vectors (C0, C1) where:
  - Label 1 (real difference): P1 = P0 XOR diff, both encrypted under the same random key
  - Label 0 (random pair):     P1 sampled independently at random
"""

from typing import Dict, Optional, Tuple
import numpy as np
from .speck import enc_round
from .cbc_generator import convert_to_binary_32


def generate_gohr_differential_dataset(
    n_samples: int,
    rounds: int = 5,
    diff: Tuple[int, int] = (0x0040, 0x0000),
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate N ciphertext pair vectors of shape (N, 64) and binary labels (N,).
    Each sample uses an independently drawn 64-bit key and 32-bit plaintext.
    """
    rng = np.random.default_rng(seed)

    # Balanced binary labels (50% class 0, 50% class 1)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    y = np.concatenate([
        np.zeros(n_neg, dtype=np.uint8),
        np.ones(n_pos, dtype=np.uint8),
    ])
    perm = rng.permutation(n_samples)
    y = y[perm]

    keys = rng.integers(0, 0x10000, size=(4, n_samples), dtype=np.uint16)
    p0l = rng.integers(0, 0x10000, size=n_samples, dtype=np.uint16)
    p0r = rng.integers(0, 0x10000, size=n_samples, dtype=np.uint16)

    p1l = p0l ^ np.uint16(diff[0])
    p1r = p0r ^ np.uint16(diff[1])

    rand_mask = (y == 0)
    n_rand = int(np.sum(rand_mask))
    p1l[rand_mask] = rng.integers(0, 0x10000, size=n_rand, dtype=np.uint16)
    p1r[rand_mask] = rng.integers(0, 0x10000, size=n_rand, dtype=np.uint16)

    # Vectorized SPECK32/64 key schedule across all N keys
    ks = [keys[3].copy()]
    l = [keys[2].copy(), keys[1].copy(), keys[0].copy()]
    for i in range(rounds - 1):
        nl, nrk = enc_round(l[i % 3], ks[i], np.full(n_samples, i, dtype=np.uint16))
        l[i % 3] = nl
        ks.append(nrk)

    c0l, c0r = p0l.copy(), p0r.copy()
    c1l, c1r = p1l.copy(), p1r.copy()
    for rk in ks:
        c0l, c0r = enc_round(c0l, c0r, rk)
        c1l, c1r = enc_round(c1l, c1r, rk)

    x_bits = np.concatenate([
        convert_to_binary_32(c0l, c0r),
        convert_to_binary_32(c1l, c1r),
    ], axis=1)

    return {'X': x_bits, 'Y': y}


def generate_gohr_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    rounds: int = 5,
    diff: Tuple[int, int] = (0x0040, 0x0000),
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate disjoint train, val, and test splits for Gohr differential analysis."""
    tr = generate_gohr_differential_dataset(n_train, rounds=rounds, diff=diff, seed=seed)
    va = generate_gohr_differential_dataset(n_val, rounds=rounds, diff=diff, seed=seed + 10_000)
    te = generate_gohr_differential_dataset(n_test, rounds=rounds, diff=diff, seed=seed + 20_000)
    return {
        'X_train': tr['X'], 'y_train': tr['Y'],
        'X_val': va['X'], 'y_val': va['Y'],
        'X_test': te['X'], 'y_test': te['Y'],
    }
