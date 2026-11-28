"""
Dataset generation and caching utilities across all rounds and keys.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from crypto.cbc_generator import CBCDatasetGenerator
from .config import ExperimentConfig, KEYS


def _required_counts(cfg: ExperimentConfig) -> Tuple[int, int, int]:
    """Compute minimum per-class train, val, and test counts needed by Exp A & Exp B."""
    max_tl_train = max(cfg.tl_sample_sizes) if cfg.tl_sample_sizes else 0
    n_train = max(cfg.dl_train_per_class, max_tl_train)
    n_val = cfg.dl_val_per_class
    n_test = max(cfg.dl_test_per_class, cfg.tl_test_per_class)
    return n_train, n_val, n_test


def _is_cached_dataset_sufficient(
    filepath: str,
    req_train_total: int,
    req_val_total: int,
    req_test_total: int
) -> bool:
    """Return True if an existing .npz file has all splits with >= required rows."""
    if not os.path.exists(filepath):
        return False
    try:
        with np.load(filepath) as npz:
            required_keys = {'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'}
            if not required_keys.issubset(set(npz.files)):
                return False
            return (
                len(npz['y_train']) >= req_train_total
                and len(npz['y_val']) >= req_val_total
                and len(npz['y_test']) >= req_test_total
            )
    except Exception:
        return False


def _seeded_key_round_seed(base_seed: int, rounds: int, key_name: str) -> int:
    """Derive a deterministic per-(round, key) RNG seed so train/test across K1/K2 stay independent."""
    key_offset = sum(ord(ch) for ch in key_name) * 1009
    return base_seed + rounds * 101 + key_offset


def ensure_dataset(
    cfg: ExperimentConfig,
    rounds: int,
    key_name: str,
    force_regenerate: bool = False,
    verbose: bool = True
) -> str:
    """Ensure dataset for (rounds, key_name) exists on disk with sufficient samples."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    filepath = cfg.dataset_path(rounds, key_name)
    n_train, n_val, n_test = _required_counts(cfg)

    if not force_regenerate and _is_cached_dataset_sufficient(
        filepath, 2 * n_train, 2 * n_val, 2 * n_test
    ):
        if verbose:
            print(f"  [cached] R={rounds:2d}, {key_name} -> {filepath}")
        return filepath

    t0 = time.time()
    seed = _seeded_key_round_seed(cfg.seed, rounds, key_name)
    gen = CBCDatasetGenerator(
        key=KEYS[key_name],
        rounds=rounds,
        p1=cfg.p1,
        p2=cfg.p2,
        seed=seed,
    )
    splits = gen.generate_split_dataset(
        train_samples_per_class=n_train,
        test_samples_per_class=n_test,
        val_samples_per_class=n_val,
        iv_mode=cfg.iv_mode,
        include_iv=cfg.include_iv,
    )
    np.savez_compressed(filepath, **splits)
    if verbose:
        total = len(splits['y_train']) + len(splits['y_val']) + len(splits['y_test'])
        print(
            f"  [generated] R={rounds:2d}, {key_name} ({total:,} total samples) "
            f"-> {filepath} ({time.time() - t0:.2f}s)"
        )
    return filepath


def ensure_all_datasets(
    cfg: ExperimentConfig,
    key_names: Optional[List[str]] = None,
    force_regenerate: bool = False,
    verbose: bool = True
) -> Dict[Tuple[int, str], str]:
    """Generate or validate datasets for all configured rounds and keys."""
    target_keys = key_names or list(KEYS.keys())
    paths: Dict[Tuple[int, str], str] = {}
    if verbose:
        n_train, n_val, n_test = _required_counts(cfg)
        print(
            f"Preparing datasets for rounds={cfg.rounds}, keys={target_keys} "
            f"(per-class train={n_train:,}, val={n_val:,}, test={n_test:,})..."
        )
    for r in cfg.rounds:
        for k_name in target_keys:
            paths[(r, k_name)] = ensure_dataset(
                cfg, r, k_name, force_regenerate=force_regenerate, verbose=verbose
            )
    return paths


def _slice_balanced(x: np.ndarray, y: np.ndarray, samples_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select exactly samples_per_class examples per class (0 and 1) and preserve balance."""
    idx0 = np.flatnonzero(y == 0)[:samples_per_class]
    idx1 = np.flatnonzero(y == 1)[:samples_per_class]
    if len(idx0) < samples_per_class or len(idx1) < samples_per_class:
        raise ValueError(
            f"Requested {samples_per_class} samples/class, but dataset only has "
            f"class0={len(idx0)}, class1={len(idx1)}."
        )
    # Interleave or concatenate deterministically (already shuffled within each split)
    sel = np.empty(2 * samples_per_class, dtype=np.int64)
    sel[0::2] = idx0
    sel[1::2] = idx1
    return x[sel], y[sel]


def load_dataset_splits(
    cfg: ExperimentConfig,
    rounds: int,
    key_name: str,
    train_per_class: Optional[int] = None,
    val_per_class: Optional[int] = None,
    test_per_class: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Load dataset splits from disk, optionally slicing exact balanced counts per class."""
    filepath = cfg.dataset_path(rounds, key_name)
    with np.load(filepath) as npz:
        data: Dict[str, np.ndarray] = {}
        if train_per_class is not None:
            data['X_train'], data['y_train'] = _slice_balanced(
                npz['X_train'], npz['y_train'], train_per_class
            )
        else:
            data['X_train'], data['y_train'] = npz['X_train'], npz['y_train']

        if val_per_class is not None and val_per_class > 0:
            data['X_val'], data['y_val'] = _slice_balanced(
                npz['X_val'], npz['y_val'], val_per_class
            )
        elif 'X_val' in npz:
            data['X_val'], data['y_val'] = npz['X_val'], npz['y_val']

        if test_per_class is not None:
            data['X_test'], data['y_test'] = _slice_balanced(
                npz['X_test'], npz['y_test'], test_per_class
            )
        else:
            data['X_test'], data['y_test'] = npz['X_test'], npz['y_test']
    return data
