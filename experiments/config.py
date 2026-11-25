"""
Central configuration for MIND-Crypt Part 1 experiments (Tables 1-6).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from crypto.speck import PAPER_KEY_1_BYTES, PAPER_KEY_2_BYTES


KEYS: Dict[str, bytes] = {
    'K1': PAPER_KEY_1_BYTES,
    'K2': PAPER_KEY_2_BYTES,
}

# Rounds evaluated in MIND-Crypt Section 6 (Tables 1-6)
PAPER_ROUNDS: List[int] = [5, 6, 21, 22]

# Round transitions evaluated in Tables 2, 3, 5, 6
ROUND_TRANSITIONS: List[Tuple[int, int]] = [(5, 6), (6, 5), (21, 22), (22, 21)]

# Sample sizes per class evaluated in TransferAnalyst (Tables 4, 5, 6)
PAPER_TL_SAMPLE_SIZES: List[int] = [5_000, 10_000, 100_000, 200_000, 290_000]


@dataclass
class ExperimentConfig:
    """Configuration controlling dataset sizes, training, and output paths."""

    data_dir: str = 'data'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    results_tag: str = 'results'

    # Cipher & plaintext settings (Section 5.1)
    rounds: List[int] = field(default_factory=lambda: list(PAPER_ROUNDS))
    transitions: List[Tuple[int, int]] = field(default_factory=lambda: list(ROUND_TRANSITIONS))
    p1: Tuple[int, int] = (0x0000, 0x0000)
    p2: Tuple[int, int] = (0x0000, 0x0001)
    iv_mode: str = 'random'
    seed: int = 42

    # Dataset sizing (per class)
    dl_train_per_class: int = 100_000
    dl_val_per_class: int = 20_000
    dl_test_per_class: int = 50_000
    tl_sample_sizes: List[int] = field(default_factory=lambda: list(PAPER_TL_SAMPLE_SIZES))
    tl_test_per_class: int = 50_000

    # DeepAnalyst ResNet architecture & Gohr cyclic training settings
    num_filters: int = 32
    depth: int = 1
    d1: int = 64
    d2: int = 64
    epochs: int = 10
    batch_size: int = 5000
    high_lr: float = 0.002
    low_lr: float = 0.0001
    cycle_epochs: int = 10
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5

    # TransferAnalyst XGBoost settings (Appendix D)
    tune_xgb: bool = False
    tune_n_iter: int = 10
    tune_cv: int = 3
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6

    def dataset_path(self, rounds: int, key_name: str) -> str:
        return f"{self.data_dir}/speck32_r{rounds}_{key_name.lower()}.npz"

    def checkpoint_path(self, rounds: int, key_name: str = 'K1') -> str:
        return f"{self.checkpoint_dir}/resnet_r{rounds}_{key_name.lower()}.pt"
