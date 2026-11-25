"""
PyTorch Dataset and DataLoader utilities for MIND-Crypt.
"""

from typing import Tuple, Dict, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CiphertextDataset(Dataset):
    """
    Dataset representing binary ciphertext vectors (N, 32) and labels (N,).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        x: (N, 32) binary array (uint8 or float32)
        y: (N,) labels (0 or 1)
        """
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_dataloaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 5000,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Creates train, val (if present), and test DataLoaders from data_dict.
    """
    loaders = {}

    if 'X_train' in data_dict and 'y_train' in data_dict:
        train_ds = CiphertextDataset(data_dict['X_train'], data_dict['y_train'])
        loaders['train'] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )

    if 'X_val' in data_dict and 'y_val' in data_dict:
        val_ds = CiphertextDataset(data_dict['X_val'], data_dict['y_val'])
        loaders['val'] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

    if 'X_test' in data_dict and 'y_test' in data_dict:
        test_ds = CiphertextDataset(data_dict['X_test'], data_dict['y_test'])
        loaders['test'] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

    return loaders
