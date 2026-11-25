"""
Feature extraction utility for TransferAnalyst in MIND-Crypt.
"""

from typing import Union
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def extract_latent_features(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int = 5000,
    device: torch.device = torch.device('cpu'),
    feature_level: str = 'flatten'
) -> np.ndarray:
    """
    Extracts latent representations using a pre-trained ResNet model.
    x: (N, 32) binary ciphertext array
    returns: (N, 512) latent features array
    """
    model.eval()
    model.to(device)

    tensor_x = torch.from_numpy(x.astype(np.float32))
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feature_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            feats = model.extract_features(batch_x, feature_level=feature_level)
            feature_list.append(feats.cpu().numpy())

    return np.concatenate(feature_list, axis=0)
