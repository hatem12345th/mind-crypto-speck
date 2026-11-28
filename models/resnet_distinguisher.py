"""
PyTorch 1D Residual Neural Network (ResNet) Distinguisher for MIND-Crypt.

Follows the architecture family of Christoph Gohr (CRYPTO 2019) as used in
MIND-Crypt (arXiv:2405.19683): 1x1 bit-sliced convolution, a residual tower of
kernel-3 Conv1D blocks, and a two-layer dense head.

Note on capacity: this project defaults to depth=1 (one residual block),
whereas Gohr's published distinguishers use depth=10 trained on ~10^7 samples.
Accuracies from the default configuration are therefore lower bounds and are
not directly comparable to Gohr's reported numbers.
"""

import os
from typing import Tuple, Dict, Optional, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class ResBlock1D(nn.Module):
    """Residual Block with 2 Conv1D layers and skip connection."""

    def __init__(self, num_filters: int = 32, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = out + residual
        return out


class ResNetDistinguisher(nn.Module):
    """
    MIND-Crypt / Gohr 1D ResNet Distinguisher.

    Takes 32-bit binary ciphertext vectors, reshapes into (batch, 2 channels, 16 bits),
    processes through 1D Conv + Residual Tower, and extracts features for XGBoost or predicts class.
    With num_channels=4 it accepts 64-bit ciphertext pairs (Gohr's differential setting).

    `depth` is the number of residual blocks (Gohr 2019 uses 10; default here is 1).
    """

    def __init__(
        self,
        num_filters: int = 32,
        depth: int = 1,
        d1: int = 64,
        d2: int = 64,
        num_classes: int = 1,
        word_size: int = 16,
        num_channels: int = 2
    ):
        super().__init__()
        self.num_filters = num_filters
        self.depth = depth
        self.d1 = d1
        self.d2 = d2
        self.num_classes = num_classes
        self.word_size = word_size
        self.num_channels = num_channels  # 2 for single 32-bit block, 4 for 64-bit pair

        # Block 1: Bit-sliced expansion
        # Expands 2 input channels (L and R words) to num_filters using 1x1 conv
        self.conv0 = nn.Conv1d(self.num_channels, num_filters, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm1d(num_filters)
        self.relu0 = nn.ReLU(inplace=True)

        # Block 2: Residual Tower
        self.res_tower = nn.ModuleList([
            ResBlock1D(num_filters=num_filters, kernel_size=3) for _ in range(depth)
        ])

        # Feature extractor dimension: num_filters * word_size (e.g. 32 * 16 = 512)
        self.flatten_dim = num_filters * word_size

        # Block 3: Prediction Head
        self.fc1 = nn.Linear(self.flatten_dim, d1, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(d1)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(d1, d2, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(d2)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(d2, num_classes)

    def extract_features(self, x: torch.Tensor, feature_level: str = 'flatten') -> torch.Tensor:
        """
        Extract latent representations for transfer learning (TransferAnalyst).
        Args:
            x: (batch, 32) float tensor of binary bits
            feature_level: 'flatten' (512-d from conv tower) or 'fc1' (64-d)
        """
        # Reshape (batch, 32) -> (batch, 2, 16)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.num_channels, self.word_size)

        # Conv expansion
        out = self.relu0(self.bn0(self.conv0(x_reshaped)))

        # Residual tower
        for block in self.res_tower:
            out = block(out)

        # Flatten layer (penultimate representation described in Section 5.2)
        features_flat = out.view(batch_size, -1)

        if feature_level == 'flatten':
            return features_flat

        # Optional fc1 features
        h1 = self.relu_fc1(self.bn_fc1(self.fc1(features_flat)))
        return h1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (shape: (batch, 1))."""
        feats = self.extract_features(x, feature_level='flatten')
        h1 = self.relu_fc1(self.bn_fc1(self.fc1(feats)))
        h2 = self.relu_fc2(self.bn_fc2(self.fc2(h1)))
        logits = self.fc_out(h2)
        return logits


def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, float]:
    """
    Computes Accuracy, True Positive Rate (TPR), and True Negative Rate (TNR).
    Class 1 = Positive (P2), Class 0 = Negative (P1).
    """
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': float(acc),
        'tpr': float(tpr),
        'tnr': float(tnr)
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """Evaluate model on a dataloader and return loss, acc, tpr, and tnr."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y_batch.cpu().numpy().flatten().astype(int))

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(all_targets)
    return metrics


def cyclic_lr_factor(epoch_idx: int, cycle_epochs: int = 10, high_lr: float = 0.002, low_lr: float = 0.0001) -> float:
    """
    Gohr (2019) cyclic learning rate multiplier relative to high_lr:
    Linearly decays from high_lr to low_lr over cycle_epochs, then repeats.
    """
    if cycle_epochs <= 1:
        return 1.0
    pos = epoch_idx % cycle_epochs
    target_lr = low_lr + ((cycle_epochs - 1) - pos) / (cycle_epochs - 1) * (high_lr - low_lr)
    return target_lr / high_lr


def save_checkpoint(
    model: ResNetDistinguisher,
    checkpoint_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save ResNetDistinguisher weights and architecture config to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'config': {
            'num_filters': model.num_filters,
            'depth': model.depth,
            'd1': model.d1,
            'd2': model.d2,
            'num_classes': model.num_classes,
            'word_size': model.word_size,
            'num_channels': model.num_channels,
        },
        'metadata': metadata or {},
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu')
) -> Tuple[ResNetDistinguisher, Dict[str, Any]]:
    """Load a saved ResNetDistinguisher checkpoint and return (model, metadata)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload['config']
    model = ResNetDistinguisher(**cfg)
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    model.eval()
    return model, payload.get('metadata', {})


def train_model(
    model: ResNetDistinguisher,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 0.002,
    low_lr: float = 0.0001,
    cycle_epochs: int = 10,
    scheduler_type: str = 'cyclic',
    weight_decay: float = 1e-5,
    checkpoint_path: Optional[str] = None,
    load_best: bool = True,
    early_stopping_patience: Optional[int] = None,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train ResNetDistinguisher using Adam, BCEWithLogitsLoss, cyclic/cosine LR,
    and optional best-validation checkpointing.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cyclic':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: cyclic_lr_factor(ep, cycle_epochs=cycle_epochs, high_lr=lr, low_lr=low_lr)
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=low_lr)
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    criterion = nn.BCEWithLogitsLoss()

    history: Dict[str, Any] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_tpr': [],
        'val_tnr': [],
        'lr': [],
        'best_epoch': 1,
        'best_val_loss': float('inf'),
    }

    best_loss = float('inf')
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            running_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        scheduler.step()

        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        monitored_loss = train_loss
        val_info = ""
        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device=device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_tpr'].append(val_metrics['tpr'])
            history['val_tnr'].append(val_metrics['tnr'])
            monitored_loss = val_metrics['loss']
            val_info = (
                f" | Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.4f} "
                f"(TPR: {val_metrics['tpr']:.4f}, TNR: {val_metrics['tnr']:.4f})"
            )

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            history['best_epoch'] = epoch
            history['best_val_loss'] = float(best_loss)
            epochs_without_improvement = 0
            if load_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                save_checkpoint(
                    model,
                    checkpoint_path,
                    metadata={
                        'epoch': epoch,
                        'best_loss': float(best_loss),
                        'val_metrics': val_metrics,
                    },
                )
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} (lr={current_lr:.5f}) | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}{val_info}"
            )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"Early stopping triggered after {epoch} epochs (best epoch: {history['best_epoch']}).")
            break

    if load_best and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history
