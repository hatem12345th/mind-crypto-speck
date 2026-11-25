"""
Experiment C: Gohr Differential Distinguisher Sweep Across SPECK32/64 Rounds.

Evaluates the exact same 1D-ResNet architecture (num_channels=4, 64-bit pair input)
on Gohr's differential cryptanalysis task (Delta = (0x0040, 0) over random keys & plaintexts)
across all configured rounds (e.g., 1 through 22).
"""

import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from crypto.gohr_generator import generate_gohr_splits
from models.dataset import create_dataloaders
from models.resnet_distinguisher import (
    ResNetDistinguisher,
    evaluate_model,
    train_model,
)
from .config import ExperimentConfig
from .reporting import print_table, save_results


def _compute_roc_auc(model: ResNetDistinguisher, loader, device: torch.device) -> float:
    """Compute ROC-AUC on a PyTorch DataLoader."""
    model.eval()
    probs_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten()
            probs_list.append(probs)
            targets_list.append(yb.numpy().flatten())
    y_true = np.concatenate(targets_list)
    y_prob = np.concatenate(probs_list)
    return float(roc_auc_score(y_true, y_prob))


def run_experiment_c_gohr(
    cfg: ExperimentConfig,
    n_train: int = 50_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    epochs: int = 8,
    diff: Tuple[int, int] = (0x0040, 0x0000),
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Train and evaluate Gohr's 1D-ResNet differential distinguisher across cfg.rounds.
    """
    records: List[Dict[str, Any]] = []
    table_rows: List[List[Any]] = []

    if verbose:
        print(
            f"\n[Gohr Differential] Sweeping rounds={cfg.rounds} "
            f"(diff=({hex(diff[0])}, {hex(diff[1])}), train={n_train:,}, epochs={epochs})..."
        )

    for r in cfg.rounds:
        t0 = time.time()
        splits = generate_gohr_splits(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            rounds=r,
            diff=diff,
            seed=cfg.seed + r * 37,
        )
        loaders = create_dataloaders(splits, batch_size=cfg.batch_size)

        torch.manual_seed(cfg.seed + r)
        model = ResNetDistinguisher(
            num_filters=cfg.num_filters,
            depth=cfg.depth,
            d1=cfg.d1,
            d2=cfg.d2,
            num_channels=4,
        )
        train_model(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            epochs=epochs,
            lr=cfg.high_lr,
            low_lr=cfg.low_lr,
            cycle_epochs=epochs,
            scheduler_type='cyclic',
            weight_decay=cfg.weight_decay,
            load_best=True,
            early_stopping_patience=4,
            device=device,
            verbose=False,
        )
        metrics = evaluate_model(model, loaders['test'], device=device)
        auc_score = _compute_roc_auc(model, loaders['test'], device=device)
        elapsed = time.time() - t0

        records.append({
            'rounds': r,
            'diff_l': hex(diff[0]),
            'diff_r': hex(diff[1]),
            'n_train': n_train,
            'n_test': n_test,
            'accuracy': round(metrics['accuracy'], 4),
            'tpr': round(metrics['tpr'], 4),
            'tnr': round(metrics['tnr'], 4),
            'roc_auc': round(auc_score, 4),
            'loss': round(metrics['loss'], 4),
            'seconds': round(elapsed, 2),
        })
        table_rows.append([
            r,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['tpr']:.4f}",
            f"{metrics['tnr']:.4f}",
            f"{auc_score:.4f}",
            f"{elapsed:.1f}s",
        ])
        if verbose:
            print(
                f"  R={r:2d} | Acc={metrics['accuracy']:.4f} "
                f"(TPR={metrics['tpr']:.4f}, TNR={metrics['tnr']:.4f}) | "
                f"AUC={auc_score:.4f} [{elapsed:.1f}s]"
            )

    save_results(
        records,
        json_path=os.path.join(cfg.results_dir, f'exp_c_gohr_{cfg.results_tag}.json'),
        csv_path=os.path.join(cfg.results_dir, f'exp_c_gohr_{cfg.results_tag}.csv'),
    )

    if verbose:
        print_table(
            "Experiment C (Gohr Differential Distinguisher): Accuracy & AUC Across Rounds",
            ["Round", "Accuracy", "TPR", "TNR", "ROC-AUC", "Time"],
            table_rows,
        )

    return records
