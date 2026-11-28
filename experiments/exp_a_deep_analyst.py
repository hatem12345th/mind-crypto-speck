"""
Experiment A: DeepAnalyst (Deep Learning Indistinguishability Evaluation).

Covers MIND-Crypt Section 6 Tables 1, 2, and 3 across rounds {5, 6, 21, 22}
and keys {K1, K2}:
  - Table 1: Same Round, Same Key & Same Round, Different Key
  - Table 2: Different Round, Same Key
  - Table 3: Different Round, Different Key
"""

import os
import time
from typing import Any, Dict, List, Tuple
import torch

from models.dataset import create_dataloaders
from models.resnet_distinguisher import (
    ResNetDistinguisher,
    evaluate_model,
    load_checkpoint,
    train_model,
)
from .config import ExperimentConfig
from .dataset_builder import load_dataset_splits
from .reporting import annotate_significance, print_table, save_results


def _eval_model_on_target(
    model: ResNetDistinguisher,
    cfg: ExperimentConfig,
    target_round: int,
    target_key: str,
    device: torch.device
) -> Dict[str, float]:
    """Load test split for (target_round, target_key) and evaluate model."""
    splits = load_dataset_splits(
        cfg,
        rounds=target_round,
        key_name=target_key,
        train_per_class=0,
        val_per_class=0,
        test_per_class=cfg.dl_test_per_class,
    )
    test_loader = create_dataloaders(
        {'X_test': splits['X_test'], 'y_test': splits['y_test']},
        batch_size=cfg.batch_size,
    )['test']
    return evaluate_model(model, test_loader, device=device)


def _train_or_load_base_models(
    cfg: ExperimentConfig,
    device: torch.device,
    force_retrain: bool = False,
    verbose: bool = True
) -> Tuple[Dict[int, ResNetDistinguisher], Dict[int, Dict[str, Any]]]:
    """Train (or load cached checkpoint) DeepAnalyst ResNet on K1 for each configured round."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    models: Dict[int, ResNetDistinguisher] = {}
    histories: Dict[int, Dict[str, Any]] = {}

    for r in cfg.rounds:
        ckpt_path = cfg.checkpoint_path(r, 'K1')
        if not force_retrain and os.path.exists(ckpt_path):
            model, meta = load_checkpoint(ckpt_path, device=device)
            models[r] = model
            histories[r] = meta
            if verbose:
                print(f"[DeepAnalyst] Loaded checkpoint for R={r:2d}, K1 -> {ckpt_path}")
            continue

        if verbose:
            print(f"\n[DeepAnalyst] Training ResNet for R={r:2d}, K1 ({2 * cfg.dl_train_per_class:,} samples, {cfg.epochs} epochs)...")
        t0 = time.time()
        splits = load_dataset_splits(
            cfg,
            rounds=r,
            key_name='K1',
            train_per_class=cfg.dl_train_per_class,
            val_per_class=cfg.dl_val_per_class,
            test_per_class=cfg.dl_test_per_class,
        )
        loaders = create_dataloaders(splits, batch_size=cfg.batch_size)

        torch.manual_seed(cfg.seed + r)
        model = ResNetDistinguisher(
            num_filters=cfg.num_filters,
            depth=cfg.depth,
            d1=cfg.d1,
            d2=cfg.d2,
        )
        hist = train_model(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders.get('val'),
            epochs=cfg.epochs,
            lr=cfg.high_lr,
            low_lr=cfg.low_lr,
            cycle_epochs=cfg.cycle_epochs,
            scheduler_type='cyclic',
            weight_decay=cfg.weight_decay,
            checkpoint_path=ckpt_path,
            load_best=True,
            early_stopping_patience=cfg.early_stopping_patience,
            device=device,
            verbose=verbose,
        )
        hist['train_seconds'] = round(time.time() - t0, 2)
        models[r] = model
        histories[r] = hist

    return models, histories


def run_experiment_a(
    cfg: ExperimentConfig,
    device: torch.device = torch.device('cpu'),
    force_retrain: bool = False,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Execute Experiment A across all 4 matrix settings and save CSV/JSON tables.
    """
    models, _ = _train_or_load_base_models(
        cfg, device=device, force_retrain=force_retrain, verbose=verbose
    )

    records: List[Dict[str, Any]] = []
    same_round_same_key_acc: Dict[int, float] = {}

    # 1. Table 1: Same Round, Same Key & Same Round, Different Key
    t1_rows = []
    for r in cfg.rounds:
        m_same = _eval_model_on_target(models[r], cfg, r, 'K1', device)
        m_diff = _eval_model_on_target(models[r], cfg, r, 'K2', device)
        same_round_same_key_acc[r] = m_same['accuracy']

        for scenario, target_key, m in [
            ('Same Round, Same Key', 'K1', m_same),
            ('Same Round, Different Key', 'K2', m_diff),
        ]:
            records.append({
                'table': 'Table 1',
                'scenario': scenario,
                'train_round': r,
                'train_key': 'K1',
                'eval_round': r,
                'eval_key': target_key,
                'accuracy': round(m['accuracy'], 4),
                'tpr': round(m['tpr'], 4),
                'tnr': round(m['tnr'], 4),
                'loss': round(m['loss'], 4),
            })

        t1_rows.append([
            r,
            f"{m_same['accuracy']:.4f}", f"{m_same['tpr']:.4f}", f"{m_same['tnr']:.4f}",
            f"{m_diff['accuracy']:.4f}", f"{m_diff['tpr']:.4f}", f"{m_diff['tnr']:.4f}",
        ])

    # Filter transitions to only rounds present in cfg.rounds
    active_transitions = [
        (src_r, dst_r) for (src_r, dst_r) in cfg.transitions
        if src_r in models and dst_r in cfg.rounds
    ]

    # 2. Table 2: Different Round, Same Key (K1 -> K1)
    # 3. Table 3: Different Round, Different Key (K1 -> K2)
    t2_rows = []
    t3_rows = []
    for src_r, dst_r in active_transitions:
        base_acc = same_round_same_key_acc[src_r]

        m_same_key = _eval_model_on_target(models[src_r], cfg, dst_r, 'K1', device)
        records.append({
            'table': 'Table 2',
            'scenario': 'Different Round, Same Key',
            'train_round': src_r,
            'train_key': 'K1',
            'eval_round': dst_r,
            'eval_key': 'K1',
            'accuracy': round(m_same_key['accuracy'], 4),
            'tpr': round(m_same_key['tpr'], 4),
            'tnr': round(m_same_key['tnr'], 4),
            'loss': round(m_same_key['loss'], 4),
        })
        t2_rows.append([
            src_r, f"{base_acc:.4f}", dst_r,
            f"{m_same_key['accuracy']:.4f}", f"{m_same_key['tpr']:.4f}", f"{m_same_key['tnr']:.4f}",
        ])

        m_diff_key = _eval_model_on_target(models[src_r], cfg, dst_r, 'K2', device)
        records.append({
            'table': 'Table 3',
            'scenario': 'Different Round, Different Key',
            'train_round': src_r,
            'train_key': 'K1',
            'eval_round': dst_r,
            'eval_key': 'K2',
            'accuracy': round(m_diff_key['accuracy'], 4),
            'tpr': round(m_diff_key['tpr'], 4),
            'tnr': round(m_diff_key['tnr'], 4),
            'loss': round(m_diff_key['loss'], 4),
        })
        t3_rows.append([
            f"{src_r}-K1", f"{base_acc:.4f}", f"{dst_r}-K2",
            f"{m_diff_key['accuracy']:.4f}", f"{m_diff_key['tpr']:.4f}", f"{m_diff_key['tnr']:.4f}",
        ])

    annotate_significance(records, n_test=2 * cfg.dl_test_per_class)
    save_results(
        records,
        json_path=os.path.join(cfg.results_dir, f'exp_a_{cfg.results_tag}.json'),
        csv_path=os.path.join(cfg.results_dir, f'exp_a_{cfg.results_tag}.csv'),
    )

    if verbose:
        print_table(
            "Table 1 (DeepAnalyst): Same Round / Same Key vs. Same Round / Different Key",
            ["Rounds", "SameK Acc", "SameK TPR", "SameK TNR", "DiffK Acc", "DiffK TPR", "DiffK TNR"],
            t1_rows,
        )
        if t2_rows:
            print_table(
                "Table 2 (DeepAnalyst): Different Round, Same Key (K1)",
                ["Train R", "Train Acc", "Eval R", "Eval Acc", "TPR", "TNR"],
                t2_rows,
            )
        if t3_rows:
            print_table(
                "Table 3 (DeepAnalyst): Different Round, Different Key (K1 -> K2)",
                ["Train (R-K)", "Train Acc", "Eval (R-K)", "Eval Acc", "TPR", "TNR"],
                t3_rows,
            )

    return records
