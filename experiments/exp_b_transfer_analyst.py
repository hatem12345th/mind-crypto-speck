"""
Experiment B: TransferAnalyst (Transfer Learning + XGBoost Sample-Size Scaling).

Covers MIND-Crypt Section 6 Tables 4, 5, and 6 across sample sizes per class
(default: 5k, 10k, 100k, 200k, 290k) and rounds {5, 6, 21, 22}:
  - Table 4: Same Round, Different Key (K1 extractor -> K2 XGBoost)
  - Table 5: Different Rounds, Same Key (K1 extractor -> R+-1, K1 XGBoost)
  - Table 6: Different Rounds, Different Key (K1 extractor -> R+-1, K2 XGBoost)
"""

import os
import time
from typing import Any, Dict, List, Tuple
import torch

from models.resnet_distinguisher import load_checkpoint
from transfer.feature_extractor import extract_latent_features
from transfer.xgboost_eval import train_and_eval_xgboost
from .config import ExperimentConfig
from .dataset_builder import load_dataset_splits
from .reporting import print_table, save_results


def _evaluate_tl_pair(
    cfg: ExperimentConfig,
    src_round: int,
    dst_round: int,
    dst_key: str,
    table_name: str,
    scenario: str,
    device: torch.device,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract 512-dim latent features once using checkpoint (src_round, K1)
    on dataset (dst_round, dst_key), then evaluate XGBoost across all
    configured sample sizes per class.
    """
    ckpt_path = cfg.checkpoint_path(src_round, 'K1')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Missing DeepAnalyst checkpoint {ckpt_path}. Run Experiment A first."
        )

    extractor, _ = load_checkpoint(ckpt_path, device=device)
    max_train_per_class = max(cfg.tl_sample_sizes)

    splits = load_dataset_splits(
        cfg,
        rounds=dst_round,
        key_name=dst_key,
        train_per_class=max_train_per_class,
        val_per_class=0,
        test_per_class=cfg.tl_test_per_class,
    )

    # Extract features once for max train & test, then slice balanced prefixes [:2*n]
    t0 = time.time()
    x_train_all = extract_latent_features(
        extractor, splits['X_train'], batch_size=cfg.batch_size, device=device
    )
    y_train_all = splits['y_train']
    x_test_feats = extract_latent_features(
        extractor, splits['X_test'], batch_size=cfg.batch_size, device=device
    )
    y_test = splits['y_test']
    if verbose:
        print(
            f"  [TL Features] Extractor R={src_round}-K1 -> Target R={dst_round}-{dst_key} "
            f"(extracted {len(y_train_all):,} train + {len(y_test):,} test in {time.time() - t0:.2f}s)"
        )

    xgb_params = {
        'n_estimators': cfg.xgb_n_estimators,
        'max_depth': cfg.xgb_max_depth,
    }

    pair_records: List[Dict[str, Any]] = []
    for n_per_class in sorted(cfg.tl_sample_sizes):
        t_fit = time.time()
        n_rows = 2 * n_per_class
        x_tr_sub = x_train_all[:n_rows]
        y_tr_sub = y_train_all[:n_rows]

        res = train_and_eval_xgboost(
            x_train=x_tr_sub,
            y_train=y_tr_sub,
            x_test=x_test_feats,
            y_test=y_test,
            xgb_params=xgb_params,
            tune=cfg.tune_xgb,
            tune_n_iter=cfg.tune_n_iter,
            tune_cv=cfg.tune_cv,
            verbose=False,
        )
        m = res['metrics']
        elapsed = time.time() - t_fit
        if verbose:
            print(
                f"    n={n_per_class:>7,}/class ({n_rows:>7,} total) | "
                f"Acc={m['accuracy']:.4f} (TPR={m['tpr']:.4f}, TNR={m['tnr']:.4f}) [{elapsed:.2f}s]"
            )

        rec: Dict[str, Any] = {
            'table': table_name,
            'scenario': scenario,
            'extractor_round': src_round,
            'extractor_key': 'K1',
            'eval_round': dst_round,
            'eval_key': dst_key,
            'samples_per_class': n_per_class,
            'total_train_samples': n_rows,
            'accuracy': round(m['accuracy'], 4),
            'tpr': round(m['tpr'], 4),
            'tnr': round(m['tnr'], 4),
            'fit_seconds': round(elapsed, 2),
        }
        if 'cv_best_score' in res:
            rec['cv_best_score'] = round(res['cv_best_score'], 4)
        pair_records.append(rec)

    return pair_records


def run_experiment_b(
    cfg: ExperimentConfig,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Execute Experiment B across Tables 4, 5, and 6 and save CSV/JSON outputs.
    """
    all_records: List[Dict[str, Any]] = []

    active_transitions: List[Tuple[int, int]] = [
        (src_r, dst_r) for (src_r, dst_r) in cfg.transitions
        if src_r in cfg.rounds and dst_r in cfg.rounds
    ]

    # Table 4: Same Round, Different Key (R-K1 -> R-K2)
    if verbose:
        print("\n[TransferAnalyst] Running Table 4: Same Round, Different Key...")
    t4_records: List[Dict[str, Any]] = []
    for r in cfg.rounds:
        t4_records.extend(
            _evaluate_tl_pair(
                cfg, r, r, 'K2', 'Table 4', 'Same Round, Different Key', device, verbose=verbose
            )
        )
    all_records.extend(t4_records)

    # Table 5: Different Rounds, Same Key (src_r-K1 -> dst_r-K1)
    if active_transitions and verbose:
        print("\n[TransferAnalyst] Running Table 5: Different Rounds, Same Key...")
    t5_records: List[Dict[str, Any]] = []
    for src_r, dst_r in active_transitions:
        t5_records.extend(
            _evaluate_tl_pair(
                cfg, src_r, dst_r, 'K1', 'Table 5', 'Different Rounds, Same Key', device, verbose=verbose
            )
        )
    all_records.extend(t5_records)

    # Table 6: Different Rounds, Different Key (src_r-K1 -> dst_r-K2)
    if active_transitions and verbose:
        print("\n[TransferAnalyst] Running Table 6: Different Rounds, Different Key...")
    t6_records: List[Dict[str, Any]] = []
    for src_r, dst_r in active_transitions:
        t6_records.extend(
            _evaluate_tl_pair(
                cfg, src_r, dst_r, 'K2', 'Table 6', 'Different Rounds, Different Key', device, verbose=verbose
            )
        )
    all_records.extend(t6_records)

    save_results(
        all_records,
        json_path=os.path.join(cfg.results_dir, f'exp_b_{cfg.results_tag}.json'),
        csv_path=os.path.join(cfg.results_dir, f'exp_b_{cfg.results_tag}.csv'),
    )

    if verbose:
        if t4_records:
            t4_sorted = sorted(t4_records, key=lambda d: (d['samples_per_class'], d['eval_round']))
            print_table(
                "Table 4 (TransferAnalyst): Same Round, Different Key (K1 -> K2)",
                ["Samples/cls (k)", "Rounds", "Accuracy", "TPR", "TNR"],
                [
                    [d['samples_per_class'] // 1000, d['eval_round'], f"{d['accuracy']:.4f}", f"{d['tpr']:.4f}", f"{d['tnr']:.4f}"]
                    for d in t4_sorted
                ],
            )
        if t5_records:
            t5_sorted = sorted(t5_records, key=lambda d: (d['samples_per_class'], d['extractor_round']))
            print_table(
                "Table 5 (TransferAnalyst): Different Rounds, Same Key (K1)",
                ["Samples/cls (k)", "Extractor R", "Eval R", "Accuracy", "TPR", "TNR"],
                [
                    [d['samples_per_class'] // 1000, d['extractor_round'], d['eval_round'], f"{d['accuracy']:.4f}", f"{d['tpr']:.4f}", f"{d['tnr']:.4f}"]
                    for d in t5_sorted
                ],
            )
        if t6_records:
            t6_sorted = sorted(t6_records, key=lambda d: (d['samples_per_class'], d['extractor_round']))
            print_table(
                "Table 6 (TransferAnalyst): Different Rounds, Different Key (K1 -> K2)",
                ["Samples/cls (k)", "Extractor (R-K)", "Eval (R-K)", "Accuracy", "TPR", "TNR"],
                [
                    [
                        d['samples_per_class'] // 1000,
                        f"{d['extractor_round']}-K1",
                        f"{d['eval_round']}-K2",
                        f"{d['accuracy']:.4f}",
                        f"{d['tpr']:.4f}",
                        f"{d['tnr']:.4f}",
                    ]
                    for d in t6_sorted
                ],
            )

    return all_records
