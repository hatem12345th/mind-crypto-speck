"""
Unified CLI runner for MIND-Crypt Experiments (arXiv:2405.19683).

Executes:
  1. SPECK32/64 NSA test vector sanity check
  2. CBC dataset generation across configured rounds and keys {K1, K2}
  3. Experiment A (DeepAnalyst): Tables 1, 2, and 3
  4. Experiment B (TransferAnalyst): Tables 4, 5, and 6
  5. Optional Experiment C: Gohr Differential Distinguisher sweep across rounds
"""

import argparse
import time
import torch

from crypto.speck import check_speck_testvector
from experiments.config import ExperimentConfig, PAPER_ROUNDS, PAPER_TL_SAMPLE_SIZES
from experiments.dataset_builder import ensure_all_datasets
from experiments.exp_a_deep_analyst import run_experiment_a
from experiments.exp_b_transfer_analyst import run_experiment_b
from experiments.exp_c_gohr_sweep import run_experiment_c_gohr


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create ExperimentConfig from CLI profile and optional overrides."""
    if args.profile == 'quick':
        cfg = ExperimentConfig(
            dl_train_per_class=25_000,
            dl_val_per_class=5_000,
            dl_test_per_class=10_000,
            tl_sample_sizes=[5_000, 10_000, 25_000],
            tl_test_per_class=10_000,
            epochs=5,
            cycle_epochs=5,
            xgb_n_estimators=50,
        )
    elif args.profile == 'standard':
        cfg = ExperimentConfig(
            dl_train_per_class=100_000,
            dl_val_per_class=20_000,
            dl_test_per_class=50_000,
            tl_sample_sizes=[5_000, 10_000, 100_000],
            tl_test_per_class=50_000,
            epochs=10,
            cycle_epochs=10,
            xgb_n_estimators=100,
        )
    else:  # 'paper' — full TransferAnalyst scaling up to 290k/class (580k total)
        cfg = ExperimentConfig(
            dl_train_per_class=290_000,
            dl_val_per_class=50_000,
            dl_test_per_class=100_000,
            tl_sample_sizes=list(PAPER_TL_SAMPLE_SIZES),
            tl_test_per_class=100_000,
            epochs=10,
            cycle_epochs=10,
            xgb_n_estimators=100,
        )

    if args.all_rounds:
        cfg.rounds = list(range(1, 23))
        cfg.transitions = [(r, r + 1) for r in range(1, 22)]
        cfg.results_tag = 'r1_22'
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.results_tag is not None:
        cfg.results_tag = args.results_tag
    if args.dl_train_per_class is not None:
        cfg.dl_train_per_class = args.dl_train_per_class
    if args.dl_test_per_class is not None:
        cfg.dl_test_per_class = args.dl_test_per_class
    if args.tl_sizes is not None:
        cfg.tl_sample_sizes = args.tl_sizes
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.tune:
        cfg.tune_xgb = True
    if args.tune_n_iter is not None:
        cfg.tune_n_iter = args.tune_n_iter

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIND-Crypt Experiments (Tables 1-6 & Round Sweeps)")
    parser.add_argument(
        '--profile',
        choices=['quick', 'standard', 'paper'],
        default='standard',
        help="Execution scale preset (default: standard)."
    )
    parser.add_argument(
        '--all-rounds',
        action='store_true',
        help="Evaluate every round from 1 to 22 (transitions r -> r+1) and tag outputs 'r1_22'."
    )
    parser.add_argument(
        '--rounds',
        type=int,
        nargs='+',
        default=None,
        help=f"SPECK32/64 rounds to evaluate (default: {PAPER_ROUNDS})."
    )
    parser.add_argument('--results-tag', type=str, default=None, help="Output filename suffix tag (default: results or r1_22).")
    parser.add_argument('--dl-train-per-class', type=int, default=None, help="Override DL training samples per class.")
    parser.add_argument('--dl-test-per-class', type=int, default=None, help="Override DL test samples per class.")
    parser.add_argument('--tl-sizes', type=int, nargs='+', default=None, help="Override TransferAnalyst sample sizes per class.")
    parser.add_argument('--epochs', type=int, default=None, help="Override ResNet training epochs.")
    parser.add_argument('--batch-size', type=int, default=None, help="Override PyTorch batch size.")
    parser.add_argument('--tune', action='store_true', help="Enable RandomizedSearchCV hyperparameter tuning (Appendix D).")
    parser.add_argument('--tune-n-iter', type=int, default=None, help="Number of RandomizedSearchCV iterations.")
    parser.add_argument('--run-gohr', action='store_true', help="Also run Experiment C (Gohr Differential sweep across rounds).")
    parser.add_argument('--gohr-train-samples', type=int, default=50_000, help="Training samples for Gohr sweep.")
    parser.add_argument('--gohr-epochs', type=int, default=8, help="Epochs per round for Gohr sweep.")
    parser.add_argument('--force-data', action='store_true', help="Regenerate cached .npz datasets.")
    parser.add_argument('--force-retrain', action='store_true', help="Retrain ResNet models even if checkpoints exist.")
    parser.add_argument('--skip-exp-a', action='store_true', help="Skip Experiment A (requires existing checkpoints).")
    parser.add_argument('--skip-exp-b', action='store_true', help="Skip Experiment B.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_start = time.time()
    print("=" * 72)
    print(f"MIND-Crypt Runner | profile={args.profile} | tag={cfg.results_tag} | device={device}")
    print(f"Rounds: {cfg.rounds} | TL sample sizes/class: {cfg.tl_sample_sizes} | Tune XGB: {cfg.tune_xgb}")
    print("=" * 72)

    assert check_speck_testvector(), "SPECK32/64 NSA test vector verification failed!"
    print("[✓] SPECK32/64 NSA test vector verified.")

    if not (args.skip_exp_a and args.skip_exp_b):
        ensure_all_datasets(cfg, force_regenerate=args.force_data, verbose=True)

    if not args.skip_exp_a:
        run_experiment_a(cfg, device=device, force_retrain=args.force_retrain, verbose=True)

    if not args.skip_exp_b:
        run_experiment_b(cfg, device=device, verbose=True)

    if args.run_gohr:
        run_experiment_c_gohr(
            cfg,
            n_train=args.gohr_train_samples,
            n_val=max(args.gohr_train_samples // 5, 2_000),
            n_test=max(args.gohr_train_samples // 5, 2_000),
            epochs=args.gohr_epochs,
            device=device,
            verbose=True,
        )

    print(f"\nCompleted MIND-Crypt run in {time.time() - t_start:.2f}s.")
    print(f"Saved results in '{cfg.results_dir}/' and checkpoints in '{cfg.checkpoint_dir}/'.")


if __name__ == '__main__':
    main()
