"""
End-to-End Dataset Generation and Verification Script for MIND-Crypt.

Generates datasets for SPECK32/64 in CBC mode across rounds (5, 6)
and keys (K1, K2), saves them to disk, and runs an end-to-end validation.
"""

import os
import time
import numpy as np
import torch
from crypto.speck import PAPER_KEY_1_BYTES, PAPER_KEY_2_BYTES, check_speck_testvector
from crypto.cbc_generator import CBCDatasetGenerator
from models.resnet_distinguisher import ResNetDistinguisher, train_model, evaluate_model
from models.dataset import create_dataloaders
from transfer.feature_extractor import extract_latent_features
from transfer.xgboost_eval import train_and_eval_xgboost


def generate_all_datasets(data_dir: str = 'data', samples_per_class: int = 25000):
    """
    Generate datasets for evaluation scenarios:
    - R5_K1 (Training / Base)
    - R6_K1 (Different Round, Same Key)
    - R5_K2 (Same Round, Different Key)
    - R6_K2 (Different Round, Different Key)
    """
    os.makedirs(data_dir, exist_ok=True)
    configs = [
        ('speck32_r5_k1.npz', 5, PAPER_KEY_1_BYTES, "Round 5, Key K1"),
        ('speck32_r6_k1.npz', 6, PAPER_KEY_1_BYTES, "Round 6, Key K1"),
        ('speck32_r5_k2.npz', 5, PAPER_KEY_2_BYTES, "Round 5, Key K2"),
        ('speck32_r6_k2.npz', 6, PAPER_KEY_2_BYTES, "Round 6, Key K2"),
    ]

    print("=" * 70)
    print("STEP 1: Generating CBC Datasets (SPECK32/64)")
    print(f"Samples per dataset: {2 * samples_per_class:,} (50% Class 0 / 50% Class 1)")
    print("=" * 70)

    datasets = {}
    for filename, rounds, key, desc in configs:
        t0 = time.time()
        gen = CBCDatasetGenerator(key=key, rounds=rounds, seed=42)
        # 70% train, 15% val, 15% test
        n_train = int(samples_per_class * 0.70)
        n_val = int(samples_per_class * 0.15)
        n_test = samples_per_class - n_train - n_val

        splits = gen.generate_split_dataset(
            train_samples_per_class=n_train,
            test_samples_per_class=n_test,
            val_samples_per_class=n_val,
            iv_mode='random'
        )
        filepath = os.path.join(data_dir, filename)
        gen.save_dataset_to_disk(splits, filepath)
        t1 = time.time()

        print(f"  [✓] {desc:<30} -> {filepath:<25} ({t1 - t0:.2f}s)")
        datasets[filename.replace('.npz', '')] = splits

    return datasets


def run_pipeline_test(datasets: dict):
    """Run end-to-end model training, evaluation, and transfer learning test."""
    print("\n" + "=" * 70)
    print("STEP 2: End-to-End Pipeline Verification")
    print("=" * 70)

    # 1. Verify cryptographic primitives
    print("[1/5] Verifying SPECK32/64 NSA test vector...")
    assert check_speck_testvector(), "SPECK test vector verification failed!"
    print("      -> NSA test vector: PASSED.")

    # 2. Setup DataLoaders for R5_K1
    print("\n[2/5] Preparing PyTorch DataLoaders (R5_K1)...")
    r5_k1 = datasets['speck32_r5_k1']
    loaders = create_dataloaders(r5_k1, batch_size=2048)
    print(f"      Train samples: {len(r5_k1['y_train']):,}")
    print(f"      Val samples:   {len(r5_k1['y_val']):,}")
    print(f"      Test samples:  {len(r5_k1['y_test']):,}")

    # 3. Train DeepAnalyst ResNet
    print("\n[3/5] Training DeepAnalyst 1D-ResNet on R5_K1 (5 epochs)...")
    model = ResNetDistinguisher(num_filters=32, depth=1, d1=64, d2=64)
    train_model(
        model,
        loaders['train'],
        loaders['val'],
        epochs=5,
        lr=0.003,
        verbose=True
    )

    # 4. Evaluate across Scenarios (DeepAnalyst)
    print("\n[4/5] Evaluating DeepAnalyst across cryptographic scenarios:")
    scenarios = [
        ("Same Round, Same Key (R5_K1)", r5_k1['X_test'], r5_k1['y_test']),
        ("Different Round, Same Key (R6_K1)", datasets['speck32_r6_k1']['X_test'], datasets['speck32_r6_k1']['y_test']),
        ("Same Round, Different Key (R5_K2)", datasets['speck32_r5_k2']['X_test'], datasets['speck32_r5_k2']['y_test']),
        ("Different Round, Different Key (R6_K2)", datasets['speck32_r6_k2']['X_test'], datasets['speck32_r6_k2']['y_test']),
    ]

    dl_results = {}
    for name, x_eval, y_eval in scenarios:
        eval_loader = create_dataloaders({'X_test': x_eval, 'y_test': y_eval}, batch_size=2048)['test']
        metrics = evaluate_model(model, eval_loader)
        dl_results[name] = metrics
        print(f"      * {name:<40}: Accuracy = {metrics['accuracy']:.4f} (TPR: {metrics['tpr']:.4f}, TNR: {metrics['tnr']:.4f})")

    # 5. Transfer Learning with TransferAnalyst (XGBoost)
    print("\n[5/5] Running TransferAnalyst (Feature Extraction + XGBoost):")
    # Feature extraction from pre-trained ResNet on Different Key dataset (R5_K2)
    r5_k2 = datasets['speck32_r5_k2']
    print("      Extracting 512-dim latent representations from pre-trained ResNet...")
    x_train_feats = extract_latent_features(model, r5_k2['X_train'][:5000])
    y_train_sub = r5_k2['y_train'][:5000]
    x_test_feats = extract_latent_features(model, r5_k2['X_test'][:2000])
    y_test_sub = r5_k2['y_test'][:2000]

    print(f"      Training XGBoost on {len(y_train_sub):,} extracted feature vectors...")
    tl_res = train_and_eval_xgboost(
        x_train_feats, y_train_sub,
        x_test_feats, y_test_sub,
        xgb_params={'n_estimators': 50, 'max_depth': 4},
        verbose=False
    )
    tl_acc = tl_res['metrics']['accuracy']
    tl_tpr = tl_res['metrics']['tpr']
    tl_tnr = tl_res['metrics']['tnr']
    print(f"      -> TransferAnalyst on R5_K2: Accuracy = {tl_acc:.4f} (TPR: {tl_tpr:.4f}, TNR: {tl_tnr:.4f})")

    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"{'Scenario':<42} | {'DL Accuracy':<12} | {'DL TPR':<8} | {'DL TNR':<8}")
    print("-" * 76)
    for name, m in dl_results.items():
        print(f"{name:<42} | {m['accuracy']:<12.4f} | {m['tpr']:<8.4f} | {m['tnr']:<8.4f}")
    print("-" * 76)
    print(f"TransferAnalyst (TL on Diff Key R5_K2)    | {tl_acc:<12.4f} | {tl_tpr:<8.4f} | {tl_tnr:<8.4f}")
    print("=" * 70)
    print("All datasets generated and test pipeline completed successfully!")


if __name__ == '__main__':
    datasets = generate_all_datasets(data_dir='data', samples_per_class=20000)
    run_pipeline_test(datasets)
