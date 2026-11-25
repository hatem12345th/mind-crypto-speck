# MIND-Crypt: Assessing Cryptographic Indistinguishability of SPECK32/64

A reproduction and evaluation framework for **MIND-Crypt** based on:
> *Breaking Indistinguishability with Transfer Learning: A First Look at SPECK32/64 Lightweight Block Ciphers* ([arXiv:2405.19683](https://arxiv.org/html/2405.19683v1))  
> Authors: Jimmy Dani, Kalyan Nakka, Nitesh Saxena (Texas A&M University)

---

## Environment Setup

Activate the pre-configured virtual environment:
```bash
source ../venv/bin/activate
```

Installed libraries used:
* **PyTorch** 2.15.0
* **XGBoost** 3.4.1
* **Scikit-Learn** 1.9.1 & **SciPy** 1.18.1
* **NumPy** 2.5.4

---

## Project Architecture

```
speck-sight/
├── crypto/
│   ├── speck.py                   # SPECK32/64 implementation (verified against NSA test vectors)
│   └── cbc_generator.py           # Vectorized CBC mode dataset generator
├── models/
│   ├── resnet_distinguisher.py    # PyTorch 1D-ResNet, Gohr cyclic LR, and checkpointing
│   └── dataset.py                 # PyTorch Dataset and DataLoader wrappers
├── transfer/
│   ├── feature_extractor.py       # 512-dim latent feature extraction (Flatten layer)
│   └── xgboost_eval.py            # XGBoost classifier + Appendix D RandomizedSearchCV tuning
├── experiments/
│   ├── config.py                  # Rounds {5,6,21,22}, keys {K1,K2}, sample sizes, hyperparameters
│   ├── dataset_builder.py         # Dataset generation, caching, and balanced split loading
│   ├── exp_a_deep_analyst.py      # Experiment A (Tables 1, 2, 3)
│   ├── exp_b_transfer_analyst.py  # Experiment B (Tables 4, 5, 6)
│   └── reporting.py               # ASCII table printing and JSON/CSV export
├── tests/
│   ├── test_crypto.py             # Cryptographic and dataset unit tests
│   ├── test_model.py              # PyTorch ResNet model tests
│   ├── test_transfer.py           # Transfer learning pipeline tests
│   └── test_experiments.py        # Cyclic LR, checkpointing, tuning, and pipeline integration tests
├── data/                          # Cached datasets (.npz)
├── checkpoints/                   # Best validation ResNet checkpoints (.pt)
├── results/                       # Experiment A & B metrics (.csv and .json)
├── notebooks/
│   └── mind_crypt_analysis.ipynb  # Interactive visualizations, sanity ablations & avalanche plots
├── run_all.py                     # Unified CLI runner for Part 1 (Tables 1-6)
└── generate_dataset_and_test.py   # Lightweight smoke-test script
```

---

## Running Tests & Experiments

### 1. Run Complete Unit & Integration Test Suite
```bash
source ../venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
```

### 2. Run Part 1 Experiments (Tables 1–6 across Rounds 5, 6, 21, 22)
Presets available via `--profile`:
* `--profile quick`: Fast verification across all 4 rounds (`5, 6, 21, 22`) with smaller sample sizes.
* `--profile standard`: Default benchmark across all 4 rounds (`5, 6, 21, 22`), 100k DL samples/class, 10 epochs (Gohr cyclic LR), and TL sample-size scaling (`5k, 10k, 100k` per class).
* `--profile paper`: Full TransferAnalyst sample-size sweep up to **290,000 samples/class** (`5k, 10k, 100k, 200k, 290k` = up to 580,000 samples).

```bash
# Standard run across Rounds 5, 6, 21, 22
python run_all.py --profile standard

# Full paper TransferAnalyst scaling (5k..290k per class) + Appendix D XGBoost tuning
python run_all.py --profile paper --tune
```

---

## Evaluation Matrix Covered (Tables 1–6)

| Table | Framework | Scenario | Rounds / Keys Evaluated |
|---|---|---|---|
| **Table 1** | DeepAnalyst (ResNet) | Same Round, Same Key & Diff Key | $R \in \{5, 6, 21, 22\},\ K_1 \to K_1, K_2$ |
| **Table 2** | DeepAnalyst (ResNet) | Different Round, Same Key | $5 \leftrightarrow 6,\ 21 \leftrightarrow 22$ under $K_1$ |
| **Table 3** | DeepAnalyst (ResNet) | Different Round, Different Key | $(5,K_1) \to (6,K_2),\ (6,K_1) \to (5,K_2),\ (21,K_1) \to (22,K_2),\ (22,K_1) \to (21,K_2)$ |
| **Table 4** | TransferAnalyst (XGBoost) | Same Round, Different Key | Sample-size scaling across $R \in \{5, 6, 21, 22\}$ |
| **Table 5** | TransferAnalyst (XGBoost) | Different Rounds, Same Key | Sample-size scaling across $5 \leftrightarrow 6,\ 21 \leftrightarrow 22$ |
| **Table 6** | TransferAnalyst (XGBoost) | Different Rounds, Different Key | Sample-size scaling across $(R,K_1) \to (R \pm 1,K_2)$ |
