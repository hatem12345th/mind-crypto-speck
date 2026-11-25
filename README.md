# MIND-Crypt: Assessing Cryptographic Indistinguishability of SPECK32/64

A reproduction, full 22-round evaluation, and ablation framework for **MIND-Crypt** based on:
> *Breaking Indistinguishability with Transfer Learning: A First Look at SPECK32/64 Lightweight Block Ciphers* ([arXiv:2405.19683](https://arxiv.org/html/2405.19683v1))  
> Authors: Jimmy Dani, Kalyan Nakka, Nitesh Saxena (Texas A&M University)

---

## Environment Setup

### Option A: Pre-configured Environment
```bash
source ../venv/bin/activate
```

### Option B: Fresh Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verified with Python 3.13 (`torch 2.15.0`, `xgboost 3.4.1`, `scikit-learn 1.9.1`, `scipy 1.18.1`, `numpy 2.5.4`, `pandas 3.0.6`, `matplotlib 3.11.2`, `seaborn 0.13.2`).

---

## Project Architecture

```
speck-sight/
├── crypto/
│   ├── speck.py                   # SPECK32/64 implementation (verified against NSA test vectors)
│   ├── cbc_generator.py           # Vectorized CBC mode dataset generator (random/fixed/counter IV)
│   └── gohr_generator.py          # Gohr (2019) differential ciphertext-pair dataset generator
├── models/
│   ├── resnet_distinguisher.py    # PyTorch 1D-ResNet, Gohr cyclic LR, and checkpointing
│   └── dataset.py                 # PyTorch Dataset and DataLoader wrappers
├── transfer/
│   ├── feature_extractor.py       # 512-dim latent feature extraction (Flatten layer)
│   └── xgboost_eval.py            # XGBoost classifier + Appendix D RandomizedSearchCV tuning
├── experiments/
│   ├── config.py                  # Rounds, keys {K1,K2}, sample sizes, hyperparameters
│   ├── dataset_builder.py         # Dataset generation, caching, and balanced split loading
│   ├── exp_a_deep_analyst.py      # Experiment A: DeepAnalyst (Tables 1, 2, 3)
│   ├── exp_b_transfer_analyst.py  # Experiment B: TransferAnalyst (Tables 4, 5, 6)
│   ├── exp_c_gohr_sweep.py        # Experiment C: Gohr differential distinguisher round sweep
│   └── reporting.py               # ASCII table printing and JSON/CSV export
├── notebooks/
│   └── mind_crypt_analysis.ipynb  # Executed notebook with plots, ROC curves, ablations & avalanche heatmap
├── results/
│   ├── exp_a_results.{csv,json}   # Part 1 DeepAnalyst results (Rounds 5, 6, 21, 22)
│   ├── exp_b_results.{csv,json}   # Part 1 TransferAnalyst results (5k, 10k, 25k/class)
│   ├── exp_a_r1_22.{csv,json}     # DeepAnalyst CBC sweep across all rounds 1..22
│   ├── exp_b_r1_22.{csv,json}     # TransferAnalyst CBC sweep across all rounds 1..22
│   └── exp_c_gohr_r1_22.{csv,json}# Gohr differential sweep across all rounds 1..22
├── tests/
│   ├── test_crypto.py             # Cryptographic and CBC dataset unit tests
│   ├── test_model.py              # PyTorch ResNet model tests
│   ├── test_transfer.py           # Transfer learning pipeline tests
│   └── test_experiments.py        # Cyclic LR, checkpointing, tuning, Gohr & pipeline tests
├── data/                          # Cached datasets (.npz, gitignored)
├── checkpoints/                   # Best validation ResNet checkpoints (.pt, gitignored)
├── requirements.txt               # Project dependencies
├── run_all.py                     # Unified CLI runner (Tables 1-6, 1..22 sweep & Gohr sweep)
└── generate_dataset_and_test.py   # Lightweight smoke-test script
```

---

## Running Tests, Experiments & Notebook

### 1. Run Complete Unit & Integration Test Suite (19 tests)
```bash
source ../venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py' -v
```

### 2. Run Experiments (`run_all.py`)
Presets available via `--profile`:
* `--profile quick`: Fast verification (`25k` DL train/class, `5` epochs, `[5k, 10k, 25k]` TL sweep).
* `--profile standard`: Default benchmark (`100k` DL train/class, `10` epochs, `[5k, 10k, 100k]` TL sweep).
* `--profile paper`: Full TransferAnalyst sample-size sweep up to **290,000 samples/class** (`5k, 10k, 100k, 200k, 290k`).

```bash
# Part 1 evaluation matrix across Rounds 5, 6, 21, 22
python run_all.py --profile standard

# Full paper TransferAnalyst scaling (5k..290k/class) + Appendix D XGBoost tuning
python run_all.py --profile paper --tune

# Sweep ALL rounds from 1 to 22 (CBC DeepAnalyst + TransferAnalyst + Gohr Differential)
python run_all.py --profile quick --all-rounds --tl-sizes 10000 --run-gohr --gohr-train-samples 50000 --gohr-epochs 8
```

### 3. Open Interactive Analysis Notebook
```bash
jupyter lab notebooks/mind_crypt_analysis.ipynb
```

---

## Key Empirical Findings (`results/` & Notebook)

| Setting | Round(s) | Test Accuracy | Interpretation |
|---|---|---|---|
| **Random-IV CBC (DeepAnalyst & TransferAnalyst)** | `1 → 22` | `0.4909 – 0.5072` | Uniform random 32-bit IVs blind single-block plaintexts; ciphertexts remain indistinguishable from random guessing on unseen splits (while training accuracy climbs via memorization). |
| **Fixed-IV CBC / ECB (Sanity Check)** | `5` | `1.0000` | Without random IVs, deterministic ciphertexts separate trivially. |
| **Gohr Differential (`Δ=(0x0040, 0)`, random keys)** | `1 → 6` | `1.0000 → 0.6617` | The same `ResNetDistinguisher` (`num_channels=4`) learns real ARX differential trails across unseen random keys (`R=1: 1.000`, `R=4: 0.949`, `R=5: 0.859`, `R=6: 0.662`). |
| **Gohr Differential (`Δ=(0x0040, 0)`, random keys)** | `7 → 22` | `0.4909 – 0.5094` | Differential bias vanishes to the chance baseline from Round 7 onward (at 50k training samples), matching the ~round-6 saturation observed in the 22-round avalanche sweep. |

---

## Evaluation Matrix Covered (Tables 1–6)

| Table | Framework | Scenario | Rounds / Keys Evaluated |
|---|---|---|---|
| **Table 1** | DeepAnalyst (ResNet) | Same Round, Same Key & Diff Key | $R \in \{5, 6, 21, 22\}$ (and `1..22` with `--all-rounds`), $K_1 \to K_1, K_2$ |
| **Table 2** | DeepAnalyst (ResNet) | Different Round, Same Key | $5 \leftrightarrow 6,\ 21 \leftrightarrow 22$ (or $r \to r+1$) under $K_1$ |
| **Table 3** | DeepAnalyst (ResNet) | Different Round, Different Key | $(R,K_1) \to (R \pm 1,K_2)$ |
| **Table 4** | TransferAnalyst (XGBoost) | Same Round, Different Key | Sample-size scaling across $K_1 \to K_2$ |
| **Table 5** | TransferAnalyst (XGBoost) | Different Rounds, Same Key | Sample-size scaling across $R \to R \pm 1$ under $K_1$ |
| **Table 6** | TransferAnalyst (XGBoost) | Different Rounds, Different Key | Sample-size scaling across $(R,K_1) \to (R \pm 1,K_2)$ |
