# MIND-Crypt: Assessing Cryptographic Indistinguishability of SPECK32/64

A reproduction, full 22-round evaluation, and ablation framework for **MIND-Crypt** based on:
> *Breaking Indistinguishability with Transfer Learning: A First Look at SPECK32/64 Lightweight Block Ciphers* ([arXiv:2405.19683](https://arxiv.org/html/2405.19683v1))  
> Authors: Jimmy Dani, Kalyan Nakka, Nitesh Saxena (Texas A&M University)

---

## Environment Setup

### Option A: Existing Virtual Environment
```bash
source venv/bin/activate   # adjust the path if your venv lives elsewhere
```

### Option B: Fresh Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verified with Python 3.13; core versions are pinned in `requirements.txt` (`torch 2.15.0`, `xgboost 3.4.1`, `scikit-learn 1.9.1`, `scipy 1.18.1`, `numpy 2.5.4`, `pandas 3.0.6`, `matplotlib 3.11.2`, `seaborn 0.13.2`).

---

## Project Architecture

```
mind-crypto-speck/
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
│   └── reporting.py               # ASCII tables, JSON/CSV export, 95% CIs & p-values vs. chance
├── notebooks/
│   └── mind_crypt_analysis.ipynb  # Executed notebook with plots, ROC curves, ablations & avalanche heatmap
├── results/
│   ├── exp_a_results.{csv,json}   # Part 1 DeepAnalyst results (Rounds 5, 6, 21, 22)
│   ├── exp_b_results.{csv,json}   # Part 1 TransferAnalyst results (5k, 10k, 25k/class)
│   ├── exp_a_r1_22.{csv,json}     # DeepAnalyst CBC sweep across all rounds 1..22
│   ├── exp_b_r1_22.{csv,json}     # TransferAnalyst CBC sweep across all rounds 1..22
│   ├── exp_c_gohr_r1_22.{csv,json}# Gohr differential sweep across all rounds 1..22
│   ├── exp_{a,b}_iv_r1_22_s{1,2,3}.* # IV-visible CBC sweep (adversary sees IV || C), one file per seed
│   ├── exp_{a,b}_iv_r1_22_seeds.*   # IV-visible CBC: mean / std across seeds
│   ├── exp_c_gohr_r1_22_s{1,2,3}.*  # Gohr sweep, one file per seed
│   ├── exp_c_gohr_r1_22_seeds.*     # Gohr sweep: mean / std across 3 seeds
│   └── exp_c_gohr_depth5_300k.*     # Gohr capacity ablation (depth 5, 300k pairs, rounds 5-8)
├── tests/
│   ├── test_crypto.py             # Cryptographic and CBC dataset unit tests
│   ├── test_model.py              # PyTorch ResNet model tests
│   ├── test_transfer.py           # Transfer learning pipeline tests
│   ├── test_reporting.py          # Confidence interval / significance helper tests
│   ├── test_iv_mode.py            # IV || C dataset, config variants, seed summaries
│   └── test_experiments.py        # Cyclic LR, checkpointing, tuning, Gohr & pipeline tests
├── data/                          # Cached datasets (.npz, gitignored)
├── checkpoints/                   # Best validation ResNet checkpoints (.pt, gitignored)
├── requirements.txt               # Project dependencies
├── run_all.py                     # Unified CLI runner (Tables 1-6, 1..22 sweep & Gohr sweep)
└── generate_dataset_and_test.py   # Lightweight smoke-test script
```

---

## Running Tests, Experiments & Notebook

### 1. Run Complete Unit & Integration Test Suite (32 tests)
```bash
source venv/bin/activate
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

# Realistic CBC adversary (sees IV || C), all rounds, repeated over 3 seeds
python run_all.py --profile quick --all-rounds --tl-sizes 10000 --epochs 10 --include-iv --seeds 1 2 3

# Gohr sweep over 3 seeds, and a capacity ablation with a deeper ResNet
python run_all.py --profile quick --all-rounds --skip-exp-a --skip-exp-b --run-gohr --seeds 1 2 3
python run_all.py --profile quick --rounds 5 6 7 8 --skip-exp-a --skip-exp-b --run-gohr \
    --depth 5 --gohr-train-samples 300000 --gohr-epochs 10 --results-tag depth5_300k
```

Extra options: `--include-iv` (64-bit `IV || C` input), `--depth N` (residual blocks), and `--seeds S1 S2 ...` (repeat the run and write `*_seeds` summaries with mean / std accuracy). Dataset and checkpoint filenames include the input mode and seed, so different runs never reuse each other's cache.

### 3. Open Interactive Analysis Notebook
```bash
jupyter lab notebooks/mind_crypt_analysis.ipynb
```

---

## Key Empirical Findings (`results/` & Notebook)

| Setting | Round(s) | Test Accuracy | Interpretation |
|---|---|---|---|
| **Random-IV CBC, ciphertext only (paper setting)** | `1 → 22` | `0.4909 – 0.5072` | **50% by construction**, not a property of SPECK (see below). |
| **Random-IV CBC, adversary sees `IV ‖ C`** | `1` | `1.000 ± 0` | Fully broken after one round, even under an unseen key. |
| **Random-IV CBC, `IV ‖ C`, transfer from round 1** | `2` | `0.533 ± 0.013` (K1), `0.599 ± 0.039` (K2) | XGBoost on features from the **round-1** ResNet distinguishes round 2 on every seed (p < 10⁻⁷), while a ResNet trained on round 2 itself stays at 0.50. |
| **Random-IV CBC, adversary sees `IV ‖ C`** | `2 → 22` (all other settings) | `0.496 – 0.505` | Not distinguishable with this budget (25k samples/class, depth-1 ResNet, 3 seeds). |
| **Fixed-IV CBC / ECB (Sanity Check)** | `5` | `1.0000` | Only two distinct ciphertexts exist, so separation is trivial. |
| **Gohr Differential (`Δ=(0x0040, 0)`, random keys), 3 seeds** | `1 → 6` | `1.000 → 0.648 ± 0.009` | The same ResNet (`num_channels=4`) learns real differential trails across unseen random keys. |
| **Gohr Differential, deeper ResNet (depth 5, 300k pairs)** | `5, 6` | `0.884`, `0.720` | More capacity and data raise accuracy, approaching Gohr's 0.93 / 0.79. |
| **Gohr Differential, both budgets** | `7 → 22` | `0.494 – 0.503` | No bias **detected** (1 of 50 evaluations across seeds and budgets has p < 0.05, as chance predicts). This is a budget limit: Gohr (2019) reaches ≈0.62 at round 7 with a depth-10 ResNet trained on 10⁷ samples. |

### Why ciphertext-only CBC gives exactly 50%

In the paper setting the distinguisher only sees `C = E_K(P ⊕ IV)`. For a uniformly random IV, `P ⊕ IV` is uniform whatever `P` is, and `E_K` is a permutation, so `C` is uniform for **both** plaintexts. The two classes have identical distributions at every round count, even with zero rounds. No model can beat 50%, so Tables 1–6 in this setting cannot tell anything about SPECK's strength (`tests/test_iv_mode.py` checks this empirically). Training accuracy above 50% is memorisation.

In real CBC the IV is sent in the clear, so the realistic adversary sees `IV ‖ C`. `--include-iv` enables this setting.

### IV-visible CBC (`IV ‖ C`, mean ± std over seeds 1, 2, 3)

| Round | DeepAnalyst, same key (K1) | DeepAnalyst, different key (K2) | TransferAnalyst (XGBoost, K2) |
|---|---|---|---|
| 1 | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| 2 | 0.497 ± 0.001 | 0.500 ± 0.002 | 0.500 ± 0.006 |
| 3 | 0.499 ± 0.003 | 0.499 ± 0.001 | 0.499 ± 0.004 |
| 4 – 22 | 0.498 – 0.502 | 0.497 – 0.503 | 0.497 – 0.505 |

Round 1 breaks because the key cancels: after one round, `x ⊕ y = rol2(P_y ⊕ IV_y)`, so the plaintext bit can be read directly from `IV ‖ C` (`tests/test_iv_mode.py` decodes it exactly). Round 2 adds a key-dependent modular addition. A ResNet trained directly on round 2 does not learn it with this budget.

**Transfer learning breaks round 2.** XGBoost trained on the 512-dimensional features of the **round-1** ResNet reaches **0.533 ± 0.013** on round 2 under the same key (Table 5) and **0.599 ± 0.039** under a different key (Table 6). This holds on all three seeds, with p between 10⁻⁷ and 10⁻²⁶⁰. The round-1 network has learned the `IV`/`x ⊕ y` bit relations, and those features stay partly informative one round later. A round-2 extractor gives 0.50. This is the clearest evidence in the project for MIND-Crypt's TransferAnalyst idea: features learned on a weaker round can expose a stronger one. Gohr's pair-based distinguisher still works up to round 6, because a chosen input difference leaks far more than one ciphertext.

### Gohr differential distinguisher (mean ± std over seeds 1, 2, 3)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 – 22 |
|---|---|---|---|---|---|---|---|---|
| Depth 1, 50k pairs | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.001 | 0.953 ± 0.005 | 0.858 ± 0.006 | 0.648 ± 0.009 | 0.503 ± 0.004 | 0.494 – 0.503 |
| Depth 5, 300k pairs (seed 42) | – | – | – | – | 0.884 | 0.720 | 0.500 | 0.500 (R8) |
| Gohr (2019), depth 10, 10⁷ pairs | – | – | – | – | 0.929 | 0.788 | 0.616 | – |

### Statistical significance

Every result record (CSV/JSON in `results/`) includes `n_test`, a Wilson 95% confidence interval (`ci95_low`, `ci95_high`) and a two-sided binomial p-value against the 0.5 chance baseline (`p_vs_chance`). As a rule of thumb, chance alone moves accuracy by about ±0.007 at `n_test = 20,000` and about ±0.010 at `n_test = 10,000`.

* **Gohr rounds 1–6** are significant beyond doubt (p < 10⁻²⁰⁰), and the seed-to-seed spread is below 0.01.
* **Ciphertext-only CBC:** 7 of the 202 DeepAnalyst/TransferAnalyst evaluations have p < 0.05. That is no more than the ~5% expected from multiple testing alone, and none survives a Bonferroni correction, as the argument above predicts.
* **IV-visible CBC:** apart from round 1 and the round-1 → round-2 transfer above, 13 of the 574 per-seed evaluations have p < 0.05 (≈2%, below the 5% expected by chance).
* **Seeds:** the IV-visible CBC and Gohr sweeps were repeated over 3 seeds (`*_seeds` files). The original ciphertext-only tables are single-seed, since they are provably at chance.

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
