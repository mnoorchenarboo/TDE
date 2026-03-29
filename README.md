# TDE — Temporal Deep Explainer

A neural network-based post-hoc explainability method for time series forecasting models. TDE learns feature attribution maps φ(x) that indicate how much each feature at each time step contributed to a model's prediction — similar to SHAP values but trained as a reusable neural network for fast inference.

---

## Overview

TDE trains a dedicated explainer network that takes a time series input and outputs attributions of the same shape (Batch × Time × Features). The explainer uses:

- **Temporal Convolution Block** — Conv1D → GELU → LayerNorm → Dropout
- **Multi-Head Attention** — applied over convolution outputs
- **Residual concatenation** of conv and attention outputs
- **Feature Attribution Mapping** — Conv1D with learned weights
- **Softshrink sparsity activation** — automatically zeros small/irrelevant attributions

**Training objective:**

```
L_total = L_coalition + λ₁‖φ‖₁ + λ₂‖φ‖₂² + λ_s Σ‖φ_t − φ_{t-1}‖₂²
```

Where:
- `L_coalition`: Coalition fidelity loss (mask-weighted attributions match model prediction differences)
- `λ₁‖φ‖₁`: L1 regularization (sparsity)
- `λ₂‖φ‖₂²`: L2 regularization
- `λ_s Σ‖φ_t − φ_{t-1}‖₂²`: Temporal smoothness across time steps

---

## Supported Models

Ten deep learning architectures for time series forecasting:

| Type | Models |
|------|--------|
| RNN-based | LSTM, GRU, Bidirectional LSTM (BLSTM), Bidirectional GRU (BGRU) |
| CNN-based | CNN1D, DCNN (Dilated CNN), TCN (Temporal CNN), WaveNet |
| Transformer-based | TFT (Temporal Fusion Transformer), TST (Transformer for Series) |

---

## Supported Explainability Methods

TDE is compared against eight baseline XAI methods via `xai.py`:

1. **TDE** (this work)
2. **FastSHAP**
3. Gradient SHAP
4. Deep SHAP
5. Kernel SHAP
6. Permutation SHAP
7. Partition SHAP
8. LIME
9. Sampling SHAP

Evaluation metrics: **fidelity**, **reliability** (correlation + MSE), **sparsity**, **complexity**, **efficiency error**, **computation time**.

---

## Datasets

Energy consumption forecasting data from two sources:

- **BDG2** — 17 commercial building types: office, retail, warehouse, health, education, lodging, parking, assembly, food, utility, religion, science, industrial, services, public, other, unknown
- **London Hydro** — 19 residential household time series

Features include energy consumption, weather (temperature, humidity, wind speed, pressure), and temporal encodings (hour, day of week, month, weekend flag).

---

## Project Structure

```
TDE/
├── tde.py                  # Main CLI: trains and compares TDE/FastSHAP explainers
├── tde_ablation.py         # Ablation study: architectural variants
├── dl.py                   # Deep learning models + data loading
├── xai.py                  # Compares 9 XAI methods with evaluation metrics
├── Results.py              # Generates heatmap visualizations from results DB
├── ablation_stats.py       # Statistical analysis of ablation study results
├── Functions/
│   ├── tde_class.py        # TemporalDeepExplainer and FastSHAPExplainer classes
│   ├── preprocess.py       # Data loading and preprocessing from SQLite
│   └── utils.py            # Utility functions
└── databases/
    ├── energy_data.db      # Source energy datasets (pre-supplied)
    ├── benchmark_results.db        # Model training results (auto-created)
    ├── explainer_results.db        # TDE/FastSHAP metadata and hyperparameters (auto-created)
    ├── xai_results.db              # XAI comparison results (auto-created)
    └── ablation_results.db         # Ablation study results (auto-created)
```

---

## Installation

```bash
pip install torch numpy pandas scikit-learn optuna shap lime matplotlib seaborn tqdm
```

No `requirements.txt` is included; install the above dependencies via pip or conda.

---

## Usage

### Train and evaluate explainers

```bash
python tde.py
```

The script runs interactively and prompts for:

1. **Primary use** — building type / domain (e.g., `office`, `residential`, or `all`)
2. **Dataset option** — building/site index number
3. **Models** — one or more from the 10 supported architectures, or `all`
4. **Explainer type** — `0` TDE, `1` FastSHAP, `2` Both
5. **Training fraction** — proportion of training data to use (0.05–1.0)
6. **Optimization trials** — number of Optuna hyperparameter search trials (default: 30)
7. **Test samples** — number of test samples to evaluate (default: 5)

The system auto-detects existing results and offers to skip, replace, or retrain.

### Compare XAI methods

```bash
python xai.py
```

Runs all 9 explainability methods on the trained models and saves comparison metrics to `xai_results.db`.

### Run ablation study

```bash
python tde_ablation.py
```

Tests three architectural variants against the full TDE baseline:

| Variant | Description |
|---------|-------------|
| `baseline` | Full TDE (loaded from disk) |
| `arch_no_attn` | No multi-head attention block |
| `arch_no_conv` | No temporal convolution block |
| `mask_uniform` | Uniform random masking instead of Shapley kernel sampling |

### Generate visualizations

```bash
python Results.py
```

Produces SHAP-style heatmaps and comparison plots from the results databases.

---

## Hyperparameter Tuning

Hyperparameter optimization uses **Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler.

**Search space includes:**

| Parameter | Description |
|-----------|-------------|
| `l1_lambda` | L1 regularization weight (log-scale) |
| `l2_lambda` | L2 regularization weight (log-scale) |
| `smoothness_lambda` | Temporal smoothness weight |
| `sparsity_threshold` | Softshrink activation threshold |
| `hidden_dim` | Hidden channel size |
| `kernel_size` | Conv1D kernel width |
| `n_attention_heads` | Number of attention heads |
| `dropout_rate` | Dropout probability |
| `batch_size` | Training batch size |
| `learning_rate` | Optimizer learning rate |
| `optimizer_type` | `adam` or `adamw` |
| `samples_per_feature` | Coalition mask samples per training example |

**Tuning settings:**

| Mode | Trials | Epochs/trial | Final epochs |
|------|--------|-------------|--------------|
| Production | 30 | 20 | 100 |
| Debug | 5 | 10 | 50 |

Best hyperparameters are saved to `explainer_results.db` and as `hyperparameters.json` under each model's results folder.

---

## Results Structure

```
results/{primary_use}/option_{N}/{model}/
├── tde/
│   ├── tde_explainer.pt               # Trained TDE model weights
│   ├── optuna_study.db                # Optuna study (all trials)
│   ├── hyperparameters.json           # Best hyperparameters
│   ├── hyperparameter_importance.png  # Optuna importance plot
│   └── convergence.png                # Training loss curve
├── fastshap/
│   └── [same structure]
└── training_TIMESTAMP.log             # Training log
```

All structured results are also stored in SQLite databases under `databases/`.

---

## Citation

If you use this code, please cite the associated paper (TBD).
