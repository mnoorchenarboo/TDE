# ============================
# IMPORTS
# ============================
import os
import sys
import json
import time
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from dl import load_complete_model
from Functions.tde_class import TemporalDeepExplainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================
# PATHS
# ============================
PATH_DBS         = Path("databases")
PATH_DBS.mkdir(parents=True, exist_ok=True)

ABLATION_DB      = PATH_DBS / "ablation_results.db"
BENCHMARK_DB     = PATH_DBS / "benchmark_results.db"
ENERGY_DB        = PATH_DBS / "energy_data.db"
RESULTS_BASE_DIR = "results"

# ============================
# ABLATION BASE CONFIG
# ============================
BASE_CONFIG = {
    'use_conv':            True,
    'use_attention':       True,
    'use_layernorm':       True,
    'activation':          'gelu',
    'hidden_dim':          128,
    'kernel_size':         3,
    'n_attention_heads':   4,
    'dropout_rate':        0.2,
    'sparsity_threshold':  0.01,
    'masking_strategy':    'shapley',
    'paired_sampling':     True,
    'samples_per_feature': 2,
    'use_l1':              True,
    'use_l2':              True,
    'use_smoothness':      True,
    'l1_lambda':           0.01,
    'l2_lambda':           0.01,
    'smoothness_lambda':   0.1,
    'n_epochs':            50,
    'batch_size':          256,
    'learning_rate':       1e-3,
    'optimizer_type':      'adam',
    'weight_decay':        1e-4,
    'patience':            5,
    'min_lr':              1e-6,
}

# ============================
# ABLATION VARIANT DEFINITIONS
# Baseline is loaded from disk (saved by tde.py), never retrained.
# ============================
ABLATION_VARIANTS = {
    # Baseline — loaded from explainer_results.db / disk, never retrained
    'baseline':      {'group': 'baseline',     'desc': 'Full TDE — loaded from disk'},
    # Architecture ablations
    'arch_no_attn':  {'group': 'architecture', 'desc': 'No attention — conv block only',  'use_attention': False},
    'arch_no_conv':  {'group': 'architecture', 'desc': 'No conv — attention only',        'use_conv': False},
    # Masking ablations
    'mask_uniform':  {'group': 'masking',      'desc': 'Uniform sampling — no Shapley kernel', 'masking_strategy': 'uniform'},
}

# Only optimisation hyperparameters are loaded from the DB.
# Structural keys (use_conv, masking_strategy, …) always come from the variant definition.
_TUNABLE_KEYS = {
    'l1_lambda', 'l2_lambda', 'smoothness_lambda', 'sparsity_threshold',
    'hidden_dim', 'kernel_size', 'n_attention_heads', 'dropout_rate',
    'batch_size', 'learning_rate', 'optimizer_type', 'samples_per_feature',
}

# ============================
# HYPERPARAMETER LOADING
# ============================
def load_best_tde_params(primary_use, option_number, model_name):
    """Load tuned hyperparameters from explainer_results.db.

    Returns a dict of {param: value} for keys in _TUNABLE_KEYS only.
    Falls back to {} (BASE_CONFIG defaults) if not found.
    """
    try:
        explainer_db = PATH_DBS / "explainer_results.db"
        if not explainer_db.exists():
            print(f"   ⚠️  explainer_results.db not found — using BASE_CONFIG defaults")
            return {}
        conn   = sqlite3.connect(explainer_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT best_hyperparameters FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type='TDE'",
            (primary_use, option_number, model_name))
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            params = {k: v for k, v in json.loads(row[0]).items() if k in _TUNABLE_KEYS}
            if params:
                print(f"   ✅ Loaded {len(params)} tuned hyperparameters for {model_name}")
                return params
    except Exception as e:
        print(f"   ⚠️  Could not load tuned params: {e}")
    print(f"   ⚠️  No tuned TDE params found for {model_name} — using BASE_CONFIG defaults")
    return {}


def resolve_config(variant_key, tuned_base=None):
    """Build the full config dict for a variant.

    Priority: BASE_CONFIG → tuned_base (DB values) → variant structural overrides.
    """
    variant = ABLATION_VARIANTS[variant_key]
    cfg = {**BASE_CONFIG}
    if tuned_base:
        cfg.update(tuned_base)
    for k, v in variant.items():
        if k not in ('group', 'desc'):
            cfg[k] = v
    cfg['variant_key']   = variant_key
    cfg['variant_group'] = variant['group']
    cfg['variant_desc']  = variant['desc']
    return cfg


# ============================
# ABLATION NETWORK
# ============================
class AblationTDENetwork(nn.Module):
    """Configurable TDE network for ablation variants (not used for baseline)."""

    def __init__(self, time_steps, n_features, cfg):
        super().__init__()
        self.time_steps          = time_steps
        self.n_features          = n_features
        self.cfg                 = cfg
        hidden_dim               = cfg['hidden_dim']
        self.sparsity_threshold  = cfg['sparsity_threshold']

        if cfg['use_conv']:
            padding         = (cfg['kernel_size'] - 1) // 2
            self.conv       = nn.Conv1d(n_features, hidden_dim, cfg['kernel_size'], padding=padding)
            self.gelu       = nn.GELU()
            self.relu       = nn.ReLU()
            if cfg['use_layernorm']:
                self.layer_norm = nn.LayerNorm(hidden_dim)
            self.dropout    = nn.Dropout(cfg['dropout_rate'])
        else:
            self.input_proj = nn.Linear(n_features, hidden_dim)
            self.dropout    = nn.Dropout(cfg['dropout_rate'])

        if cfg['use_attention']:
            n_heads = cfg['n_attention_heads']
            while hidden_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=n_heads,
                dropout=cfg['dropout_rate'], batch_first=True,
            )

        self.W_F = nn.Conv1d(2 * hidden_dim, n_features, kernel_size=1)
        nn.init.xavier_uniform_(self.W_F.weight, gain=0.1)
        if self.W_F.bias is not None:
            nn.init.zeros_(self.W_F.bias)

    def forward(self, x, baseline=None):
        if self.cfg['use_conv']:
            h = x.permute(0, 2, 1)
            h = self.conv(h)
            h = h.permute(0, 2, 1)
            h = self.gelu(h) if self.cfg['activation'] == 'gelu' else self.relu(h)
            if self.cfg['use_layernorm']:
                h = self.layer_norm(h)
            H_conv = self.dropout(h)
        else:
            H_conv = self.dropout(self.input_proj(x))

        H_attn     = self.attention(H_conv, H_conv, H_conv)[0] if self.cfg['use_attention'] else torch.zeros_like(H_conv)
        H_combined = torch.cat([H_conv, H_attn], dim=-1)
        phi_prime  = self.W_F(H_combined.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.sign(phi_prime) * torch.relu(torch.abs(phi_prime) - self.sparsity_threshold)


# ============================
# ABLATION TRAINER
# ============================
class AblationTrainer:
    """Trains one AblationTDENetwork variant. Not used for the baseline variant."""

    def __init__(self, cfg):
        self.cfg              = cfg
        self.device           = device
        self.network          = None
        self.baseline         = None
        self.base_pred        = None
        self._shapley_probs   = None
        self.scaler           = GradScaler() if torch.cuda.is_available() else None
        self.history          = {'train_loss': [], 'val_loss': [], 'lr': []}
        self.best_val_loss    = float('inf')
        self.n_epochs_trained = 0
        self.training_time_s  = 0.0

    def _setup(self, X_train, model_predict_func):
        T, D             = X_train.shape[1], X_train.shape[2]
        self.T, self.D   = T, D
        self.model_func  = model_predict_func
        X_t              = torch.FloatTensor(X_train).to(self.device)
        self.baseline    = torch.median(X_t, dim=0)[0]
        b_np             = self.baseline.unsqueeze(0).cpu().numpy()
        bp               = model_predict_func(b_np)
        self.base_pred   = torch.tensor(float(np.atleast_1d(bp).flatten()[0]), dtype=torch.float32, device=self.device)
        self.network     = AblationTDENetwork(T, D, self.cfg).to(self.device)
        if D > 1:
            k         = torch.arange(1, D, device=self.device, dtype=torch.float64)
            log_binom = (torch.lgamma(torch.tensor(D + 1.0, device=self.device, dtype=torch.float64))
                         - torch.lgamma(k + 1) - torch.lgamma(D - k + 1))
            w         = ((D - 1) / (k * (D - k) * torch.exp(log_binom) + 1e-10)).float()
            self._shapley_probs = w / w.sum()
        else:
            self._shapley_probs = torch.ones(1, device=self.device)

    def _generate_masks(self, batch_size):
        D, total = self.D, batch_size * self.cfg['samples_per_feature']
        if self.cfg['masking_strategy'] == 'shapley':
            k_idx     = torch.multinomial(self._shapley_probs, total, replacement=True)
            k_samples = torch.arange(1, D, device=self.device)[k_idx]
        else:
            k_samples = torch.randint(1, D, (total,), device=self.device)
        rand       = torch.rand(total, D, device=self.device)
        masks_feat = (torch.argsort(rand, dim=1) < k_samples.unsqueeze(1)).float()
        masks      = masks_feat.unsqueeze(1).expand(-1, self.T, -1).contiguous()
        if self.cfg['paired_sampling']:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks

    def _predict(self, x):
        with torch.no_grad():
            if isinstance(x, torch.Tensor): x = x.cpu().numpy()
            out = self.model_func(x)
            return torch.tensor(np.atleast_1d(out).flatten(), dtype=torch.float32, device=self.device)

    def _process_batch(self, X_batch, optimizer):
        bs      = X_batch.size(0)
        X_batch = X_batch.to(self.device, non_blocking=True)
        S       = self.cfg['samples_per_feature']
        X_exp   = X_batch.repeat(S, 1, 1)
        masks   = self._generate_masks(bs)
        total   = masks.size(0)
        rep     = max(1, total // (bs * S))
        X_p     = X_exp.repeat(rep, 1, 1)[:total]
        b_p     = self.baseline.unsqueeze(0).expand(total, -1, -1).contiguous()
        X_masked = torch.addcmul(b_p, X_p - b_p, masks)
        f_masked = self._predict(X_masked)

        use_amp = self.scaler is not None and self.device.type == 'cuda'
        with autocast(enabled=use_amp):
            phi      = self.network(X_masked, self.baseline)
            fidelity = ((phi.sum(dim=(1, 2)) - (f_masked - self.base_pred)) ** 2).mean()
            loss     = fidelity
            if self.cfg['use_l1']:
                loss = loss + self.cfg['l1_lambda'] * torch.abs(phi).mean()
            if self.cfg['use_l2']:
                loss = loss + self.cfg['l2_lambda'] * (phi ** 2).mean()
            if self.cfg['use_smoothness'] and phi.size(1) > 1:
                loss = loss + self.cfg['smoothness_lambda'] * (phi[:, 1:] - phi[:, :-1]).pow(2).mean()

        if not torch.isfinite(loss):
            return float('inf')

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()
        return loss.item()

    def _validate(self, X_val):
        self.network.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_val)), batch_size=self.cfg['batch_size'], shuffle=False)
        total, n = 0.0, 0
        with torch.no_grad():
            for (xb,) in loader:
                xb    = xb.to(self.device)
                phi   = self.network(xb, self.baseline)
                preds = self._predict(xb)
                err   = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                if torch.isfinite(err):
                    total += err.item(); n += 1
        self.network.train()
        return total / max(n, 1) if n > 0 else float('inf')

    def train(self, X_train, X_val, model_predict_func):
        self._setup(X_train, model_predict_func)
        use_cuda  = self.device.type == 'cuda'
        n_workers = 4 if use_cuda else 0
        eff_bs    = min(self.cfg['batch_size'], len(X_train) - 1)
        if eff_bs < 1:
            return float('inf'), 0.0, 0
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=eff_bs, shuffle=True,
            num_workers=n_workers, pin_memory=use_cuda,
            persistent_workers=n_workers > 0,
            prefetch_factor=2 if n_workers > 0 else None,
            drop_last=True,
        )
        opt_cls   = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(self.cfg['optimizer_type'], torch.optim.Adam)
        optimizer = opt_cls(self.network.parameters(), lr=self.cfg['learning_rate'], weight_decay=self.cfg['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, min_lr=self.cfg['min_lr'])
        best_val, best_weights, no_improve = float('inf'), None, 0
        t0 = time.time()

        for epoch in range(self.cfg['n_epochs']):
            self.network.train()
            epoch_loss, n_batches = 0.0, 0
            for (xb,) in loader:
                bl = self._process_batch(xb, optimizer)
                if bl != float('inf'):
                    epoch_loss += bl; n_batches += 1
            if n_batches == 0:
                break
            epoch_loss /= n_batches
            val_loss    = self._validate(X_val)
            if val_loss == float('inf'):
                continue
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            self.n_epochs_trained = epoch + 1
            if val_loss < best_val - 1e-6:
                best_val     = val_loss
                best_weights = {k: v.clone() for k, v in self.network.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
            if no_improve >= self.cfg['patience']:
                break

        self.training_time_s = time.time() - t0
        self.best_val_loss   = best_val
        if best_weights:
            self.network.load_state_dict(best_weights)
        self.network.eval()
        return best_val, self.training_time_s, self.n_epochs_trained

    def explain(self, sample):
        if self.network is None:
            raise ValueError("Trainer not trained yet.")
        if isinstance(sample, np.ndarray):
            sample = torch.FloatTensor(sample)
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        sample = sample.to(self.device)
        self.network.eval()
        t0 = time.time()
        with torch.no_grad():
            phi = self.network(sample, self.baseline).cpu().numpy()[0]
        return phi, (time.time() - t0) * 1000.0


# ============================
# DATABASE
# ============================
def init_database(db_path=ABLATION_DB):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ablation_runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_use       TEXT    NOT NULL,
            option_number     INTEGER NOT NULL,
            model_name        TEXT    NOT NULL,
            variant_key       TEXT    NOT NULL,
            variant_group     TEXT    NOT NULL,
            variant_desc      TEXT    NOT NULL,
            config_json       TEXT    NOT NULL,
            best_val_loss     REAL,
            training_time_s   REAL,
            n_epochs_trained  INTEGER,
            status            TEXT    NOT NULL DEFAULT 'pending',
            timestamp         TEXT,
            UNIQUE(primary_use, option_number, model_name, variant_key)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ablation_samples (
            primary_use          TEXT    NOT NULL,
            option_number        INTEGER NOT NULL,
            sample_idx           INTEGER NOT NULL,
            sample_json          TEXT    NOT NULL,
            noisy_sample_json    TEXT    NOT NULL,
            PRIMARY KEY (primary_use, option_number, sample_idx)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ablation_metrics (
            run_id                        INTEGER NOT NULL,
            sample_idx                    INTEGER NOT NULL,
            inference_time_ms             REAL,
            fidelity                      REAL,
            sparsity                      REAL,
            complexity                    REAL,
            efficiency_error              REAL,
            shap_values_json              TEXT,
            noisy_inference_time_ms       REAL,
            noisy_fidelity                REAL,
            noisy_sparsity                REAL,
            noisy_complexity              REAL,
            noisy_efficiency_error        REAL,
            shap_values_noisy_json        TEXT,
            shap_mae                      REAL,
            reliability_correlation       REAL,
            reliability_topk_overlap      REAL,
            reliability_kendall_tau       REAL,
            reliability_ped               REAL,
            PRIMARY KEY (run_id, sample_idx),
            FOREIGN KEY (run_id) REFERENCES ablation_runs(id)
        )
    ''')
    conn.commit()
    conn.close()


def upsert_run(primary_use, option_number, model_name, cfg, status='pending', db_path=ABLATION_DB):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute('''
        INSERT INTO ablation_runs
            (primary_use, option_number, model_name, variant_key, variant_group,
             variant_desc, config_json, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(primary_use, option_number, model_name, variant_key)
        DO UPDATE SET status=excluded.status, timestamp=excluded.timestamp
    ''', (primary_use, option_number, model_name,
          cfg['variant_key'], cfg['variant_group'], cfg['variant_desc'],
          json.dumps({k: v for k, v in cfg.items() if k not in ('variant_key','variant_group','variant_desc')}),
          status, datetime.now().isoformat()))
    conn.commit()
    cur.execute('SELECT id FROM ablation_runs WHERE primary_use=? AND option_number=? AND model_name=? AND variant_key=?',
                (primary_use, option_number, model_name, cfg['variant_key']))
    run_id = cur.fetchone()[0]
    conn.close()
    return run_id


def update_run_training(run_id, best_val_loss, training_time_s, n_epochs, status='complete', db_path=ABLATION_DB):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        UPDATE ablation_runs SET best_val_loss=?, training_time_s=?, n_epochs_trained=?, status=?, timestamp=? WHERE id=?
    ''', (best_val_loss, training_time_s, n_epochs, status, datetime.now().isoformat(), run_id))
    conn.commit()
    conn.close()


def save_sample(primary_use, option_number, sample_idx, sample, noisy, db_path=ABLATION_DB):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        INSERT OR IGNORE INTO ablation_samples (primary_use, option_number, sample_idx, sample_json, noisy_sample_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (primary_use, option_number, sample_idx, json.dumps(sample.tolist()), json.dumps(noisy.tolist())))
    conn.commit()
    conn.close()


def load_samples(primary_use, option_number, db_path=ABLATION_DB):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sample_idx, sample_json, noisy_sample_json
        FROM ablation_samples WHERE primary_use=? AND option_number=? ORDER BY sample_idx
    ''', (primary_use, option_number))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None
    return {r[0]: {'original': np.array(json.loads(r[1]), dtype=np.float32),
                   'noisy':    np.array(json.loads(r[2]), dtype=np.float32)} for r in rows}


def save_metrics(run_id, sample_idx,
                 inference_ms, fidelity, sparsity, complexity, eff_err, shap_vals,
                 noisy_inference_ms, noisy_fidelity, noisy_sparsity, noisy_complexity,
                 noisy_eff_err, shap_vals_noisy,
                 shap_mae, rel_corr, rel_topk, rel_kt, rel_ped,
                 db_path=ABLATION_DB):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        INSERT OR REPLACE INTO ablation_metrics
            (run_id, sample_idx,
             inference_time_ms, fidelity, sparsity, complexity, efficiency_error, shap_values_json,
             noisy_inference_time_ms, noisy_fidelity, noisy_sparsity, noisy_complexity,
             noisy_efficiency_error, shap_values_noisy_json,
             shap_mae, reliability_correlation, reliability_topk_overlap,
             reliability_kendall_tau, reliability_ped)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, sample_idx,
          inference_ms, fidelity, sparsity, complexity, eff_err,
          json.dumps(shap_vals.tolist()) if shap_vals is not None else None,
          noisy_inference_ms, noisy_fidelity, noisy_sparsity, noisy_complexity, noisy_eff_err,
          json.dumps(shap_vals_noisy.tolist()) if shap_vals_noisy is not None else None,
          shap_mae, rel_corr, rel_topk, rel_kt, rel_ped))
    conn.commit()
    conn.close()


def delete_existing_run(primary_use, option_number, model_name, variant_key, db_path=ABLATION_DB):
    """Delete any existing run row and its metric rows so results can be replaced cleanly."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute('''
        SELECT id FROM ablation_runs
        WHERE primary_use=? AND option_number=? AND model_name=? AND variant_key=?
    ''', (primary_use, option_number, model_name, variant_key))
    row = cur.fetchone()
    if row:
        run_id = row[0]
        conn.execute('DELETE FROM ablation_metrics WHERE run_id=?', (run_id,))
        conn.execute('DELETE FROM ablation_runs WHERE id=?', (run_id,))
        conn.commit()
    conn.close()


# ============================
# METRICS
# ============================
NOISE_STD           = 0.05
FIDELITY_TOPK_PCT   = 10.0
SPARSITY_THRESH_PCT = 1.0
RELIABILITY_TOPK    = 10.0

class _WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out[:, 0:1] if out.ndim > 1 and out.shape[1] > 0 else out

def _pred_scalar(wrapped, x_np, T, D):
    xt = torch.tensor(x_np.reshape(-1, T, D), dtype=torch.float32, device=device)
    with torch.no_grad():
        return wrapped(xt).cpu().numpy().flatten()[0]

def compute_fidelity(wrapped, sample, shap_vals, baseline_2d, T, D):
    if shap_vals is None: return None
    try:
        orig = _pred_scalar(wrapped, sample, T, D)
        k    = max(1, int(shap_vals.size * FIDELITY_TOPK_PCT / 100))
        top  = np.argsort(np.abs(shap_vals).flatten())[-k:]
        msk  = sample.copy()
        for idx in top:
            msk[idx // D, idx % D] = baseline_2d[idx // D, idx % D]
        return float(abs(orig - _pred_scalar(wrapped, msk, T, D)))
    except Exception:
        return None

def compute_sparsity(shap_vals):
    if shap_vals is None: return None
    mv = np.max(np.abs(shap_vals))
    return 100.0 if mv == 0 else float(np.sum(np.abs(shap_vals) < mv * SPARSITY_THRESH_PCT / 100) / shap_vals.size * 100)

def compute_complexity(shap_vals):
    if shap_vals is None: return None
    p = np.abs(shap_vals).flatten() + 1e-10
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))

def compute_shap_mae(phi_orig, phi_noisy):
    if phi_orig is None or phi_noisy is None: return None
    try:
        return float(np.mean(np.abs(phi_orig.flatten() - phi_noisy.flatten())))
    except Exception:
        return None

def compute_reliability(shap_orig, shap_noisy):
    null = {'correlation': None, 'topk_overlap': None, 'kendall_tau': None, 'ped': None}
    if shap_orig is None or shap_noisy is None: return null
    try:
        o, n  = shap_orig.flatten(), shap_noisy.flatten()
        mask  = np.isfinite(o) & np.isfinite(n)
        if mask.sum() < 10: return null
        ov, nv  = o[mask], n[mask]
        corr    = float(pearsonr(ov, nv)[0])
        k       = max(1, int(len(o) * RELIABILITY_TOPK / 100))
        overlap = float(len(set(np.argsort(np.abs(o))[-k:]) & set(np.argsort(np.abs(n))[-k:])) / k * 100)
        kt      = float(kendalltau(ov, nv)[0])
        mx      = max(np.max(np.abs(ov)), np.max(np.abs(nv)), 1e-10)
        ped     = float(np.mean(np.abs(ov - nv)) / mx * 100)
        return {
            'correlation':  corr    if np.isfinite(corr)    else None,
            'topk_overlap': overlap if np.isfinite(overlap) else None,
            'kendall_tau':  kt      if np.isfinite(kt)      else None,
            'ped':          ped     if np.isfinite(ped)     else None,
        }
    except Exception:
        return null

def compute_efficiency_error(wrapped, sample, shap_vals, baseline_2d, T, D):
    if shap_vals is None: return None
    try:
        diff = _pred_scalar(wrapped, sample, T, D) - _pred_scalar(wrapped, baseline_2d, T, D)
        return float(abs(np.sum(shap_vals) - diff) / (abs(diff) + 1e-10))
    except Exception:
        return None


# ============================
# DATA HELPERS
# ============================
def load_dataset(primary_use, option_number):
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path=ENERGY_DB, primary_use=primary_use, option_number=option_number,
        scaled=True, scale_type="both",
    )

def get_datasets():
    """Return only datasets for which at least one TDE has been trained."""
    explainer_db = PATH_DBS / "explainer_results.db"
    if not explainer_db.exists():
        print("❌ explainer_results.db not found — run tde.py first.")
        return []
    conn = sqlite3.connect(explainer_db)
    df   = pd.read_sql_query(
        "SELECT DISTINCT primary_use, option_number FROM explainer_metadata WHERE explainer_type='TDE' ORDER BY primary_use, option_number",
        conn)
    conn.close()
    if df.empty:
        print("❌ No trained TDE explainers found — run tde.py first.")
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])} for _, r in df.iterrows()]

def get_models(primary_use, option_number):
    """Return only models for which a TDE has been trained."""
    explainer_db = PATH_DBS / "explainer_results.db"
    if not explainer_db.exists():
        return []
    conn = sqlite3.connect(explainer_db)
    df   = pd.read_sql_query(
        "SELECT DISTINCT model_name FROM explainer_metadata WHERE primary_use=? AND option_number=? AND explainer_type='TDE'",
        conn, params=(primary_use, option_number))
    conn.close()
    return df['model_name'].tolist()

def make_predict_fn(model, T, D):
    def predict(X):
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        xt = torch.tensor(X.reshape(-1, T, D), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(xt).cpu().numpy()
        return out[:, 0] if out.ndim > 1 and out.shape[1] > 0 else out.flatten()
    return predict


# ============================
# SHARED EVALUATION LOOP
# ============================
def _evaluate_samples(run_id, explainer_explain_fn, wrapped, baseline_2d, T, D, samples, verbose_prefix=''):
    """Run the metric evaluation loop shared by both baseline and ablation variants.

    explainer_explain_fn(sample) must return (phi_np, inference_ms).
    """
    sample_metrics = []
    for s_idx, s_data in samples.items():
        orig_sample  = s_data['original']
        noisy_sample = s_data['noisy']

        phi_orig,  infer_ms       = explainer_explain_fn(orig_sample)
        fid      = compute_fidelity(wrapped, orig_sample,  phi_orig,  baseline_2d, T, D)
        spa      = compute_sparsity(phi_orig)
        com      = compute_complexity(phi_orig)
        eff      = compute_efficiency_error(wrapped, orig_sample, phi_orig, baseline_2d, T, D)

        phi_noisy, noisy_infer_ms = explainer_explain_fn(noisy_sample)
        noisy_fid = compute_fidelity(wrapped, noisy_sample, phi_noisy, baseline_2d, T, D)
        noisy_spa = compute_sparsity(phi_noisy)
        noisy_com = compute_complexity(phi_noisy)
        noisy_eff = compute_efficiency_error(wrapped, noisy_sample, phi_noisy, baseline_2d, T, D)

        mae = compute_shap_mae(phi_orig, phi_noisy)
        rel = compute_reliability(phi_orig, phi_noisy)

        save_metrics(
            run_id, s_idx,
            infer_ms,       fid,       spa,       com,       eff,       phi_orig,
            noisy_infer_ms, noisy_fid, noisy_spa, noisy_com, noisy_eff, phi_noisy,
            mae, rel['correlation'], rel['topk_overlap'], rel['kendall_tau'], rel['ped'],
        )
        sample_metrics.append({
            'sample_idx':       s_idx,
            'inference_ms':     infer_ms,
            'fidelity':         fid,
            'sparsity':         spa,
            'complexity':       com,
            'efficiency_error': eff,
            'mae':              mae,
            'kendall_tau':      rel['kendall_tau'],
        })
    return sample_metrics


# ============================
# CORE: RUN BASELINE (load from disk)
# ============================
def run_baseline_variant(primary_use, option_number, model_name,
                          samples, model_obj, T, D, verbose=True):
    """Load the TDE explainer saved by tde.py and evaluate it — no retraining."""
    variant_key = 'baseline'
    delete_existing_run(primary_use, option_number, model_name, variant_key)

    cfg     = resolve_config(variant_key)
    wrapped = _WrappedModel(model_obj).to(device)
    wrapped.eval()

    exp_path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / "tde"
    if not (exp_path / "tde_explainer.pt").exists():
        print(f"      ❌ baseline — TDE explainer not found at {exp_path} (run tde.py first)")
        return None

    explainer   = TemporalDeepExplainer.load(str(exp_path), filename="tde_explainer", device_override=device)
    baseline_2d = explainer.baseline.cpu().numpy()

    # Pull training metadata recorded by tde.py for the DB row
    best_val, train_t = None, None
    try:
        explainer_db = PATH_DBS / "explainer_results.db"
        conn   = sqlite3.connect(explainer_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT best_validation_loss, training_time FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type='TDE'",
            (primary_use, option_number, model_name))
        row = cursor.fetchone()
        conn.close()
        if row:
            best_val, train_t = row[0], row[1]
    except Exception:
        pass

    run_id = upsert_run(primary_use, option_number, model_name, cfg, status='evaluating')

    if verbose:
        print(f"      ✅ {'baseline':<20} [baseline] Loaded from disk — not retrained")

    def explain_fn(sample):
        t0  = time.time()
        phi = explainer.explain(sample)
        return phi, (time.time() - t0) * 1000.0

    sample_metrics = _evaluate_samples(run_id, explain_fn, wrapped, baseline_2d, T, D, samples)
    update_run_training(run_id, best_val, train_t, None, status='complete')

    if verbose:
        avg = lambda key: np.nanmean([m[key] for m in sample_metrics if m[key] is not None] or [float('nan')])
        print(f"         fid={avg('fidelity'):.4f}  sparsity={avg('sparsity'):.1f}%  "
              f"infer={avg('inference_ms'):.2f}ms  MAE={avg('mae'):.5f}  τ={avg('kendall_tau'):.4f}")

    return {
        'variant_key':     variant_key,
        'run_id':          run_id,
        'best_val_loss':   best_val,
        'training_time_s': train_t,
        'n_epochs':        None,
        'sample_metrics':  sample_metrics,
    }


# ============================
# CORE: RUN ONE ABLATION VARIANT (retrain)
# ============================
def run_variant(primary_use, option_number, model_name, variant_key,
                X_train, X_val, model_predict_fn, samples, model_obj,
                tuned_base=None, verbose=True):
    """Train an ablation variant from scratch and evaluate it."""
    delete_existing_run(primary_use, option_number, model_name, variant_key)

    cfg     = resolve_config(variant_key, tuned_base)
    T, D    = X_train.shape[1], X_train.shape[2]
    wrapped = _WrappedModel(model_obj).to(device)
    wrapped.eval()
    run_id  = upsert_run(primary_use, option_number, model_name, cfg, status='running')

    if verbose:
        print(f"      🔄 {variant_key:<20} [{cfg['variant_group']}] {cfg['variant_desc']}")

    trainer = AblationTrainer(cfg)
    try:
        best_val, train_t, n_ep = trainer.train(X_train, X_val, model_predict_fn)
    except Exception as e:
        print(f"         ❌ Training failed: {e}")
        update_run_training(run_id, None, None, None, status='failed')
        return None

    baseline_2d = trainer.baseline.cpu().numpy()

    if verbose:
        print(f"         val={best_val:.5f}  t={train_t:.1f}s  epochs={n_ep}")

    update_run_training(run_id, best_val, train_t, n_ep, status='evaluating')

    sample_metrics = _evaluate_samples(run_id, trainer.explain, wrapped, baseline_2d, T, D, samples)
    update_run_training(run_id, best_val, train_t, n_ep, status='complete')

    if verbose:
        avg = lambda key: np.nanmean([m[key] for m in sample_metrics if m[key] is not None] or [float('nan')])
        print(f"         fid={avg('fidelity'):.4f}  sparsity={avg('sparsity'):.1f}%  "
              f"infer={avg('inference_ms'):.2f}ms  MAE={avg('mae'):.5f}  τ={avg('kendall_tau'):.4f}")

    return {
        'variant_key':     variant_key,
        'run_id':          run_id,
        'best_val_loss':   best_val,
        'training_time_s': train_t,
        'n_epochs':        n_ep,
        'sample_metrics':  sample_metrics,
    }


# ============================
# MAIN RUNNER
# ============================
def run_ablation_study(primary_use, option_number, models, n_samples, n_epochs,
                       training_fraction, variants_to_run=None):
    if variants_to_run is None:
        variants_to_run = list(ABLATION_VARIANTS.keys())

    n_retrain = sum(1 for vk in variants_to_run if ABLATION_VARIANTS[vk]['group'] != 'baseline')
    print(f"\n{'='*80}")
    print(f"🔬 ABLATION STUDY: {primary_use} — Option {option_number}")
    print(f"   Variants: {len(variants_to_run)} (1 loaded from disk + {n_retrain} retrained)  |  Models: {len(models)}  |  Samples: {n_samples}")
    print(f"{'='*80}")

    container = load_dataset(primary_use, option_number)
    X_all = np.concatenate([container.X_train, container.X_val], axis=0)
    T, D  = X_all.shape[1], X_all.shape[2]

    np.random.seed(42)
    n_use        = max(1, int(len(X_all) * training_fraction))
    X_sub        = X_all[np.random.choice(len(X_all), n_use, replace=False)]
    n_val        = max(1, int(len(X_sub) * 0.20))
    X_train, X_val = X_sub[:-n_val], X_sub[-n_val:]
    print(f"   Training subset: {len(X_train)} train / {len(X_val)} val  ({training_fraction:.0%} of data)")

    existing_samples = load_samples(primary_use, option_number)
    if existing_samples:
        samples = existing_samples
        print(f"   Loaded {len(samples)} locked test samples from DB")
    else:
        np.random.seed(42)
        idx_pool = np.random.choice(len(container.X_test), min(n_samples, len(container.X_test)), replace=False)
        samples  = {}
        for i in idx_pool:
            orig  = container.X_test[i].astype(np.float32)
            noisy = np.clip(orig + np.random.normal(0, NOISE_STD, orig.shape), 0, 1).astype(np.float32)
            save_sample(primary_use, option_number, int(i), orig, noisy)
            samples[int(i)] = {'original': orig, 'noisy': noisy}
        print(f"   Created and locked {len(samples)} test samples (noise_std={NOISE_STD})")

    global BASE_CONFIG
    BASE_CONFIG['n_epochs'] = n_epochs

    for model_name in models:
        print(f"\n  🤖 MODEL: {model_name}")
        model_dir  = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
        model_path = model_dir / "trained_model.pt"
        if not model_path.exists():
            print(f"     ❌ Model not found: {model_path}")
            continue

        model      = load_complete_model(str(model_path), device=device)
        predict_fn = make_predict_fn(model, T, D)
        tuned_base = load_best_tde_params(primary_use, option_number, model_name)

        current_group = None
        for vk in variants_to_run:
            grp = ABLATION_VARIANTS[vk]['group']
            if grp != current_group:
                current_group = grp
                print(f"\n    ── {grp.upper()} ──")

            if vk == 'baseline':
                run_baseline_variant(primary_use, option_number, model_name,
                                     samples, model, T, D)
            else:
                run_variant(primary_use, option_number, model_name, vk,
                            X_train, X_val, predict_fn, samples, model,
                            tuned_base=tuned_base)

        del model
        torch.cuda.empty_cache()


# ============================
# SUMMARY TABLE
# ============================
def print_summary(primary_use=None, option_number=None, db_path=ABLATION_DB):
    conn = sqlite3.connect(db_path)
    where, params = '', []
    if primary_use is not None:
        where  = 'WHERE ar.primary_use=? AND ar.option_number=?'
        params = [primary_use, option_number]

    df = pd.read_sql_query(f'''
        SELECT ar.variant_key, ar.variant_desc, ar.model_name,
               ar.best_val_loss   AS val_loss,
               ar.training_time_s AS train_time_s,
               AVG(am.shap_mae)                AS shap_mae,
               AVG(am.reliability_kendall_tau) AS kendall_tau
        FROM ablation_runs ar
        JOIN ablation_metrics am ON am.run_id = ar.id
        {where}
        GROUP BY ar.variant_key, ar.model_name
        ORDER BY ar.model_name,
                 CASE ar.variant_key WHEN 'baseline' THEN 0 ELSE 1 END,
                 ar.variant_key
    ''', conn, params=params)
    conn.close()

    if df.empty:
        print("\n  No ablation results found.")
        return

    for model_name in sorted(df['model_name'].unique()):
        mdf = df[df['model_name'] == model_name].copy()

        print(f"\n{'='*88}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*88}")
        print(f"  {'Variant':<22} {'Description':<36} {'Val Loss':>10} {'MAE':>10} {'Kendall τ':>10} {'Train (s)':>10}")
        print(f"  {'-'*88}")

        for _, row in mdf.iterrows():
            vl  = f"{row['val_loss']:.5f}"     if not pd.isna(row['val_loss'])     else 'N/A'
            mae = f"{row['shap_mae']:.5f}"     if not pd.isna(row['shap_mae'])     else 'N/A'
            kt  = f"{row['kendall_tau']:.4f}"  if not pd.isna(row['kendall_tau'])  else 'N/A'
            tt  = f"{row['train_time_s']:.1f}" if not pd.isna(row['train_time_s']) else 'N/A'
            print(f"  {row['variant_key']:<22} {row['variant_desc']:<36} {vl:>10} {mae:>10} {kt:>10} {tt:>10}")

        print(f"{'='*88}")


# ============================
# USER INPUT
# ============================
def get_user_inputs():
    print("\n" + "="*80)
    print("🔬 TDE Ablation Study")
    print("="*80)
    print(f"  Groups: architecture ({sum(1 for v in ABLATION_VARIANTS.values() if v['group']=='architecture')} variants), "
          f"masking ({sum(1 for v in ABLATION_VARIANTS.values() if v['group']=='masking')} variant)")
    print("  Baseline: loaded from disk (tde.py output) — not retrained.")
    print("  Hyperparameters for ablation variants loaded from explainer_results.db.")
    print("="*80)

    datasets = get_datasets()
    if not datasets:
        return None

    uses = sorted(set(d['primary_use'] for d in datasets))
    print("\n📁 Primary Uses:")
    for i, u in enumerate(uses): print(f"  {i}: {u}")
    ui = input(f"\n--> Select [0-{len(uses)-1}] [0]: ").strip()
    try:   pu = uses[int(ui)] if ui else uses[0]
    except: pu = uses[0]

    ds_for_use = [d for d in datasets if d['primary_use'] == pu]
    if len(ds_for_use) > 1:
        print(f"\n📋 Options for {pu}:")
        for i, d in enumerate(ds_for_use): print(f"  {i}: option {d['option_number']}")
        oi = input(f"--> Select [0-{len(ds_for_use)-1}] [0]: ").strip()
        try:   ds = ds_for_use[int(oi)] if oi else ds_for_use[0]
        except: ds = ds_for_use[0]
    else:
        ds = ds_for_use[0]

    primary_use, option_number = ds['primary_use'], ds['option_number']

    MODEL_ORDER = ['LSTM', 'BLSTM', 'GRU', 'BGRU', 'CNN1D', 'DCNN', 'TCN', 'TFT', 'TST', 'WaveNet']
    raw_models  = get_models(primary_use, option_number)
    models      = sorted(raw_models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

    print(f"\n🤖 Models with trained TDE:")
    for i, m in enumerate(models, 1): print(f"  {i:>2}: {m}")
    mi = input("--> Select numbers (comma-sep), range (e.g. 1-5), or 'all' [all]: ").strip().lower()
    if mi in ('', 'all'):
        sel_models = models
    else:
        selected_idx = set()
        for part in mi.replace(' ', '').split(','):
            if '-' in part:
                try:
                    lo, hi = part.split('-')
                    selected_idx.update(range(int(lo), int(hi) + 1))
                except ValueError:
                    pass
            elif part.isdigit():
                selected_idx.add(int(part))
        sel_models = [models[i - 1] for i in sorted(selected_idx) if 1 <= i <= len(models)]
        if not sel_models:
            print("  No valid selection — using all models.")
            sel_models = models

    variants = list(ABLATION_VARIANTS.keys())

    n_epochs  = int(input("\n--> Training epochs per ablation variant [50]: ").strip() or 50)
    n_samples = int(input("--> Test samples for metric evaluation [10]: ").strip() or 10)
    frac_in   = input("--> Training data fraction (0.05–1.0) [0.20]: ").strip()
    try:   frac = max(0.05, min(1.0, float(frac_in)))
    except: frac = 0.20

    n_retrain = sum(1 for vk in variants if ABLATION_VARIANTS[vk]['group'] != 'baseline')
    print(f"\n{'='*80}")
    print(f"  Dataset:           {primary_use} — Option {option_number}")
    print(f"  Models:            {', '.join(sel_models)}")
    print(f"  Variants:          {len(variants)} (1 loaded from disk + {n_retrain} retrained)")
    print(f"  Epochs/variant:    {n_epochs}")
    print(f"  Test samples:      {n_samples}")
    print(f"  Training fraction: {frac:.0%}")
    print(f"{'='*80}")

    return {
        'primary_use':       primary_use,
        'option_number':     option_number,
        'models':            sel_models,
        'variants':          variants,
        'n_epochs':          n_epochs,
        'n_samples':         n_samples,
        'training_fraction': frac,
    }


# ============================
# MAIN
# ============================
def main():
    init_database()
    cfg = get_user_inputs()
    if cfg is None:
        return
    confirm = input("\n--> Proceed? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelled.")
        return
    run_ablation_study(
        primary_use=cfg['primary_use'],
        option_number=cfg['option_number'],
        models=cfg['models'],
        n_samples=cfg['n_samples'],
        n_epochs=cfg['n_epochs'],
        training_fraction=cfg['training_fraction'],
        variants_to_run=cfg['variants'],
    )
    show = input("\n--> Show results summary now? (y/n) [y]: ").strip().lower()
    if show in ('', 'y', 'yes'):
        print_summary(cfg['primary_use'], cfg['option_number'])


if __name__ == "__main__":
    main()