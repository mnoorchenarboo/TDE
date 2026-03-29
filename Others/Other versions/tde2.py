"""
TDE & FastSHAP Training System v6.0

Cleaned and optimized version with:
- Removed duplicate code
- Consolidated functions
- Improved error handling
- Better tensor/numpy type management
- Streamlined training pipeline

ARCHITECTURE:
- TDE: Dilated Conv → Multi-Head Attention → Direct Input Connection → Soft Threshold
- FastSHAP: Pure MLP with element-wise masking
"""

# ============================
# LIBRARY IMPORTS
# ============================

import os
import sys
import json
import time
import shutil
import sqlite3
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap

from dl import load_complete_model


# ============================
# CONFIGURATION
# ============================

BENCHMARK_DB = "benchmark_results.db"
EXPLAINER_DB = "explainer_results.db"
RESULTS_BASE_DIR = "results"

# Standardized method names
METHOD_NAMES = {
    'tde': 'TDE',
    'fastshap': 'FastSHAP',
    'gradient_shap': 'Gradient_SHAP',
    'deep_shap': 'Deep_SHAP',
}

NEURAL_EXPLAINER_TYPES = ['tde', 'fastshap']

# Training configuration
DEBUG_MODE = True
DEBUG_TRAINING_FRACTION = 0.20
DEBUG_TRIAL_EPOCHS = 10
DEBUG_FINAL_EPOCHS = 50
DEBUG_N_TRIALS = 5

PROD_TRAINING_FRACTION = 0.30
PROD_TRIAL_EPOCHS = 20
PROD_FINAL_EPOCHS = 100
PROD_N_TRIALS = 30

VALIDATION_SPLIT = 0.20
NOISE_STD = 0.01
EARLY_STOP_PATIENCE = 5
TARGET_SPARSITY = 0.70

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {device}")


# ============================
# UTILITY FUNCTIONS
# ============================

def get_standard_method_name(method_key):
    """Convert method identifier to standardized display name."""
    key_lower = method_key.lower().replace('-', '_').replace(' ', '_')
    return METHOD_NAMES.get(key_lower, method_key)


def get_method_key(method_name):
    """Convert standardized method name to lowercase key."""
    if method_name.lower() in METHOD_NAMES:
        return method_name.lower()
    method_keys = {v: k for k, v in METHOD_NAMES.items()}
    return method_keys.get(method_name, method_name.lower())


def to_numpy(x):
    """Safely convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x, dtype=torch.float32):
    """Safely convert to tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def setup_logger(log_path):
    """Setup logger for training progress."""
    logger = logging.getLogger('Explainer')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================
# DATABASE FUNCTIONS
# ============================

def init_database():
    """Initialize SQLite database for storing explainer results."""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tde_hyperparameter_trials (
            primary_use TEXT, option_number INTEGER, model_name TEXT,
            trial_number INTEGER, hyperparameters TEXT, validation_loss REAL,
            n_training_samples INTEGER, timestamp TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, trial_number))
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fastshap_hyperparameter_trials (
            primary_use TEXT, option_number INTEGER, model_name TEXT,
            trial_number INTEGER, hyperparameters TEXT, validation_loss REAL,
            n_training_samples INTEGER, timestamp TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, trial_number))
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS explainer_metadata (
            primary_use TEXT, option_number INTEGER, model_name TEXT,
            explainer_type TEXT, best_hyperparameters TEXT, best_validation_loss REAL,
            final_training_loss REAL, n_training_samples INTEGER,
            time_steps INTEGER, n_features INTEGER, optimization_time REAL,
            training_time REAL, n_trials INTEGER, explainer_path TEXT,
            feature_names TEXT, timestamp TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, explainer_type))
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparison_results (
            primary_use TEXT, option_number INTEGER, model_name TEXT,
            sample_idx INTEGER, method TEXT, fidelity REAL,
            reliability_correlation REAL, reliability_mse REAL,
            sparsity REAL, complexity REAL, efficiency_error REAL,
            computation_time REAL, timestamp TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, sample_idx, method))
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {EXPLAINER_DB}")


def get_optuna_db_path(primary_use, option_number, model_name, explainer_type):
    """Get path for Optuna study database."""
    optuna_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type.lower()
    optuna_dir.mkdir(parents=True, exist_ok=True)
    return str(optuna_dir / "optuna_study.db")


def save_trial(explainer_type, primary_use, option_number, model_name, trial_num, params, loss, n_train):
    """Save hyperparameter trial to database."""
    table = 'tde_hyperparameter_trials' if explainer_type == 'tde' else 'fastshap_hyperparameter_trials'
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, trial_num, json.dumps(params), 
          loss, n_train, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_explainer_metadata(primary_use, option_number, model_name, explainer_type,
                            best_params, best_loss, final_loss, n_train, time_steps,
                            n_features, opt_time, train_time, n_trials, path, feature_names):
    """Save explainer metadata to database."""
    standard_type = get_standard_method_name(explainer_type)
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO explainer_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, standard_type, json.dumps(best_params),
          best_loss, final_loss, n_train, time_steps, n_features, opt_time, train_time,
          n_trials, path, json.dumps(feature_names), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_comparison(primary_use, option_number, model_name, sample_idx, method,
                    fidelity, rel_corr, rel_mse, sparsity, complexity, eff_err, comp_time):
    """Save comparison results to database."""
    standard_method = get_standard_method_name(method)
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO comparison_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, sample_idx, standard_method,
          fidelity, rel_corr, rel_mse, sparsity, complexity, eff_err, comp_time,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def delete_existing_results(primary_use, option_number, model_name, explainer_type):
    """Delete existing results for a specific explainer before retraining."""
    print(f"    🗑️ Deleting existing results for {model_name}/{explainer_type.upper()}...")
    
    try:
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM explainer_metadata 
            WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?
        ''', (primary_use, option_number, model_name, explainer_type.upper()))
        
        table = 'tde_hyperparameter_trials' if explainer_type.lower() == 'tde' else 'fastshap_hyperparameter_trials'
        cursor.execute(f'''
            DELETE FROM {table} WHERE primary_use=? AND option_number=? AND model_name=?
        ''', (primary_use, option_number, model_name))
        
        cursor.execute('''
            DELETE FROM comparison_results 
            WHERE primary_use=? AND option_number=? AND model_name=? AND method=?
        ''', (primary_use, option_number, model_name, explainer_type.upper()))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"    ⚠️ Database deletion error: {e}")
    
    try:
        optuna_db = get_optuna_db_path(primary_use, option_number, model_name, explainer_type.lower())
        if Path(optuna_db).exists():
            os.remove(optuna_db)
    except Exception as e:
        print(f"    ⚠️ Optuna deletion error: {e}")
    
    try:
        exp_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type.lower()
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
    except Exception as e:
        print(f"    ⚠️ File deletion error: {e}")


# ============================
# MODEL WRAPPER
# ============================

class SingleHorizonWrapper(nn.Module):
    """Wrapper to extract single prediction horizon from multi-horizon models."""
    
    def __init__(self, base_model, horizon_idx=0):
        super().__init__()
        self.base_model = base_model
        self.horizon_idx = horizon_idx
    
    def forward(self, x):
        out = self.base_model(x)
        if out.ndim > 1 and out.shape[1] > self.horizon_idx:
            return out[:, self.horizon_idx:self.horizon_idx + 1]
        return out


# ============================
# TDE NETWORK
# ============================

class TemporalExplainerNetwork(nn.Module):
    """
    Temporal Deep Explainer (TDE) Network.
    
    Architecture:
    1. Dilated Conv - captures local temporal patterns
    2. Attention Gate - learns global importance weights
    3. Direct Input Connection - ensures input-dependence
    4. Soft Thresholding - promotes sparsity
    """
    
    def __init__(self, time_steps, n_features, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, use_attention_gate=True):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.sparsity_threshold = sparsity_threshold
        self.use_attention_gate = use_attention_gate
        
        # Dilated Temporal Convolutions
        conv_layers = []
        in_channels = n_features
        for i in range(n_conv_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            conv_layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = hidden_dim
        self.conv = nn.Sequential(*conv_layers)
        self.output_proj = nn.Conv1d(hidden_dim, n_features, 1)
        
        # Attention Gate
        if use_attention_gate:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=n_attention_heads,
                dropout=dropout_rate, batch_first=True
            )
            self.gate_proj = nn.Sequential(
                nn.Conv1d(hidden_dim, n_features, 1),
                nn.Sigmoid()
            )
        
        # Direct Input Connection
        self.input_weight = nn.Parameter(torch.zeros(time_steps, n_features))
        
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, baseline=None):
        h = x.permute(0, 2, 1)
        h = self.conv(h)
        conv_out = self.output_proj(h)
        
        if self.use_attention_gate:
            h_att = h.permute(0, 2, 1)
            attn_out, _ = self.attention(h_att, h_att, h_att)
            attn_out = attn_out.permute(0, 2, 1)
            gate = self.gate_proj(attn_out)
            conv_out = conv_out * gate
        
        conv_out = conv_out.permute(0, 2, 1)
        
        if baseline is not None:
            if baseline.dim() == 2:
                baseline = baseline.unsqueeze(0)
            diff = x - baseline
        else:
            diff = x
        
        input_contrib = diff * torch.tanh(self.input_weight).unsqueeze(0)
        output = conv_out + input_contrib
        output = torch.sign(output) * torch.relu(torch.abs(output) - self.sparsity_threshold)
        
        return output


# ============================
# FASTSHAP NETWORK
# ============================

class FastSHAPNetwork(nn.Module):
    """Pure MLP for FastSHAP value prediction."""
    
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout_rate=0.2):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


# ============================
# BASE EXPLAINER CLASS
# ============================

class BaseExplainer:
    """Base class with shared functionality for TDE and FastSHAP."""
    
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True,
                 min_lr=1e-6, weight_decay=1e-4, optimizer_type='adam',
                 learning_rate=1e-3, paired_sampling=True, samples_per_feature=2, **kwargs):
        
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.model_predict_func = None
        
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        self._gpu_model = None
        self._model_on_gpu = False
        self.scaler = GradScaler() if torch.cuda.is_available() else None
    
    def _compute_shapley_kernel(self, d):
        """Compute Shapley kernel weights for coalition sampling."""
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float64)
        log_binom = (torch.lgamma(torch.tensor(d + 1.0, device=self.device, dtype=torch.float64))
                     - torch.lgamma(k_values + 1) - torch.lgamma(d - k_values + 1))
        binom_coeffs = torch.exp(log_binom)
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        weights = weights.float()
        probs = weights / weights.sum()
        return weights, probs
    
    def _get_predictions(self, inputs, time_steps=None, n_features=None):
        """Get predictions from black-box model."""
        ts = time_steps or self.time_steps
        nf = n_features or self.n_features
        
        with torch.no_grad():
            if self._model_on_gpu and self._gpu_model is not None:
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                elif inputs.device != self.device:
                    inputs = inputs.to(self.device)
                
                if inputs.ndim == 2:
                    inputs = inputs.view(-1, ts, nf)
                
                pred = self._gpu_model(inputs)
                if pred.ndim > 1 and pred.shape[1] > 0:
                    return pred[:, 0]
                return pred.flatten()
            else:
                if isinstance(inputs, torch.Tensor):
                    inputs_np = inputs.cpu().numpy()
                else:
                    inputs_np = inputs
                
                if inputs_np.ndim == 2:
                    inputs_np = inputs_np.reshape(-1, ts, nf)
                
                preds = self.model_predict_func(inputs_np)
                return torch.tensor(np.atleast_1d(preds).flatten(),
                                   dtype=torch.float32, device=self.device)
    
    def _create_dataloader(self, X, shuffle=True):
        """Create optimized DataLoader."""
        use_cuda = self.device.type == 'cuda'
        num_workers = 4 if use_cuda else 0
        effective_batch = min(self.batch_size, len(X) - 1)
        
        if effective_batch < 1:
            return None
        
        return DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=effective_batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
    
    def _get_optimizer(self):
        """Create optimizer and scheduler."""
        opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(
            self.optimizer_type, torch.optim.Adam)
        optimizer = opt_cls(self.explainer.parameters(), lr=self.learning_rate,
                           weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2,
                                      factor=0.5, min_lr=self.min_lr)
        return optimizer, scheduler
    
    def _training_loop(self, loader, val_data, optimizer, scheduler, process_batch_func, validate_func):
        """Common training loop."""
        best_val, best_weights, no_improve = float('inf'), None, 0
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            epoch_loss, n_batches = 0.0, 0
            
            for (X_batch,) in loader:
                batch_loss = process_batch_func(X_batch, optimizer)
                if batch_loss != float('inf'):
                    epoch_loss += batch_loss
                    n_batches += 1
            
            if n_batches == 0:
                if self.verbose:
                    print(f"    [ERROR] All batches failed at epoch {epoch + 1}")
                return float('inf')
            
            epoch_loss /= n_batches
            val_loss = validate_func(val_data)
            
            if val_loss == float('inf'):
                continue
            
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                if hasattr(self.explainer, '_orig_mod'):
                    best_weights = {k: v.clone() for k, v in self.explainer._orig_mod.state_dict().items()}
                else:
                    best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch + 1:3d} | L:{epoch_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")
            
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [STOP] epoch {epoch + 1}")
                break
        
        if best_weights:
            if hasattr(self.explainer, '_orig_mod'):
                self.explainer._orig_mod.load_state_dict(best_weights)
            else:
                self.explainer.load_state_dict(best_weights)
        
        self.best_loss = best_val
        return best_val


# ============================
# TDE TRAINER
# ============================

class TemporalDeepExplainer(BaseExplainer):
    """Trainer for Temporal Deep Explainer (TDE)."""
    
    def __init__(self, l1_lambda=0.01, l2_lambda=0.01, smoothness_lambda=0.1,
                 efficiency_lambda=0.1, sparsity_lambda=0.1, target_sparsity=0.70,
                 hidden_dim=128, n_conv_layers=2, kernel_size=3, dropout_rate=0.2,
                 sparsity_threshold=0.01, n_attention_heads=4, window_size=6,
                 masking_mode='window', **kwargs):
        
        super().__init__(**kwargs)
        
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda
        self.sparsity_lambda = sparsity_lambda
        self.target_sparsity = target_sparsity
        self.hidden_dim = hidden_dim
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.sparsity_threshold = sparsity_threshold
        self.n_attention_heads = n_attention_heads
        self.window_size = window_size
        self.masking_mode = masking_mode
        
        self.n_windows = None
        self._shapley_probs_features = None
        self._baseline_cache = None
        
        self._init_params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        self._init_params.update(kwargs)
    
    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        """Initialize explainer network and compute baseline."""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.n_windows = (self.time_steps + self.window_size - 1) // self.window_size
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        if gpu_model is not None:
            self._gpu_model = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        base_preds = self._get_predictions(self.baseline.unsqueeze(0))
        self.base_pred = base_preds[0] if base_preds.numel() > 0 else base_preds
        if not isinstance(self.base_pred, torch.Tensor):
            self.base_pred = torch.tensor(float(self.base_pred), device=self.device)
        
        self.explainer = TemporalExplainerNetwork(
            self.time_steps, self.n_features, self.hidden_dim, self.n_conv_layers,
            self.kernel_size, self.dropout_rate, self.sparsity_threshold,
            self.n_attention_heads
        ).to(self.device)
        
        _, self._shapley_probs_features = self._compute_shapley_kernel(self.n_features)
        self._baseline_cache = None
        
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.explainer = torch.compile(self.explainer, mode='reduce-overhead')
            except Exception:
                pass
    
    def _generate_masks(self, batch_size):
        """Generate masks based on selected masking mode."""
        total = batch_size * self.samples_per_feature
        
        if self.masking_mode == 'window':
            masks = torch.ones(total, self.time_steps, self.n_features, device=self.device)
            max_windows = max(2, self.n_windows)
            n_windows_to_mask = torch.randint(1, max_windows, (total,), device=self.device)
            n_features_to_mask = torch.randint(1, self.n_features + 1, (total,), device=self.device)
            window_rand = torch.rand(total, self.n_windows, device=self.device)
            feature_rand = torch.rand(total, self.n_features, device=self.device)
            
            for i in range(total):
                _, top_windows = torch.topk(window_rand[i], n_windows_to_mask[i].item())
                _, top_features = torch.topk(feature_rand[i], n_features_to_mask[i].item())
                for w_idx in top_windows:
                    start = w_idx.item() * self.window_size
                    end = min(start + self.window_size, self.time_steps)
                    masks[i, start:end, top_features] = 0.0
        else:
            probs_f = self._shapley_probs_features
            k_idx = torch.multinomial(probs_f, total, replacement=True)
            k_samples = torch.arange(1, self.n_features, device=self.device)[k_idx]
            rand = torch.rand(total, self.n_features, device=self.device)
            sorted_idx = torch.argsort(rand, dim=1)
            masks = (sorted_idx < k_samples.unsqueeze(1)).float()
            masks = masks.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks
    
    def _process_batch(self, X_batch, optimizer):
        """Process single training batch."""
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device, non_blocking=True)
        
        expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks = self._generate_masks(batch_size)
        total = masks.size(0)
        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat_factor, 1, 1)[:total]
        
        if self._baseline_cache is None or self._baseline_cache.size(0) < total:
            max_cache = max(total, self.batch_size * self.samples_per_feature * 4)
            self._baseline_cache = self.baseline.unsqueeze(0).expand(max_cache, -1, -1).contiguous().clone()
        baseline_paired = self._baseline_cache[:total]
        
        masked = torch.addcmul(baseline_paired, X_paired - baseline_paired, masks)
        preds_masked = self._get_predictions(masked)
        
        if self.paired_sampling:
            n_unique = total // 2
            preds_unique = self._get_predictions(X_paired[:n_unique])
            preds_orig = preds_unique.repeat(2)
        else:
            preds_orig = self._get_predictions(X_paired)
        
        use_amp = self.scaler is not None and self.device.type == 'cuda'
        
        with autocast(enabled=use_amp):
            phi = self.explainer(X_paired, self.baseline)
            
            masked_sum = (masks * phi).sum(dim=(1, 2))
            pred_diff = preds_masked - self.base_pred
            coalition_loss = ((masked_sum - pred_diff) ** 2).mean()
            
            phi_sum = phi.sum(dim=(1, 2))
            orig_diff = preds_orig - self.base_pred
            eff_loss = self.efficiency_lambda * ((phi_sum - orig_diff) ** 2).mean()
            
            smooth_loss = self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean() if phi.size(1) > 1 else torch.tensor(0.0, device=self.device)
            
            phi_abs = torch.abs(phi)
            l1_loss = self.l1_lambda * phi_abs.mean()
            l2_loss = self.l2_lambda * (phi ** 2).mean()
            
            with torch.no_grad():
                max_val = phi_abs.max()
                threshold = max_val * 0.01 if max_val > 1e-10 else 1e-10
                current_sparsity = (phi_abs < threshold).float().mean()
            sparsity_loss = self.sparsity_lambda * (current_sparsity - self.target_sparsity) ** 2
            
            loss = coalition_loss + eff_loss + smooth_loss + l1_loss + l2_loss + sparsity_loss
        
        if not torch.isfinite(loss):
            return float('inf')
        
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss.item()
    
    def _validate(self, X_val):
        """Compute validation loss."""
        self.explainer.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_val)), batch_size=self.batch_size, shuffle=False)
        total_loss, n = 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi = self.explainer(X_batch, self.baseline)
                preds = self._get_predictions(X_batch)
                eff_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                if torch.isfinite(eff_err):
                    total_loss += eff_err.item()
                    n += 1
        
        self.explainer.train()
        return total_loss / max(n, 1) if n > 0 else float('inf')
    
    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        """Train the TDE explainer."""
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')
        
        loader = self._create_dataloader(X_train)
        if loader is None:
            return float('inf')
        
        optimizer, scheduler = self._get_optimizer()
        return self._training_loop(loader, X_val, optimizer, scheduler, self._process_batch, self._validate)
    
    def explain(self, instance):
        """Generate SHAP values for a single instance."""
        if self.explainer is None:
            raise ValueError("Explainer not trained")
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        instance = instance.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi = self.explainer(instance, self.baseline).cpu().numpy()[0]
        return phi
    
    def save(self, path, filename="tde_explainer"):
        """Save trained explainer to disk."""
        os.makedirs(path, exist_ok=True)
        
        state_dict = self.explainer._orig_mod.state_dict() if hasattr(self.explainer, '_orig_mod') else self.explainer.state_dict()
        base_pred_save = self.base_pred.cpu() if isinstance(self.base_pred, torch.Tensor) else torch.tensor(float(self.base_pred))
        
        state = {
            'explainer': state_dict, 'baseline': self.baseline.cpu(), 'base_pred': base_pred_save,
            'time_steps': self.time_steps, 'n_features': self.n_features, 'n_windows': self.n_windows,
            'feature_names': self.feature_names, 'best_loss': self.best_loss,
            'history': self.history, 'init_params': self._init_params
        }
        torch.save(state, os.path.join(path, f"{filename}.pt"))
        return os.path.join(path, f"{filename}.pt")
    
    @classmethod
    def load(cls, path, filename="tde_explainer", device_override=None):
        """Load trained explainer from disk."""
        dev = device_override or device
        state = torch.load(os.path.join(path, f"{filename}.pt"), map_location=dev, weights_only=False)
        
        params = state.get('init_params', {})
        exp = cls(**params)
        exp.device = dev
        exp.time_steps = state['time_steps']
        exp.n_features = state['n_features']
        exp.n_windows = state.get('n_windows', exp.time_steps // exp.window_size)
        exp.feature_names = state['feature_names']
        exp.baseline = state['baseline'].to(dev)
        exp.base_pred = state['base_pred'].to(dev) if isinstance(state['base_pred'], torch.Tensor) else torch.tensor(float(state['base_pred']), device=dev)
        exp.best_loss = state.get('best_loss', float('inf'))
        exp.history = state.get('history', {})
        
        exp.explainer = TemporalExplainerNetwork(
            exp.time_steps, exp.n_features, params.get('hidden_dim', 128),
            params.get('n_conv_layers', 2), params.get('kernel_size', 3),
            params.get('dropout_rate', 0.2), params.get('sparsity_threshold', 0.01),
            params.get('n_attention_heads', 4)
        ).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        _, exp._shapley_probs_features = exp._compute_shapley_kernel(exp.n_features)
        
        return exp


# ============================
# FASTSHAP TRAINER
# ============================

class FastSHAPExplainer(BaseExplainer):
    """Trainer for FastSHAP explainer."""
    
    def __init__(self, l1_lambda=0.01, efficiency_lambda=0.1, hidden_dim=256,
                 n_layers=2, dropout_rate=0.2, **kwargs):
        
        super().__init__(**kwargs)
        
        self.l1_lambda = l1_lambda
        self.efficiency_lambda = efficiency_lambda
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        self.input_dim = None
        self._shapley_probs_elements = None
        
        self._init_params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        self._init_params.update(kwargs)
    
    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        """Initialize FastSHAP network and compute baseline."""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.input_dim = self.time_steps * self.n_features
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        if gpu_model is not None:
            self._gpu_model = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        
        X_flat = X_train.reshape(len(X_train), -1)
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        baseline_3d = self.baseline.view(1, self.time_steps, self.n_features)
        base_preds = self._get_predictions(baseline_3d)
        self.base_pred = base_preds[0] if base_preds.numel() > 0 else base_preds
        if not isinstance(self.base_pred, torch.Tensor):
            self.base_pred = torch.tensor(float(self.base_pred), device=self.device)
        
        self.explainer = FastSHAPNetwork(self.input_dim, self.hidden_dim, self.n_layers, self.dropout_rate).to(self.device)
        _, self._shapley_probs_elements = self._compute_shapley_kernel(self.input_dim)
        
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.explainer = torch.compile(self.explainer, mode='reduce-overhead')
            except Exception:
                pass
    
    def _generate_element_masks(self, batch_size):
        """Generate element-wise binary masks."""
        probs = self._shapley_probs_elements
        total = batch_size * self.samples_per_feature
        d = self.input_dim
        
        k_idx = torch.multinomial(probs, total, replacement=True)
        k_samples = torch.arange(1, d, device=self.device)[k_idx]
        rand = torch.rand(total, d, device=self.device)
        sorted_idx = torch.argsort(rand, dim=1)
        masks = (sorted_idx < k_samples.unsqueeze(1)).float()
        
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks
    
    def _process_batch(self, X_batch_flat, optimizer):
        """Process single training batch."""
        batch_size = X_batch_flat.size(0)
        X_batch_flat = X_batch_flat.to(self.device, non_blocking=True)
        
        expanded = X_batch_flat.repeat(self.samples_per_feature, 1)
        masks = self._generate_element_masks(batch_size)
        total = masks.size(0)
        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat_factor, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).expand(total, -1)
        
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        preds_masked = self._get_predictions(masked)
        
        if self.paired_sampling:
            n_unique = total // 2
            preds_unique = self._get_predictions(X_paired[:n_unique])
            preds_orig = preds_unique.repeat(2)
        else:
            preds_orig = self._get_predictions(X_paired)
        
        use_amp = self.scaler is not None and self.device.type == 'cuda'
        
        with autocast(enabled=use_amp):
            phi = self.explainer(X_paired)
            
            masked_sum = (masks * phi).sum(dim=1)
            coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
            
            phi_sum = phi.sum(dim=1)
            eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
            
            l1_loss = self.l1_lambda * torch.abs(phi).mean()
            loss = coalition_loss + eff_loss + l1_loss
        
        if not torch.isfinite(loss):
            return float('inf')
        
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss.item()
    
    def _validate(self, X_val_flat):
        """Compute validation loss."""
        self.explainer.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_flat)), batch_size=self.batch_size, shuffle=False)
        total_loss, n = 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi = self.explainer(X_batch)
                preds = self._get_predictions(X_batch)
                eff_err = ((phi.sum(dim=1) - (preds - self.base_pred)) ** 2).mean()
                if torch.isfinite(eff_err):
                    total_loss += eff_err.item()
                    n += 1
        
        self.explainer.train()
        return total_loss / max(n, 1) if n > 0 else float('inf')
    
    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        """Train the FastSHAP explainer."""
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')
        
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        
        loader = self._create_dataloader(X_train_flat)
        if loader is None:
            return float('inf')
        
        optimizer, scheduler = self._get_optimizer()
        return self._training_loop(loader, X_val_flat, optimizer, scheduler, self._process_batch, self._validate)
    
    def explain(self, instance):
        """Generate SHAP values for a single instance."""
        if self.explainer is None:
            raise ValueError("Explainer not trained")
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        
        if instance.ndim == 3:
            instance = instance.reshape(instance.size(0), -1)
        elif instance.ndim == 2 and instance.size(0) == self.time_steps:
            instance = instance.reshape(1, -1)
        elif instance.ndim == 1:
            instance = instance.unsqueeze(0)
        
        instance = instance.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi_flat = self.explainer(instance).cpu().numpy()[0]
        
        return phi_flat.reshape(self.time_steps, self.n_features)
    
    def save(self, path, filename="fastshap_explainer"):
        """Save trained explainer to disk."""
        os.makedirs(path, exist_ok=True)
        
        state_dict = self.explainer._orig_mod.state_dict() if hasattr(self.explainer, '_orig_mod') else self.explainer.state_dict()
        base_pred_save = self.base_pred.cpu() if isinstance(self.base_pred, torch.Tensor) else torch.tensor(float(self.base_pred))
        
        state = {
            'explainer': state_dict, 'baseline': self.baseline.cpu(), 'base_pred': base_pred_save,
            'input_dim': self.input_dim, 'time_steps': self.time_steps, 'n_features': self.n_features,
            'feature_names': self.feature_names, 'best_loss': self.best_loss,
            'history': self.history, 'init_params': self._init_params
        }
        torch.save(state, os.path.join(path, f"{filename}.pt"))
        return os.path.join(path, f"{filename}.pt")
    
    @classmethod
    def load(cls, path, filename="fastshap_explainer", device_override=None):
        """Load trained explainer from disk."""
        dev = device_override or device
        state = torch.load(os.path.join(path, f"{filename}.pt"), map_location=dev, weights_only=False)
        
        params = state.get('init_params', {})
        exp = cls(**params)
        exp.device = dev
        exp.input_dim = state['input_dim']
        exp.time_steps = state['time_steps']
        exp.n_features = state['n_features']
        exp.feature_names = state['feature_names']
        exp.baseline = state['baseline'].to(dev)
        exp.base_pred = state['base_pred'].to(dev) if isinstance(state['base_pred'], torch.Tensor) else torch.tensor(float(state['base_pred']), device=dev)
        exp.best_loss = state.get('best_loss', float('inf'))
        exp.history = state.get('history', {})
        
        exp.explainer = FastSHAPNetwork(exp.input_dim, params.get('hidden_dim', 256), params.get('n_layers', 2), params.get('dropout_rate', 0.2)).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        _, exp._shapley_probs_elements = exp._compute_shapley_kernel(exp.input_dim)
        
        return exp


# ============================
# LOADING FUNCTIONS
# ============================

def load_tde_for_inference(primary_use, option_number, model_name, model_predict_func=None, device_override=None):
    """Load TDE explainer for inference."""
    dev = device_override or device
    path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / "tde"
    
    if not (path / "tde_explainer.pt").exists():
        raise FileNotFoundError(f"TDE explainer not found: {path}")
    
    tde = TemporalDeepExplainer.load(str(path), device_override=dev)
    if model_predict_func:
        tde.model_predict_func = model_predict_func
    return tde


def load_fastshap_for_inference(primary_use, option_number, model_name, model_predict_func=None, device_override=None):
    """Load FastSHAP explainer for inference."""
    dev = device_override or device
    path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / "fastshap"
    
    if not (path / "fastshap_explainer.pt").exists():
        raise FileNotFoundError(f"FastSHAP explainer not found: {path}")
    
    fs = FastSHAPExplainer.load(str(path), device_override=dev)
    if model_predict_func:
        fs.model_predict_func = model_predict_func
    return fs


# ============================
# TRADITIONAL SHAP METHODS
# ============================

class TraditionalSHAPMethods:
    """Wrapper for traditional SHAP methods."""
    
    def __init__(self, model, background, time_steps, n_features):
        self.device = device
        self.time_steps = time_steps
        self.n_features = n_features
        self.wrapped_model = SingleHorizonWrapper(model).to(device)
        self.wrapped_model.eval()
        self.background_tensor = torch.FloatTensor(background).to(device)
    
    def _compute_shap(self, instance, explainer_class, **kwargs):
        """Generic SHAP computation."""
        try:
            if isinstance(instance, np.ndarray):
                instance = torch.FloatTensor(instance)
            if instance.ndim == 2:
                instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            
            explainer = explainer_class(self.wrapped_model, self.background_tensor)
            shap_vals = explainer.shap_values(instance, **kwargs)
            
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            if isinstance(shap_vals, torch.Tensor):
                shap_vals = shap_vals.cpu().numpy()
            shap_vals = np.array(shap_vals)
            
            if shap_vals.ndim == 3 and shap_vals.shape[0] == 1:
                shap_vals = shap_vals[0]
            
            if shap_vals.shape == (self.time_steps, self.n_features):
                return shap_vals
            if shap_vals.size == self.time_steps * self.n_features:
                return shap_vals.reshape(self.time_steps, self.n_features)
            return None
        except Exception:
            return None
    
    def gradient_shap(self, instance):
        return self._compute_shap(instance, shap.GradientExplainer)
    
    def deep_shap(self, instance):
        return self._compute_shap(instance, shap.DeepExplainer, check_additivity=False)


# ============================
# METRICS
# ============================

class ExplainabilityMetrics:
    """Compute explainability metrics."""
    
    def __init__(self, model, baseline, base_pred, time_steps, n_features):
        self.wrapped_model = SingleHorizonWrapper(model).to(device)
        self.wrapped_model.eval()
        self.baseline = baseline
        self.base_pred = float(base_pred.cpu().numpy()) if isinstance(base_pred, torch.Tensor) else float(base_pred)
        self.time_steps = time_steps
        self.n_features = n_features
    
    def _get_prediction(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            return self.wrapped_model(x).cpu().numpy().flatten()[0]
    
    def fidelity(self, instance, shap_vals, top_k_pct=10):
        """Fidelity: prediction change when masking top-k features."""
        if shap_vals is None:
            return None
        
        instance = to_numpy(instance)
        if instance.ndim == 3:
            instance = instance[0]
        
        baseline = to_numpy(self.baseline)
        if baseline.ndim == 3:
            baseline = baseline[0]
        
        orig_pred = self._get_prediction(instance)
        abs_shap = np.abs(shap_vals)
        k = max(1, int(abs_shap.size * top_k_pct / 100))
        top_k_idx = np.argsort(abs_shap.flatten())[-k:]
        
        masked = instance.copy()
        for idx in top_k_idx:
            t, f = idx // self.n_features, idx % self.n_features
            masked[t, f] = baseline[t, f]
        
        return float(abs(orig_pred - self._get_prediction(masked)))
    
    def reliability(self, instance, shap_vals, shap_func, noise_std=NOISE_STD):
        """Reliability: stability under perturbation."""
        if shap_vals is None:
            return None, None, None
        
        instance = to_numpy(instance)
        if instance.ndim == 3:
            instance = instance[0]
        
        perturbed = np.clip(instance + np.random.normal(0, noise_std, instance.shape), 0, 1).astype(np.float32)
        shap_pert = shap_func(perturbed)
        
        if shap_pert is None:
            return None, None, None
        
        orig, pert = shap_vals.flatten(), shap_pert.flatten()
        mask = np.isfinite(orig) & np.isfinite(pert)
        
        if np.sum(mask) < 10:
            return None, None, None
        
        try:
            corr, _ = pearsonr(orig[mask], pert[mask])
            corr = float(corr) if np.isfinite(corr) else None
        except Exception:
            corr = None
        
        mse = float(np.mean((orig[mask] - pert[mask]) ** 2))
        
        k = max(1, int(len(orig) * 10 / 100))
        top_k_orig = set(np.argsort(np.abs(orig))[-k:])
        top_k_pert = set(np.argsort(np.abs(pert))[-k:])
        topk_overlap = float(len(top_k_orig & top_k_pert) / k)
        
        return corr, mse, topk_overlap
    
    def sparsity(self, shap_vals, threshold_pct=1):
        """Sparsity: fraction of near-zero values."""
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals)
        max_val = np.max(abs_shap)
        if max_val == 0:
            return 100.0
        threshold = max_val * threshold_pct / 100
        return float(np.sum(abs_shap < threshold) / abs_shap.size * 100)
    
    def complexity(self, shap_vals):
        """Complexity: entropy of SHAP distribution."""
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals).flatten() + 1e-10
        probs = abs_shap / np.sum(abs_shap)
        return float(-np.sum(probs * np.log(probs)))
    
    def efficiency_error(self, instance, shap_vals):
        """Efficiency: how well SHAP values sum to prediction difference."""
        if shap_vals is None:
            return None
        
        instance = to_numpy(instance)
        if instance.ndim == 3:
            instance = instance[0]
        
        pred = self._get_prediction(instance)
        expected = pred - self.base_pred
        actual = np.sum(shap_vals)
        
        return abs(actual - expected) / (abs(expected) + 1e-10)


# ============================
# VISUALIZATION
# ============================

def plot_convergence(history, save_path, title="Convergence"):
    """Plot training convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['lr'], 'orange', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_data, save_path, title="XAI Methods Comparison"):
    """Plot comparison of XAI methods."""
    methods = [m for m in metrics_data.keys() if metrics_data[m] and any(
        len(metrics_data[m].get(k, [])) > 0 for k in ['fidelity', 'reliability', 'sparsity']
    )]
    
    if not methods:
        return False
    
    color_map = {
        'TDE': '#2ecc71', 'FastSHAP': '#3498db',
        'Gradient_SHAP': '#e74c3c', 'Deep_SHAP': '#9b59b6'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_configs = [
        {'key': 'fidelity', 'title': 'Fidelity↑', 'row': 0, 'col': 0},
        {'key': 'sparsity', 'title': 'Sparsity↑', 'row': 0, 'col': 1},
        {'key': 'reliability', 'title': 'Reliability↑', 'row': 0, 'col': 2},
        {'key': 'topk_correlation', 'title': 'Top-K Overlap↑', 'row': 1, 'col': 0},
        {'key': 'efficiency', 'title': 'Efficiency Error↓', 'row': 1, 'col': 1},
        {'key': 'time', 'title': 'Time (ms)', 'row': 1, 'col': 2, 'scale': 1000},
    ]
    
    for config in plot_configs:
        ax = axes[config['row'], config['col']]
        key = config['key']
        scale = config.get('scale', 1)
        
        means, stds, colors, labels = [], [], [], []
        for method in methods:
            vals = [v * scale for v in metrics_data[method].get(key, []) if v is not None and np.isfinite(v)]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                colors.append(color_map.get(method, '#95a5a6'))
                labels.append(method.replace('_', '\n'))
        
        if means:
            x_pos = np.arange(len(means))
            ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=9)
        
        ax.set_title(config['title'], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


# ============================
# HYPERPARAMETER OPTIMIZATION
# ============================

def create_tde_objective(X_train, X_val, model_predict_func, feature_names, n_epochs):
    """Create Optuna objective for TDE."""
    def objective(trial):
        params = {
            'window_size': trial.suggest_categorical('window_size', [3, 6, 12, 24]),
            'l1_lambda': trial.suggest_float('l1_lambda', 0.0001, 1.0, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 0.0001, 0.1, log=True),
            'smoothness_lambda': trial.suggest_float('smoothness_lambda', 0.001, 0.3),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.5),
            'sparsity_lambda': trial.suggest_float('sparsity_lambda', 0.01, 0.5),
            'target_sparsity': trial.suggest_float('target_sparsity', 0.50, 0.80),
            'sparsity_threshold': trial.suggest_float('sparsity_threshold', 0.001, 0.05, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_conv_layers': trial.suggest_int('n_conv_layers', 1, 3),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'n_attention_heads': trial.suggest_categorical('n_attention_heads', [2, 4, 8]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'masking_mode': trial.suggest_categorical('masking_mode', ['window', 'feature']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 4),
        }
        
        try:
            tde = TemporalDeepExplainer(n_epochs=n_epochs, patience=EARLY_STOP_PATIENCE, verbose=False, paired_sampling=True, **params)
            val_loss = tde.train(X_train, X_val, model_predict_func, feature_names)
            del tde
            torch.cuda.empty_cache()
            return val_loss
        except Exception:
            return float('inf')
    return objective


def create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs):
    """Create Optuna objective for FastSHAP."""
    def objective(trial):
        params = {
            'l1_lambda': trial.suggest_float('l1_lambda', 0.001, 0.3, log=True),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.5),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 4),
        }
        
        try:
            fs = FastSHAPExplainer(n_epochs=n_epochs, patience=EARLY_STOP_PATIENCE, verbose=False, paired_sampling=True, **params)
            val_loss = fs.train(X_train, X_val, model_predict_func, feature_names)
            del fs
            torch.cuda.empty_cache()
            return val_loss
        except Exception:
            return float('inf')
    return objective


def run_optimization(explainer_type, X_train, X_val, model_predict_func, feature_names,
                     n_trials, n_epochs, primary_use, option_number, model_name):
    """Run hyperparameter optimization."""
    optuna_db = get_optuna_db_path(primary_use, option_number, model_name, explainer_type)
    storage = f"sqlite:///{optuna_db}"
    study_name = f"{explainer_type}_{model_name}"
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42),
                                study_name=study_name, storage=storage, load_if_exists=True)
    
    if explainer_type == 'tde':
        objective = create_tde_objective(X_train, X_val, model_predict_func, feature_names, n_epochs)
    else:
        objective = create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs)
    
    callback = lambda s, t: save_trial(explainer_type, primary_use, option_number, model_name, t.number, t.params, t.value, len(X_train))
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True, n_jobs=1)
    return study, time.time() - start


# ============================
# DATA LOADING
# ============================

def load_dataset(primary_use, option_number):
    """Load dataset using preprocess function."""
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path="energy_data.db", primary_use=primary_use, option_number=option_number,
        scaled=True, scale_type="both"
    )


def get_datasets():
    """Get all available datasets."""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query(
        'SELECT DISTINCT primary_use, option_number FROM prediction_performance ORDER BY primary_use, option_number', conn)
    conn.close()
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])} for _, r in df.iterrows()]


def get_models(primary_use, option_number):
    """Get available models for a dataset."""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query(
        'SELECT DISTINCT model_name FROM prediction_performance WHERE primary_use = ? AND option_number = ?',
        conn, params=(primary_use, option_number))
    conn.close()
    return df['model_name'].tolist()


# ============================
# TRAINING & COMPARISON
# ============================

def train_and_compare(primary_use, option_number, model_name, container, explainer_types_to_train,
                      n_trials, n_test_samples, logger):
    """Train explainers and compare with traditional SHAP methods."""
    logger.info(f"\n[MODEL] {model_name}")
    
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        logger.error(f"  Model not found: {model_path}")
        return None
    
    model = load_complete_model(str(model_path), device=device)
    time_steps, n_features = container.X_train.shape[1], container.X_train.shape[2]
    
    def predict_first_horizon(X):
        if X.ndim == 2:
            X = X.reshape(-1, time_steps, n_features)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        return pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()
    
    # Prepare training data
    X_all = np.concatenate([container.X_train, container.X_val], axis=0)
    frac = DEBUG_TRAINING_FRACTION if DEBUG_MODE else PROD_TRAINING_FRACTION
    n_samples = int(len(X_all) * frac)
    
    np.random.seed(42)
    X_all = X_all[np.random.choice(len(X_all), n_samples, replace=False)]
    n_val = int(len(X_all) * VALIDATION_SPLIT)
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    
    logger.info(f"  Data: Train={len(X_train)} Val={len(X_val)}")
    
    trial_epochs = DEBUG_TRIAL_EPOCHS if DEBUG_MODE else PROD_TRIAL_EPOCHS
    final_epochs = DEBUG_FINAL_EPOCHS if DEBUG_MODE else PROD_FINAL_EPOCHS
    
    explainers = {}
    
    # Train TDE
    if 'tde' in explainer_types_to_train:
        logger.info(f"  [TDE] Optimizing...")
        study, opt_time = run_optimization('tde', X_train, X_val, predict_first_horizon,
                                           container.feature_names, n_trials, trial_epochs,
                                           primary_use, option_number, model_name)
        
        logger.info(f"  [TDE] Best window_size: {study.best_params.get('window_size', 6)}")
        logger.info(f"  [TDE] Final training...")
        
        tde = TemporalDeepExplainer(n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE,
                                    verbose=True, paired_sampling=True, **study.best_params)
        start = time.time()
        final_loss = tde.train(X_train, X_val, predict_first_horizon, container.feature_names, gpu_model=model)
        train_time = time.time() - start
        
        tde_dir = model_dir / "tde"
        tde_dir.mkdir(parents=True, exist_ok=True)
        tde.save(str(tde_dir))
        
        save_explainer_metadata(primary_use, option_number, model_name, 'TDE', study.best_params,
                                study.best_value, final_loss, len(X_train), time_steps, n_features,
                                opt_time, train_time, n_trials, str(tde_dir), container.feature_names)
        plot_convergence(tde.history, plots_dir / "tde_convergence.png", "TDE Convergence")
        explainers['TDE'] = tde
        logger.info(f"  [TDE] Final Loss: {final_loss:.6f}")
    
    # Train FastSHAP
    if 'fastshap' in explainer_types_to_train:
        logger.info(f"  [FastSHAP] Optimizing...")
        study, opt_time = run_optimization('fastshap', X_train, X_val, predict_first_horizon,
                                           container.feature_names, n_trials, trial_epochs,
                                           primary_use, option_number, model_name)
        
        logger.info(f"  [FastSHAP] Final training...")
        fs = FastSHAPExplainer(n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE,
                               verbose=True, paired_sampling=True, **study.best_params)
        start = time.time()
        final_loss = fs.train(X_train, X_val, predict_first_horizon, container.feature_names)
        train_time = time.time() - start
        
        fs_dir = model_dir / "fastshap"
        fs_dir.mkdir(parents=True, exist_ok=True)
        fs.save(str(fs_dir))
        
        save_explainer_metadata(primary_use, option_number, model_name, 'FastSHAP', study.best_params,
                                study.best_value, final_loss, len(X_train), time_steps, n_features,
                                opt_time, train_time, n_trials, str(fs_dir), container.feature_names)
        plot_convergence(fs.history, plots_dir / "fastshap_convergence.png", "FastSHAP Convergence")
        explainers['FastSHAP'] = fs
        logger.info(f"  [FastSHAP] Final Loss: {final_loss:.6f}")
    
    # Load existing explainers
    for name, loader in [('TDE', load_tde_for_inference), ('FastSHAP', load_fastshap_for_inference)]:
        if name not in explainers:
            try:
                explainers[name] = loader(primary_use, option_number, model_name, predict_first_horizon)
            except FileNotFoundError:
                pass
    
    # Comparison
    logger.info(f"\n  [COMPARE] Evaluating on {n_test_samples} test samples...")
    
    bg_idx = np.arange(max(0, len(X_train) - min(50, len(X_train))), len(X_train))
    background = X_train[bg_idx]
    
    trad = TraditionalSHAPMethods(model, background, time_steps, n_features)
    all_methods = {'Gradient_SHAP': trad.gradient_shap, 'Deep_SHAP': trad.deep_shap}
    for name, exp in explainers.items():
        all_methods[name] = exp.explain
    
    # Get baseline for metrics
    if 'TDE' in explainers:
        baseline_np = to_numpy(explainers['TDE'].baseline)
        base_pred = explainers['TDE'].base_pred
    elif 'FastSHAP' in explainers:
        baseline_np = to_numpy(explainers['FastSHAP'].baseline).reshape(time_steps, n_features)
        base_pred = explainers['FastSHAP'].base_pred
    else:
        baseline_np = np.median(X_train, axis=0)
        base_pred = predict_first_horizon(baseline_np[np.newaxis])[0]
    
    metrics = ExplainabilityMetrics(model, baseline_np, base_pred, time_steps, n_features)
    X_test = container.X_test[:n_test_samples]
    
    logger.info(f"\n  {'Method':<14} {'Fidelity':>10} {'Rel.Corr':>10} {'Sparsity':>10} {'Efficiency':>12} {'Time(ms)':>10}")
    logger.info(f"  {'-'*70}")
    
    all_metrics = {m: {'fidelity': [], 'reliability': [], 'reliability_mse': [],
                       'topk_correlation': [], 'sparsity': [], 'efficiency': [], 'time': []}
                   for m in all_methods}
    all_results = {m: [] for m in all_methods}
    
    for idx in range(len(X_test)):
        sample = X_test[idx]
        for method, func in all_methods.items():
            start = time.time()
            try:
                shap_vals = func(sample)
            except Exception:
                continue
            comp_time = time.time() - start
            
            if shap_vals is None or shap_vals.shape != (time_steps, n_features):
                continue
            
            fid = metrics.fidelity(sample, shap_vals)
            rel_corr, rel_mse, topk_overlap = metrics.reliability(sample, shap_vals, func)
            spa = metrics.sparsity(shap_vals)
            com = metrics.complexity(shap_vals)
            eff = metrics.efficiency_error(sample, shap_vals)
            
            all_results[method].append({
                'fidelity': fid, 'reliability': rel_corr, 'reliability_mse': rel_mse,
                'topk_correlation': topk_overlap, 'sparsity': spa, 'efficiency': eff, 'time': comp_time
            })
            
            for key in ['fidelity', 'reliability', 'reliability_mse', 'topk_correlation', 'sparsity', 'efficiency', 'time']:
                all_metrics[method][key].append(all_results[method][-1].get(key))
            
            save_comparison(primary_use, option_number, model_name, idx, method,
                           fid, rel_corr, rel_mse, spa, com, eff, comp_time)
    
    # Print summary
    for method, results in all_results.items():
        if results:
            avg = lambda k: np.nanmean([r[k] for r in results if r[k] is not None])
            logger.info(f"  {method:<14} {avg('fidelity'):>10.4f} {avg('reliability'):>10.4f} "
                       f"{avg('sparsity'):>9.1f}% {avg('efficiency'):>12.4f} {avg('time')*1000:>9.2f}")
    
    # Save comparison plot
    if any(all_metrics[m]['fidelity'] for m in all_metrics):
        plot_metrics_comparison(all_metrics, plots_dir / "xai_comparison.png", f"{model_name}: XAI Comparison")
    
    del model
    torch.cuda.empty_cache()
    return all_results


# ============================
# STATUS CHECKING
# ============================

def get_incomplete_items(primary_use, option_number, models, explainer_types):
    """Check what's incomplete and return items needing processing."""
    items_to_process = {}
    status_info = {'complete': [], 'incomplete': []}
    
    for model_name in models:
        items_to_process[model_name] = []
        
        for exp_type in explainer_types:
            # Check database
            db_exists = False
            try:
                conn = sqlite3.connect(EXPLAINER_DB)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM explainer_metadata 
                    WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?
                ''', (primary_use, option_number, model_name, exp_type.upper()))
                db_exists = cursor.fetchone()[0] > 0
                conn.close()
            except:
                pass
            
            # Check disk
            exp_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / exp_type.lower()
            disk_exists = (exp_dir / f"{exp_type.lower()}_explainer.pt").exists()
            
            # Check comparison
            comp_exists = False
            try:
                conn = sqlite3.connect(EXPLAINER_DB)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM comparison_results 
                    WHERE primary_use=? AND option_number=? AND model_name=? AND method=?
                ''', (primary_use, option_number, model_name, exp_type.upper()))
                comp_exists = cursor.fetchone()[0] > 0
                conn.close()
            except:
                pass
            
            is_complete = db_exists and disk_exists and comp_exists
            
            if not is_complete:
                items_to_process[model_name].append(exp_type)
                status_info['incomplete'].append(f"{model_name}/{exp_type.upper()}")
            else:
                status_info['complete'].append(f"{model_name}/{exp_type.upper()}")
    
    return items_to_process, status_info


def show_progress_table(primary_use, option_number, models, explainer_types):
    """Show compact progress table."""
    print("\n" + "=" * 60)
    print("📊 TRAINING PROGRESS")
    print("=" * 60)
    
    header = f"{'Model':<12}"
    for exp_type in explainer_types:
        header += f" {exp_type.upper():<12}"
    print(header)
    print("-" * 60)
    
    for model_name in models:
        row = f"{model_name:<12}"
        for exp_type in explainer_types:
            exp_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / exp_type.lower()
            disk_exists = (exp_dir / f"{exp_type.lower()}_explainer.pt").exists()
            
            comp_exists = False
            try:
                conn = sqlite3.connect(EXPLAINER_DB)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM comparison_results 
                    WHERE primary_use=? AND option_number=? AND model_name=? AND method=?
                ''', (primary_use, option_number, model_name, exp_type.upper()))
                comp_exists = cursor.fetchone()[0] > 0
                conn.close()
            except:
                pass
            
            if disk_exists and comp_exists:
                row += f" ✅Done      "
            elif disk_exists:
                row += f" 🔶NoCmp     "
            else:
                row += f" ❌Missing   "
        print(row)
    
    print("=" * 60)


def show_comparison_table(primary_use, option_number, models):
    """Show comparison table for each model."""
    print("\n" + "=" * 100)
    print("📊 XAI METHODS COMPARISON")
    print("=" * 100)
    
    for model_name in models:
        try:
            conn = sqlite3.connect(EXPLAINER_DB)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT method, AVG(fidelity), AVG(sparsity), AVG(reliability_correlation),
                       AVG(efficiency_error), AVG(computation_time)
                FROM comparison_results
                WHERE primary_use=? AND option_number=? AND model_name=?
                GROUP BY method ORDER BY method
            ''', (primary_use, option_number, model_name))
            rows = cursor.fetchall()
            conn.close()
        except:
            continue
        
        if not rows:
            print(f"\n{model_name}: No data")
            continue
        
        print(f"\n┌{'─'*90}┐")
        print(f"│ {model_name:<88} │")
        print(f"├{'─'*90}┤")
        print(f"│ {'Method':<16} {'Fidelity↑':>14} {'Sparsity↑':>14} {'Reliab↑':>14} {'Effic↓':>14} {'Time(ms)':>12} │")
        print(f"├{'─'*90}┤")
        
        for row in rows:
            method, fid, spa, rel, eff, tm = row
            fid_s = f"{fid:.4f}" if fid else "N/A"
            spa_s = f"{spa:.1f}%" if spa else "N/A"
            rel_s = f"{rel:.4f}" if rel else "N/A"
            eff_s = f"{eff:.4f}" if eff else "N/A"
            tm_s = f"{tm*1000:.2f}" if tm else "N/A"
            print(f"│ {method:<16} {fid_s:>14} {spa_s:>14} {rel_s:>14} {eff_s:>14} {tm_s:>12} │")
        
        print(f"└{'─'*90}┘")
    
    print("\n" + "=" * 100)


# ============================
# USER INPUT
# ============================

def get_user_inputs():
    """Get user configuration through interactive prompts."""
    print("\n" + "=" * 60)
    print("🚀 TDE & FastSHAP Training System v6.0")
    print("=" * 60)
    
    datasets = get_datasets()
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    # Select primary use
    uses = sorted(set(d['primary_use'] for d in datasets))
    print(f"\n📁 Primary Uses:")
    for i, u in enumerate(uses):
        print(f"  {i}: {u}")
    
    use_input = input(f"\n👉 Select [0-{len(uses)-1}] [default: 0]: ").strip()
    selected_use = uses[int(use_input)] if use_input.isdigit() and int(use_input) < len(uses) else uses[0]
    
    # Get option
    options = [d['option_number'] for d in datasets if d['primary_use'] == selected_use]
    if len(options) > 1:
        print(f"\n📋 Options: {options}")
        opt_input = input(f"👉 Select option [default: {options[0]}]: ").strip()
        option_number = int(opt_input) if opt_input.isdigit() and int(opt_input) in options else options[0]
    else:
        option_number = options[0]
    
    # Get models
    models = get_models(selected_use, option_number)
    print(f"\n🤖 Models: {models}")
    model_input = input("👉 Select (comma-sep) or 'all' [default: all]: ").strip().lower()
    
    if model_input == '' or model_input == 'all':
        selected_models = models
    else:
        selected_models = [m.strip().upper() for m in model_input.split(',') if m.strip().upper() in models]
        if not selected_models:
            selected_models = models
    
    # Explainer type
    print("\n🔬 Explainers: 0=TDE, 1=FastSHAP, 2=Both")
    exp_input = input("👉 Select [default: 2]: ").strip()
    exp_choice = int(exp_input) if exp_input.isdigit() and int(exp_input) in [0, 1, 2] else 2
    explainer_types = ['tde'] if exp_choice == 0 else (['fastshap'] if exp_choice == 1 else ['tde', 'fastshap'])
    
    # Trials and samples
    n_trials = int(input(f"🎯 Trials [{DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS}]: ").strip() or (DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS))
    n_test_samples = int(input("🧪 Test samples [5]: ").strip() or 5)
    
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION")
    print("=" * 60)
    print(f"  Dataset: {selected_use} - Option {option_number}")
    print(f"  Models: {selected_models}")
    print(f"  Explainers: {[e.upper() for e in explainer_types]}")
    print(f"  Trials: {n_trials}, Test samples: {n_test_samples}")
    print("=" * 60)
    
    return {
        'primary_use': selected_use,
        'option_number': option_number,
        'models': selected_models,
        'explainer_types': explainer_types,
        'n_trials': n_trials,
        'n_test_samples': n_test_samples
    }


# ============================
# MAIN FUNCTION
# ============================

def main():
    """Main entry point."""
    init_database()
    
    config = get_user_inputs()
    if config is None:
        return
    
    primary_use = config['primary_use']
    option_number = config['option_number']
    models = config['models']
    explainer_types = config['explainer_types']
    
    # Check incomplete items
    items_to_process, status_info = get_incomplete_items(primary_use, option_number, models, explainer_types)
    total_incomplete = sum(len(v) for v in items_to_process.values())
    
    print("\n" + "=" * 60)
    if total_incomplete == 0:
        print("✅ All explainers complete!")
    else:
        print(f"🔧 Found {total_incomplete} incomplete: {status_info['incomplete']}")
    print("=" * 60)
    
    # Train missing items
    if total_incomplete > 0:
        try:
            container = load_dataset(primary_use, option_number)
            print(f"📦 Dataset: {container.X_train.shape[0]} train samples")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return
        
        for model_name in models:
            exp_types = items_to_process.get(model_name, [])
            if not exp_types:
                continue
            
            print(f"\n🔄 Training {model_name}: {[e.upper() for e in exp_types]}")
            
            for exp_type in exp_types:
                delete_existing_results(primary_use, option_number, model_name, exp_type)
            
            model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = model_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            logger = setup_logger(str(log_path))
            
            try:
                train_and_compare(primary_use, option_number, model_name, container,
                                 exp_types, config['n_trials'], config['n_test_samples'], logger)
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)
            
            show_progress_table(primary_use, option_number, models, explainer_types)
    
    # Show comparison
    show_comparison_table(primary_use, option_number, models)
    
    # Ask for retrain
    retrain = input("\n👉 Retrain any models? (model names or 'no') [no]: ").strip().lower()
    if retrain and retrain != 'no':
        models_to_retrain = [m.strip().upper() for m in retrain.split(',') if m.strip().upper() in models]
        
        if models_to_retrain:
            if 'container' not in locals():
                container = load_dataset(primary_use, option_number)
            
            for model_name in models_to_retrain:
                print(f"\n🔄 Retraining {model_name}...")
                for exp_type in explainer_types:
                    delete_existing_results(primary_use, option_number, model_name, exp_type)
                
                model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
                log_path = model_dir / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                logger = setup_logger(str(log_path))
                
                try:
                    train_and_compare(primary_use, option_number, model_name, container,
                                     explainer_types, config['n_trials'], config['n_test_samples'], logger)
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                for h in logger.handlers[:]:
                    h.close()
                    logger.removeHandler(h)
            
            show_comparison_table(primary_use, option_number, models)
    
    print("\n✅ COMPLETE")


if __name__ == "__main__":
    main()