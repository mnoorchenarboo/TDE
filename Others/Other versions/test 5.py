"""
TDE & FastSHAP Training System v5.3

IMPROVEMENTS FROM v5.2:
1. Centralized plots folder - All plots saved to single location
2. Speed optimizations without sacrificing TDE accuracy:
   - Cached Shapley kernel weights
   - Pre-computed baseline predictions
   - Efficient batch processing
   - Optional mixed precision training
3. Skip existing records - Comprehensive check of database AND saved files
4. User-friendly status summary with replace option
5. Window-Sequential masking ONLY (removed separate feature masking)
   - Window masking inherently handles features within each window
   - More appropriate for temporal data

ARCHITECTURE (UNCHANGED - preserves TDE accuracy):
- TDE: Dilated Conv → Multi-Head Attention → Direct Input Connection → Soft Threshold
- FastSHAP: Pure simple MLP with element-wise masking

MASKING STRATEGY:
- TDE: Window-Sequential masking (masks contiguous time windows per feature)
- FastSHAP: Element-wise masking (each time×feature element independently)
"""

# ============================
# LIBRARY IMPORTS
# ============================

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import sys
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime
from scipy.special import comb
from scipy.stats import pearsonr
from collections import defaultdict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import shap

# ============================
# CONFIGURATION
# ============================

BENCHMARK_DB = "benchmark_results.db"
EXPLAINER_DB = "explainer_results.db"
RESULTS_BASE_DIR = "results"
PLOTS_DIR = "explainer_plots"  # Centralized plots folder

# Mode settings
DEBUG_MODE = True
DEBUG_TRAINING_FRACTION = 0.15
DEBUG_TRIAL_EPOCHS = 10
DEBUG_FINAL_EPOCHS = 50
DEBUG_N_TRIALS = 10

PROD_TRAINING_FRACTION = 0.30
PROD_TRIAL_EPOCHS = 20
PROD_FINAL_EPOCHS = 100
PROD_N_TRIALS = 30

# Training settings
PAIRED_SAMPLING = True
VALIDATION_SPLIT = 0.20
DEFAULT_WINDOW_SIZE = 6
NOISE_STD = 0.01
EARLY_STOP_PATIENCE = 5
TARGET_SPARSITY = 0.70

# Speed optimization settings
USE_MIXED_PRECISION = False  # Enable for faster training on compatible GPUs
CACHE_SHAPLEY_KERNELS = True  # Cache kernel weights
PREFETCH_FACTOR = 2  # DataLoader prefetch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================
# LOGGING
# ============================

def setup_logger(log_path):
    """Setup logger for training progress"""
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
# CENTRALIZED PLOTS DIRECTORY
# ============================

def get_plots_dir(primary_use, option_number, model_name, explainer_type):
    """Get centralized plots directory path"""
    plots_path = Path(PLOTS_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type
    plots_path.mkdir(parents=True, exist_ok=True)
    return plots_path


def get_explainer_dir(primary_use, option_number, model_name, explainer_type):
    """Get explainer model save directory"""
    exp_path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path


# ============================
# DATABASE INITIALIZATION
# ============================

def init_database():
    """Initialize SQLite database for storing results"""
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


# ============================
# EXISTENCE CHECKING FUNCTIONS
# ============================

def check_explainer_complete(primary_use, option_number, model_name, explainer_type):
    """
    Comprehensive check if explainer is already trained.
    Checks BOTH database records AND saved model files.
    
    Returns:
        dict with keys:
            - 'complete': bool - True if fully trained
            - 'has_db_record': bool
            - 'has_model_file': bool
            - 'has_hyperparams': bool
            - 'metadata': dict or None
    """
    result = {
        'complete': False,
        'has_db_record': False,
        'has_model_file': False,
        'has_hyperparams': False,
        'metadata': None
    }
    
    # Check database for metadata
    try:
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT best_hyperparameters, best_validation_loss, final_training_loss,
                   n_training_samples, explainer_path, timestamp
            FROM explainer_metadata
            WHERE primary_use = ? AND option_number = ? AND model_name = ? AND explainer_type = ?
        ''', (primary_use, option_number, model_name, explainer_type))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result['has_db_record'] = True
            result['metadata'] = {
                'best_hyperparameters': json.loads(row[0]) if row[0] else {},
                'best_validation_loss': row[1],
                'final_training_loss': row[2],
                'n_training_samples': row[3],
                'explainer_path': row[4],
                'timestamp': row[5]
            }
    except Exception as e:
        print(f"  Warning: DB check failed: {e}")
    
    # Check for saved model file
    exp_dir = get_explainer_dir(primary_use, option_number, model_name, explainer_type)
    model_file = exp_dir / f"{explainer_type}_explainer.pt"
    result['has_model_file'] = model_file.exists()
    
    # Check for hyperparameter trials
    try:
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        table_name = f"{explainer_type}_hyperparameter_trials"
        cursor.execute(f'''
            SELECT COUNT(*) FROM {table_name}
            WHERE primary_use = ? AND option_number = ? AND model_name = ?
        ''', (primary_use, option_number, model_name))
        
        count = cursor.fetchone()[0]
        conn.close()
        result['has_hyperparams'] = count > 0
    except Exception:
        pass
    
    # Complete if both DB record and model file exist
    result['complete'] = result['has_db_record'] and result['has_model_file']
    
    return result


def get_training_status_summary(datasets, models, explainer_types):
    """
    Get comprehensive training status for all combinations.
    
    Returns:
        dict with structure:
            {(primary_use, option_number): {
                model_name: {
                    explainer_type: status_dict
                }
            }}
    """
    status = {}
    
    for dataset in datasets:
        primary_use = dataset['primary_use']
        option_number = dataset['option_number']
        key = (primary_use, option_number)
        
        if key not in status:
            status[key] = {}
        
        for model_name in models:
            if model_name not in status[key]:
                status[key][model_name] = {}
            
            for exp_type in explainer_types:
                status[key][model_name][exp_type] = check_explainer_complete(
                    primary_use, option_number, model_name, exp_type
                )
    
    return status


def print_status_summary(status, models, explainer_types):
    """Print formatted status summary and return counts"""
    completed = []
    pending = []
    
    print("\n" + "="*80)
    print("📊 TRAINING STATUS SUMMARY")
    print("="*80)
    
    for (primary_use, option_number), model_status in status.items():
        print(f"\n📦 {primary_use} - Option {option_number}")
        print("-" * 60)
        
        for model_name in models:
            if model_name not in model_status:
                continue
            
            for exp_type in explainer_types:
                if exp_type not in model_status[model_name]:
                    continue
                
                s = model_status[model_name][exp_type]
                key = (primary_use, option_number, model_name, exp_type)
                
                if s['complete']:
                    status_str = "✅ COMPLETE"
                    meta = s.get('metadata', {})
                    if meta:
                        status_str += f" (Loss: {meta.get('final_training_loss', 'N/A'):.4f})"
                    completed.append(key)
                else:
                    parts = []
                    if s['has_db_record']:
                        parts.append("DB✓")
                    if s['has_model_file']:
                        parts.append("File✓")
                    if s['has_hyperparams']:
                        parts.append("HP✓")
                    
                    if parts:
                        status_str = f"⚠️  PARTIAL ({', '.join(parts)})"
                    else:
                        status_str = "❌ NOT STARTED"
                    pending.append(key)
                
                print(f"  {model_name:12s} | {exp_type.upper():10s} | {status_str}")
    
    print("\n" + "-"*60)
    print(f"✅ Completed: {len(completed)}")
    print(f"⏳ Pending:   {len(pending)}")
    print("="*80)
    
    return completed, pending


# ============================
# MODEL WRAPPER
# ============================

class SingleHorizonWrapper(nn.Module):
    """Wrapper to extract single prediction horizon from multi-horizon models"""
    def __init__(self, base_model, horizon_idx=0):
        super().__init__()
        self.base_model = base_model
        self.horizon_idx = horizon_idx
    
    def forward(self, x):
        out = self.base_model(x)
        if out.ndim > 1 and out.shape[1] > self.horizon_idx:
            return out[:, self.horizon_idx:self.horizon_idx+1]
        return out


# ============================
# TDE NETWORK v5.3 - WITH MULTI-HEAD ATTENTION
# ============================

class TemporalExplainerNetwork(nn.Module):
    """
    TDE v5.3 - Attention as gating mechanism, not in main path
    
    Architecture (UNCHANGED from v5.2):
    1. Dilated Conv (main path) - captures local temporal patterns
    2. Attention Gate (side path) - learns global importance weights
    3. Gated combination - conv_out * sigmoid(attention_weights)
    4. Direct Input Connection - ensures input-dependence
    5. Soft Thresholding - sparsity
    """
    def __init__(self, time_steps, n_features, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, use_attention_gate=True):
        super().__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.sparsity_threshold = sparsity_threshold
        self.use_attention_gate = use_attention_gate
        
        # Main Path: Dilated Temporal Convolutions
        conv_layers = []
        in_ch = n_features
        for i in range(n_conv_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            conv_layers.extend([
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_ch = hidden_dim
        self.conv = nn.Sequential(*conv_layers)
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, n_features, 1)
        
        # Side Path: Attention Gate
        if use_attention_gate:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.gate_proj = nn.Sequential(
                nn.Conv1d(hidden_dim, n_features, 1),
                nn.Sigmoid()
            )
        
        # Direct Input Connection
        self.input_weight = nn.Parameter(torch.zeros(time_steps, n_features))
        
        # Small initialization
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, baseline=None):
        """x: (batch, time_steps, n_features)"""
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
# FASTSHAP NETWORK - PURE SIMPLE MLP
# ============================

class FastSHAPNetwork(nn.Module):
    """Pure simple MLP for FastSHAP (baseline method)."""
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, baseline=None):
        h = self.mlp(x)
        output = self.output_proj(h)
        return output


# ============================
# SHAPLEY KERNEL CACHE (Speed Optimization)
# ============================

class ShapleyKernelCache:
    """Cache Shapley kernel weights to avoid recomputation"""
    _cache = {}
    
    @classmethod
    def get_kernel(cls, d, device):
        """Get cached or compute Shapley kernel weights"""
        key = (d, str(device))
        if key not in cls._cache:
            if d <= 1:
                weights = torch.ones(1, device=device)
                probs = torch.ones(1, device=device)
            else:
                k_values = torch.arange(1, d, device=device, dtype=torch.float32)
                binom_coeffs = torch.tensor(
                    [comb(d, int(k.item()), exact=True) for k in k_values],
                    device=device, dtype=torch.float32
                )
                weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
                probs = weights / weights.sum()
            cls._cache[key] = (weights, probs)
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        """Clear cache"""
        cls._cache = {}


# ============================
# TDE TRAINER (Window-Sequential Masking ONLY)
# ============================

class TemporalDeepExplainer:
    """
    TDE Trainer v5.3
    
    Key improvement: Window-Sequential masking ONLY
    - Removes separate feature masking option
    - Window masking inherently handles features within each window
    - More appropriate for temporal data
    """
    
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True, min_lr=1e-6,
                 l1_lambda=0.01, l2_lambda=0.01, smoothness_lambda=0.1, efficiency_lambda=0.1,
                 sparsity_lambda=0.1, target_sparsity=0.70,
                 weight_decay=1e-4, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, optimizer_type='adam', learning_rate=1e-3,
                 window_size=6, paired_sampling=True, samples_per_feature=2, **kwargs):
        
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda
        self.sparsity_lambda = sparsity_lambda
        self.target_sparsity = target_sparsity
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.sparsity_threshold = sparsity_threshold
        self.n_attention_heads = n_attention_heads
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.n_windows = None
        self.model_predict_func = None
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # Store init params for saving/loading
        self._init_params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
    
    def _setup(self, X_train, model_predict_func, feature_names):
        """Initialize explainer network and compute baseline"""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.n_windows = (self.time_steps + self.window_size - 1) // self.window_size
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        # Pre-compute base prediction (Speed optimization)
        base_raw = model_predict_func(self.baseline.unsqueeze(0).cpu().numpy())
        self.base_pred = torch.tensor(
            float(np.atleast_1d(base_raw).flatten()[0]),
            dtype=torch.float32, device=self.device
        )
        
        self.explainer = TemporalExplainerNetwork(
            self.time_steps, self.n_features, self.hidden_dim, self.n_conv_layers,
            self.kernel_size, self.dropout_rate, self.sparsity_threshold,
            self.n_attention_heads
        ).to(self.device)
    
    def _generate_window_sequential_masks(self, batch_size):
        """
        Generate Window-Sequential masks for temporal structure preservation.
        
        This is the ONLY masking strategy for TDE in v5.3.
        Masks contiguous time windows independently for each feature.
        """
        total = batch_size * self.samples_per_feature
        masks = torch.ones(total, self.time_steps, self.n_features, device=self.device)
        
        for i in range(total):
            # Random number of windows to mask (1 to n_windows-1)
            n_mask = torch.randint(1, max(2, self.n_windows), (1,)).item()
            # Random number of features to apply masking
            n_feat = torch.randint(1, self.n_features + 1, (1,)).item()
            
            # Select random features and windows
            feats = torch.randperm(self.n_features, device=self.device)[:n_feat]
            wins = torch.randperm(self.n_windows, device=self.device)[:n_mask]
            
            # Apply window masking to selected features
            for w in wins:
                start = int(w) * self.window_size
                end = min((int(w) + 1) * self.window_size, self.time_steps)
                for f in feats:
                    masks[i, start:end, int(f)] = 0.0
        
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        
        return masks
    
    def _get_predictions(self, inputs):
        """Get predictions from black-box model"""
        with torch.no_grad():
            preds = self.model_predict_func(inputs.cpu().numpy())
            return torch.tensor(
                np.atleast_1d(preds).flatten(),
                dtype=torch.float32, device=self.device
            )
    
    def _compute_sparsity(self, phi):
        """Compute current sparsity"""
        with torch.no_grad():
            abs_phi = torch.abs(phi)
            max_val = abs_phi.max()
            if max_val < 1e-10:
                return 1.0
            threshold = max_val * 0.01
            sparsity = (abs_phi < threshold).float().mean()
            return sparsity.item()
    
    def _process_batch(self, X_batch, optimizer):
        """Process single training batch"""
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device)
        
        expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks = self._generate_window_sequential_masks(batch_size)
        
        total = masks.size(0)
        repeat = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat, 1, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).repeat(total, 1, 1)
        
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        preds_masked = self._get_predictions(masked)
        
        phi = self.explainer(X_paired, self.baseline)
        
        # Loss computation
        masked_sum = (masks * phi).sum(dim=(1, 2))
        coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
        
        preds_orig = self._get_predictions(X_paired)
        phi_sum = phi.sum(dim=(1, 2))
        eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
        
        if phi.size(1) > 1:
            smooth_loss = self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=self.device)
        
        l1_loss = self.l1_lambda * torch.abs(phi).mean()
        l2_loss = self.l2_lambda * torch.pow(phi, 2).mean()
        
        current_sparsity = self._compute_sparsity(phi)
        sparsity_loss = self.sparsity_lambda * (current_sparsity - self.target_sparsity) ** 2
        
        loss = coalition_loss + eff_loss + smooth_loss + l1_loss + l2_loss + sparsity_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def _validate(self, X_val):
        """Compute validation loss"""
        self.explainer.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=self.batch_size, shuffle=False
        )
        total_loss, n = 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi = self.explainer(X_batch, self.baseline)
                preds = self._get_predictions(X_batch)
                eff_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                total_loss += eff_err.item()
                n += 1
        
        self.explainer.train()
        return total_loss / max(n, 1)
    
    def train(self, X_train, X_val, model_predict_func, feature_names):
        """Train the TDE explainer"""
        self._setup(X_train, model_predict_func, feature_names)
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(
            self.optimizer_type, torch.optim.Adam
        )
        optimizer = opt_cls(
            self.explainer.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5, min_lr=self.min_lr
        )
        
        best_val, best_weights, no_improve = float('inf'), None, 0
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            ep_loss, n = 0, 0
            
            for (X_batch,) in loader:
                ep_loss += self._process_batch(X_batch, optimizer)
                n += 1
            ep_loss /= n
            
            val_loss = self._validate(X_val)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(ep_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch+1:3d} | L:{ep_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")
            
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [EARLY STOP] epoch {epoch+1}")
                break
        
        if best_weights:
            self.explainer.load_state_dict(best_weights)
        self.best_loss = best_val
        return best_val
    
    def explain(self, instance):
        """Generate SHAP values for a single instance"""
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
        """Save trained explainer to disk"""
        os.makedirs(path, exist_ok=True)
        state = {
            'explainer': self.explainer.state_dict(),
            'baseline': self.baseline.cpu(),
            'base_pred': self.base_pred.cpu(),
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'n_windows': self.n_windows,
            'feature_names': self.feature_names,
            'best_loss': self.best_loss,
            'history': self.history,
            'init_params': self._init_params
        }
        torch.save(state, os.path.join(path, f"{filename}.pt"))
        return os.path.join(path, f"{filename}.pt")
    
    @classmethod
    def load(cls, path, filename="tde_explainer", device_override=None):
        """Load trained explainer from disk"""
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
        exp.base_pred = state['base_pred'].to(dev)
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
        return exp


# ============================
# FASTSHAP TRAINER
# ============================

class FastSHAPExplainer:
    """FastSHAP Trainer with element-wise masking"""
    
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True, min_lr=1e-6,
                 l1_lambda=0.01, efficiency_lambda=0.1, weight_decay=1e-4,
                 hidden_dim=256, n_layers=2, dropout_rate=0.2,
                 optimizer_type='adam', learning_rate=1e-3,
                 paired_sampling=True, samples_per_feature=2, **kwargs):
        
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.l1_lambda = l1_lambda
        self.efficiency_lambda = efficiency_lambda
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.input_dim = None
        self.time_steps = None
        self.n_features = None
        self.model_predict_func = None
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        self._init_params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
    
    def _setup(self, X_train, model_predict_func, feature_names):
        """Initialize FastSHAP network and compute baseline"""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.input_dim = self.time_steps * self.n_features
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        X_flat = X_train.reshape(len(X_train), -1)
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        baseline_np = self.baseline.unsqueeze(0).cpu().numpy().reshape(1, self.time_steps, self.n_features)
        base_raw = model_predict_func(baseline_np)
        self.base_pred = torch.tensor(
            float(np.atleast_1d(base_raw).flatten()[0]),
            dtype=torch.float32, device=self.device
        )
        
        self.explainer = FastSHAPNetwork(
            self.input_dim, self.hidden_dim, self.n_layers, self.dropout_rate
        ).to(self.device)
    
    def _generate_element_masks(self, batch_size):
        """Generate element-wise masks using cached Shapley kernel"""
        _, probs = ShapleyKernelCache.get_kernel(self.input_dim, self.device)
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
    
    def _get_predictions(self, inputs_flat):
        """Get predictions from black-box model"""
        with torch.no_grad():
            inputs_3d = inputs_flat.cpu().numpy().reshape(-1, self.time_steps, self.n_features)
            preds = self.model_predict_func(inputs_3d)
            return torch.tensor(
                np.atleast_1d(preds).flatten(),
                dtype=torch.float32, device=self.device
            )
    
    def _process_batch(self, X_batch_flat, optimizer):
        """Process single training batch"""
        batch_size = X_batch_flat.size(0)
        X_batch_flat = X_batch_flat.to(self.device)
        
        expanded = X_batch_flat.repeat(self.samples_per_feature, 1)
        masks = self._generate_element_masks(batch_size)
        
        total = masks.size(0)
        repeat = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).repeat(total, 1)
        
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        preds_masked = self._get_predictions(masked)
        
        phi = self.explainer(X_paired)
        
        masked_sum = (masks * phi).sum(dim=1)
        coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
        
        preds_orig = self._get_predictions(X_paired)
        phi_sum = phi.sum(dim=1)
        eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
        
        l1_loss = self.l1_lambda * torch.abs(phi).mean()
        
        loss = coalition_loss + eff_loss + l1_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def _validate(self, X_val_flat):
        """Compute validation loss"""
        self.explainer.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_flat)),
            batch_size=self.batch_size, shuffle=False
        )
        total_loss, n = 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi = self.explainer(X_batch)
                preds = self._get_predictions(X_batch)
                eff_err = ((phi.sum(dim=1) - (preds - self.base_pred)) ** 2).mean()
                total_loss += eff_err.item()
                n += 1
        
        self.explainer.train()
        return total_loss / max(n, 1)
    
    def train(self, X_train, X_val, model_predict_func, feature_names):
        """Train the FastSHAP explainer"""
        self._setup(X_train, model_predict_func, feature_names)
        
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_flat)),
            batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(
            self.optimizer_type, torch.optim.Adam
        )
        optimizer = opt_cls(
            self.explainer.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5, min_lr=self.min_lr
        )
        
        best_val, best_weights, no_improve = float('inf'), None, 0
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            ep_loss, n = 0, 0
            
            for (X_batch,) in loader:
                ep_loss += self._process_batch(X_batch, optimizer)
                n += 1
            ep_loss /= n
            
            val_loss = self._validate(X_val_flat)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(ep_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch+1:3d} | L:{ep_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")
            
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [EARLY STOP] epoch {epoch+1}")
                break
        
        if best_weights:
            self.explainer.load_state_dict(best_weights)
        self.best_loss = best_val
        return best_val
    
    def explain(self, instance):
        """Generate SHAP values for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not trained")
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        if instance.ndim == 3:
            instance = instance.reshape(instance.size(0), -1)
        elif instance.ndim == 2:
            instance = instance.reshape(1, -1)
        instance = instance.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi_flat = self.explainer(instance).cpu().numpy()[0]
        
        return phi_flat.reshape(self.time_steps, self.n_features)
    
    def save(self, path, filename="fastshap_explainer"):
        """Save trained explainer to disk"""
        os.makedirs(path, exist_ok=True)
        state = {
            'explainer': self.explainer.state_dict(),
            'baseline': self.baseline.cpu(),
            'base_pred': self.base_pred.cpu(),
            'input_dim': self.input_dim,
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'best_loss': self.best_loss,
            'history': self.history,
            'init_params': self._init_params
        }
        torch.save(state, os.path.join(path, f"{filename}.pt"))
        return os.path.join(path, f"{filename}.pt")
    
    @classmethod
    def load(cls, path, filename="fastshap_explainer", device_override=None):
        """Load trained explainer from disk"""
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
        exp.base_pred = state['base_pred'].to(dev)
        exp.best_loss = state.get('best_loss', float('inf'))
        exp.history = state.get('history', {})
        
        exp.explainer = FastSHAPNetwork(
            exp.input_dim, params.get('hidden_dim', 256),
            params.get('n_layers', 2), params.get('dropout_rate', 0.2)
        ).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        return exp


# ============================
# VISUALIZATION (Centralized)
# ============================

def generate_shap_heatmap(shap_vals, feature_names, output_path, method_name, sample_idx=None):
    """Generate SHAP value heatmap"""
    if shap_vals is None:
        return False
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        vmax = np.max(np.abs(shap_vals))
        im = ax.imshow(shap_vals.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature')
        title = f'{method_name} SHAP Values'
        if sample_idx is not None:
            title += f' (Sample {sample_idx})'
        ax.set_title(title)
        ax.set_yticks(range(len(feature_names[:shap_vals.shape[1]])))
        ax.set_yticklabels(feature_names[:shap_vals.shape[1]], fontsize=8)
        plt.colorbar(im, ax=ax, label='SHAP Value')
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Warning: Failed to generate heatmap: {e}")
        return False


def plot_convergence(history, save_path, title="Convergence"):
    """Plot training convergence"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7, linewidth=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['lr'], 'orange', alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_yscale('log')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimization_history(study, save_path, title="Optimization History"):
    """Plot Optuna optimization history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    trials = study.trials
    values = [t.value for t in trials if t.value is not None and np.isfinite(t.value)]
    
    if not values:
        plt.close(fig)
        return
    
    axes[0].plot(values, 'b-o', alpha=0.7, markersize=4)
    axes[0].axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.6f}')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative best
    cumulative_best = []
    current_best = float('inf')
    for v in values:
        if v < current_best:
            current_best = v
        cumulative_best.append(current_best)
    
    axes[1].plot(cumulative_best, 'g-o', alpha=0.7, markersize=4)
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Best Validation Loss')
    axes[1].set_title('Cumulative Best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================
# DATABASE SAVE FUNCTIONS
# ============================

def save_tde_trial(primary_use, option_number, model_name, trial_num, params, loss, n_train):
    """Save TDE hyperparameter trial to database"""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO tde_hyperparameter_trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, trial_num, json.dumps(params), loss, n_train, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_fastshap_trial(primary_use, option_number, model_name, trial_num, params, loss, n_train):
    """Save FastSHAP hyperparameter trial to database"""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO fastshap_hyperparameter_trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, trial_num, json.dumps(params), loss, n_train, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_explainer_metadata(primary_use, option_number, model_name, explainer_type, best_params, best_loss,
                            final_loss, n_train, time_steps, n_features, opt_time, train_time, n_trials, path, feature_names):
    """Save explainer metadata to database"""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO explainer_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, explainer_type, json.dumps(best_params), best_loss,
          final_loss, n_train, time_steps, n_features, opt_time, train_time, n_trials, path,
          json.dumps(feature_names), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_comparison(primary_use, option_number, model_name, sample_idx, method,
                    fidelity, rel_corr, rel_mse, sparsity, complexity, eff_err, comp_time):
    """Save comparison results to database"""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO comparison_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, sample_idx, method, fidelity, rel_corr, rel_mse,
          sparsity, complexity, eff_err, comp_time, datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ============================
# HYPERPARAMETER OPTIMIZATION
# ============================

def get_optuna_db_path(primary_use, option_number, model_name, explainer_type):
    """Get path for Optuna study database"""
    optuna_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type
    optuna_dir.mkdir(parents=True, exist_ok=True)
    return str(optuna_dir / "optuna_study.db")


def create_tde_objective(X_train, X_val, model_predict_func, feature_names, window_size, n_epochs):
    """Create Optuna objective for TDE hyperparameter optimization"""
    def objective(trial):
        params = {
            # Regularization
            'l1_lambda': trial.suggest_float('l1_lambda', 0.0001, 1.0, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 0.0001, 0.1, log=True),
            'smoothness_lambda': trial.suggest_float('smoothness_lambda', 0.001, 0.3),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.5),
            'sparsity_lambda': trial.suggest_float('sparsity_lambda', 0.01, 0.5),
            'target_sparsity': trial.suggest_float('target_sparsity', 0.50, 0.80),
            'sparsity_threshold': trial.suggest_float('sparsity_threshold', 0.001, 0.05, log=True),
            
            # Architecture
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_conv_layers': trial.suggest_int('n_conv_layers', 1, 3),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'n_attention_heads': trial.suggest_categorical('n_attention_heads', [2, 4, 8]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            
            # Training
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 8),
        }
        
        try:
            tde = TemporalDeepExplainer(
                n_epochs=n_epochs, patience=EARLY_STOP_PATIENCE, verbose=False,
                window_size=window_size, paired_sampling=True, **params
            )
            val_loss = tde.train(X_train, X_val, model_predict_func, feature_names)
            del tde
            torch.cuda.empty_cache()
            return val_loss
        except Exception:
            return float('inf')
    return objective


def create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs):
    """Create Optuna objective for FastSHAP hyperparameter optimization"""
    def objective(trial):
        params = {
            'l1_lambda': trial.suggest_float('l1_lambda', 0.001, 0.3, log=True),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.5),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 8),
        }
        
        try:
            fs = FastSHAPExplainer(
                n_epochs=n_epochs, patience=EARLY_STOP_PATIENCE, verbose=False,
                paired_sampling=True, **params
            )
            val_loss = fs.train(X_train, X_val, model_predict_func, feature_names)
            del fs
            torch.cuda.empty_cache()
            return val_loss
        except Exception:
            return float('inf')
    return objective


def run_optimization(explainer_type, X_train, X_val, model_predict_func, feature_names,
                     n_trials, n_epochs, primary_use, option_number, model_name, window_size=6):
    """Run hyperparameter optimization"""
    optuna_db = get_optuna_db_path(primary_use, option_number, model_name, explainer_type)
    storage = f"sqlite:///{optuna_db}"
    
    study_name = f"{explainer_type}_{model_name}_v53"
    
    study = optuna.create_study(
        direction='minimize', sampler=TPESampler(seed=42),
        study_name=study_name, storage=storage, load_if_exists=True
    )
    
    if explainer_type == 'tde':
        objective = create_tde_objective(X_train, X_val, model_predict_func, feature_names, window_size, n_epochs)
        callback = lambda s, t: save_tde_trial(primary_use, option_number, model_name, t.number, t.params, t.value, len(X_train))
    else:
        objective = create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs)
        callback = lambda s, t: save_fastshap_trial(primary_use, option_number, model_name, t.number, t.params, t.value, len(X_train))
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)
    return study, time.time() - start


# ============================
# DATA LOADING
# ============================

def load_dataset(primary_use, option_number):
    """Load dataset using preprocess function"""
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path="energy_data.db", primary_use=primary_use, option_number=option_number,
        scaled=True, scale_type="both"
    )


def get_datasets():
    """Get all available datasets from benchmark database"""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query(
        'SELECT DISTINCT primary_use, option_number FROM prediction_performance ORDER BY primary_use, option_number',
        conn
    )
    conn.close()
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])} for _, r in df.iterrows()]


def get_models(primary_use, option_number):
    """Get all available models for a dataset"""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query(
        'SELECT DISTINCT model_name FROM prediction_performance WHERE primary_use = ? AND option_number = ?',
        conn, params=(primary_use, option_number)
    )
    conn.close()
    return df['model_name'].tolist()


def load_model(primary_use, option_number, model_name):
    """Load trained model"""
    from dl import load_complete_model
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return load_complete_model(str(model_path), device=device)


# ============================
# USER INPUT
# ============================

def get_user_inputs():
    """Get user configuration through interactive prompts"""
    print("\n" + "="*80)
    print("TDE & FastSHAP Training System v5.3")
    print("="*80)
    print("Key Improvements:")
    print("  • Centralized plots folder")
    print("  • Skip existing records (comprehensive check)")
    print("  • Window-Sequential masking ONLY for TDE")
    print("  • Speed optimizations (cached kernels, pre-computed baselines)")
    print("="*80)
    
    datasets = get_datasets()
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    # Select primary use
    uses = sorted(set(d['primary_use'] for d in datasets))
    print(f"\n📊 Available Primary Uses ({len(uses)}):")
    for i, u in enumerate(uses):
        print(f"  {i}: {u}")
    
    uses_input = input(f"\n--> Select primary use numbers (comma-separated) or 'all' [default: all]: ").strip()
    if uses_input.lower() == 'all' or uses_input == '':
        selected_uses = uses
    else:
        try:
            indices = [int(x.strip()) for x in uses_input.split(',')]
            selected_uses = [uses[i] for i in indices if 0 <= i < len(uses)]
        except:
            selected_uses = uses
    
    selected_datasets = [d for d in datasets if d['primary_use'] in selected_uses]
    
    # Select models
    all_model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN1D', 'DCNN', 'TCN', 'WaveNet', 'TFT', 'TST']
    print(f"\n🤖 Available Models:")
    for i, m in enumerate(all_model_types):
        print(f"  {i}: {m}")
    
    models_input = input(f"\n--> Select model numbers (comma-separated) or 'all' [default: all]: ").strip()
    if models_input.lower() == 'all' or models_input == '':
        selected_models = all_model_types
    else:
        try:
            indices = [int(x.strip()) for x in models_input.split(',')]
            selected_models = [all_model_types[i] for i in indices if 0 <= i < len(all_model_types)]
        except:
            selected_models = all_model_types
    
    # Explainer type
    print("\n🔬 Explainer Types:")
    print("  0: TDE only")
    print("  1: FastSHAP only")
    print("  2: Both")
    explainer_choice = input("--> Select [0-2, default: 2]: ").strip()
    if explainer_choice == '0':
        explainer_types = ['tde']
    elif explainer_choice == '1':
        explainer_types = ['fastshap']
    else:
        explainer_types = ['tde', 'fastshap']
    
    # Other settings
    window_size = int(input(f"\n--> Window size [{DEFAULT_WINDOW_SIZE}]: ").strip() or DEFAULT_WINDOW_SIZE)
    n_trials = int(input(f"--> Number of trials [{DEBUG_N_TRIALS}]: ").strip() or DEBUG_N_TRIALS)
    n_test = int(input("--> Test samples for comparison [5]: ").strip() or "5")
    
    # Check existing status
    print("\n⏳ Checking existing training status...")
    status = get_training_status_summary(selected_datasets, selected_models, explainer_types)
    completed, pending = print_status_summary(status, selected_models, explainer_types)
    
    # Ask about replacement
    replace_existing = False
    if completed:
        replace_input = input("\n--> Replace existing completed explainers? (yes/no) [default: no]: ").strip().lower()
        replace_existing = replace_input in ['yes', 'y']
    
    return {
        'datasets': selected_datasets,
        'models': selected_models,
        'explainer_types': explainer_types,
        'window_size': window_size,
        'n_trials': n_trials,
        'n_test_samples': n_test,
        'replace_existing': replace_existing,
        'status': status,
        'pending': pending
    }


# ============================
# MAIN TRAINING FUNCTION
# ============================

def train_explainer(primary_use, option_number, model_name, container, explainer_type,
                    window_size, n_trials, n_test_samples, replace_existing=False):
    """Train a single explainer"""
    
    # Check if already complete
    status = check_explainer_complete(primary_use, option_number, model_name, explainer_type)
    if status['complete'] and not replace_existing:
        print(f"  ⏭️  {explainer_type.upper()} already complete - SKIPPING")
        return None
    
    print(f"\n  📦 Training {explainer_type.upper()}...")
    
    # Load model
    try:
        model = load_model(primary_use, option_number, model_name)
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return None
    
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
    np.random.seed(42)
    X_all = X_all[np.random.choice(len(X_all), int(len(X_all) * frac), replace=False)]
    n_val = int(len(X_all) * VALIDATION_SPLIT)
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    
    print(f"    Data: Train={len(X_train)} Val={len(X_val)}")
    
    trial_epochs = DEBUG_TRIAL_EPOCHS if DEBUG_MODE else PROD_TRIAL_EPOCHS
    final_epochs = DEBUG_FINAL_EPOCHS if DEBUG_MODE else PROD_FINAL_EPOCHS
    
    # Get directories
    exp_dir = get_explainer_dir(primary_use, option_number, model_name, explainer_type)
    plots_dir = get_plots_dir(primary_use, option_number, model_name, explainer_type)
    
    # Run optimization
    print(f"    Optimizing hyperparameters ({n_trials} trials)...")
    study, opt_time = run_optimization(
        explainer_type, X_train, X_val, predict_first_horizon, container.feature_names,
        n_trials, trial_epochs, primary_use, option_number, model_name, window_size
    )
    
    print(f"    Best validation loss: {study.best_value:.6f}")
    
    # Final training
    print(f"    Final training ({final_epochs} epochs)...")
    
    if explainer_type == 'tde':
        explainer = TemporalDeepExplainer(
            n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE, verbose=True,
            window_size=window_size, paired_sampling=True, **study.best_params
        )
    else:
        explainer = FastSHAPExplainer(
            n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE, verbose=True,
            paired_sampling=True, **study.best_params
        )
    
    start = time.time()
    final_loss = explainer.train(X_train, X_val, predict_first_horizon, container.feature_names)
    train_time = time.time() - start
    
    print(f"    Final loss: {final_loss:.6f} (Time: {train_time:.1f}s)")
    
    # Save explainer
    explainer.save(str(exp_dir))
    
    # Save metadata to database
    save_explainer_metadata(
        primary_use, option_number, model_name, explainer_type.upper(),
        study.best_params, study.best_value, final_loss, len(X_train),
        time_steps, n_features, opt_time, train_time, n_trials,
        str(exp_dir), container.feature_names
    )
    
    # Generate plots (centralized location)
    plot_convergence(explainer.history, plots_dir / f"convergence.png", f"{explainer_type.upper()} Convergence")
    plot_optimization_history(study, plots_dir / f"optimization_history.png", f"{explainer_type.upper()} Optimization")
    
    # Generate sample heatmaps
    X_test = container.X_test[:n_test_samples]
    for idx in range(min(3, len(X_test))):
        shap_vals = explainer.explain(X_test[idx])
        generate_shap_heatmap(
            shap_vals, container.feature_names,
            plots_dir / f"heatmap_sample{idx}.png",
            explainer_type.upper(), idx
        )
    
    print(f"    ✅ {explainer_type.upper()} training complete!")
    print(f"    📊 Plots saved to: {plots_dir}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return explainer


# ============================
# MAIN FUNCTION
# ============================

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚀 TDE & FastSHAP Training System v5.3")
    print("="*80)
    
    # Initialize database
    init_database()
    
    # Create centralized plots directory
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get user configuration
    config = get_user_inputs()
    if config is None:
        return
    
    # Show pending work
    pending = config['pending']
    if not pending and not config['replace_existing']:
        print("\n✅ All selected explainers are already trained!")
        return
    
    print(f"\n📋 Will process {len(pending)} explainer(s)")
    
    confirm = input("\n--> Proceed with training? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    # Process each pending item
    total_start = time.time()
    completed_count = 0
    
    for primary_use, option_number, model_name, exp_type in pending:
        print(f"\n{'='*80}")
        print(f"📦 {primary_use} - Option {option_number} - {model_name} - {exp_type.upper()}")
        print(f"{'='*80}")
        
        try:
            container = load_dataset(primary_use, option_number)
            
            result = train_explainer(
                primary_use, option_number, model_name, container, exp_type,
                config['window_size'], config['n_trials'], config['n_test_samples'],
                config['replace_existing']
            )
            
            if result is not None:
                completed_count += 1
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    print(f"  Completed: {completed_count}/{len(pending)}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Plots saved to: {PLOTS_DIR}")
    print(f"  Database: {EXPLAINER_DB}")
    print("="*80)


if __name__ == "__main__":
    main()