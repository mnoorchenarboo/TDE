"""
TDE & FastSHAP Training System
- TDE: Temporal Deep Explainer with window/feature masking
- FastSHAP: Simple MLP with element-wise masking (Jethani et al., 2022)
- Both explain FIRST HORIZON ONLY
- Proper Optuna database storage with feature importance
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

warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SHAP
import shap

# Local imports
from dl import load_complete_model


# ============================
# CONFIGURATION
# ============================

BENCHMARK_DB = "benchmark_results.db"
TDE_DB = "tde_results.db"
FASTSHAP_DB = "fastshap_results.db"
RESULTS_BASE_DIR = "results"

# Debug Mode
DEBUG_MODE = True
DEBUG_TRAINING_FRACTION = 0.15
DEBUG_TRIAL_EPOCHS = 10
DEBUG_FINAL_EPOCHS = 50
DEBUG_N_TRIALS = 10

# Production Mode
PROD_TRAINING_FRACTION = 0.30
PROD_TRIAL_EPOCHS = 20
PROD_FINAL_EPOCHS = 100
PROD_N_TRIALS = 30

# FIXED Parameters
PAIRED_SAMPLING = True
VALIDATION_SPLIT = 0.20
DEFAULT_WINDOW_SIZE = 6

# Reliability testing
NOISE_STD = 0.01

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")


# ============================
# LOGGING SETUP
# ============================

def setup_logger(log_path):
    """Setup logger - Windows compatible"""
    logger = logging.getLogger('Explainer_Training')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                msg_safe = msg.encode('ascii', 'replace').decode('ascii')
                self.stream.write(msg_safe + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    ch = SafeStreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================
# DATABASE INITIALIZATION
# ============================

def init_tde_database(db_path=TDE_DB):
    """Initialize TDE database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tde_hyperparameter_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            trial_number INTEGER NOT NULL,
            hyperparameters TEXT NOT NULL,
            validation_loss REAL NOT NULL,
            n_training_samples INTEGER,
            timestamp TEXT NOT NULL,
            UNIQUE(primary_use, option_number, model_name, trial_number)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tde_model_metadata (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            best_hyperparameters TEXT NOT NULL,
            best_validation_loss REAL NOT NULL,
            final_training_loss REAL,
            n_training_samples INTEGER,
            time_steps INTEGER,
            n_features INTEGER,
            optimization_time REAL,
            training_time REAL,
            n_trials INTEGER,
            explainer_path TEXT,
            feature_names TEXT,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, model_name)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tde_comparison_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            method TEXT NOT NULL,
            fidelity REAL,
            reliability_correlation REAL,
            reliability_mse REAL,
            sparsity REAL,
            complexity REAL,
            efficiency_error REAL,
            computation_time REAL,
            timestamp TEXT NOT NULL,
            UNIQUE(primary_use, option_number, model_name, sample_idx, method)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[OK] TDE database initialized: {db_path}")


def init_fastshap_database(db_path=FASTSHAP_DB):
    """Initialize FastSHAP database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fastshap_hyperparameter_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            trial_number INTEGER NOT NULL,
            hyperparameters TEXT NOT NULL,
            validation_loss REAL NOT NULL,
            n_training_samples INTEGER,
            timestamp TEXT NOT NULL,
            UNIQUE(primary_use, option_number, model_name, trial_number)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fastshap_model_metadata (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            best_hyperparameters TEXT NOT NULL,
            best_validation_loss REAL NOT NULL,
            final_training_loss REAL,
            n_training_samples INTEGER,
            input_dim INTEGER,
            optimization_time REAL,
            training_time REAL,
            n_trials INTEGER,
            explainer_path TEXT,
            feature_names TEXT,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, model_name)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fastshap_comparison_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            method TEXT NOT NULL,
            fidelity REAL,
            reliability_correlation REAL,
            reliability_mse REAL,
            sparsity REAL,
            complexity REAL,
            efficiency_error REAL,
            computation_time REAL,
            timestamp TEXT NOT NULL,
            UNIQUE(primary_use, option_number, model_name, sample_idx, method)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[OK] FastSHAP database initialized: {db_path}")


def get_optuna_db_path(primary_use, option_number, model_name, explainer_type):
    """Get Optuna database path for a specific model"""
    optuna_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type
    optuna_dir.mkdir(parents=True, exist_ok=True)
    return str(optuna_dir / "optuna_study.db")


# ============================
# SINGLE OUTPUT MODEL WRAPPER
# ============================

class SingleHorizonWrapper(nn.Module):
    """Wraps a multi-horizon model to output only the first horizon"""
    
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
# TDE NETWORK
# ============================

class TemporalExplainerNetwork(nn.Module):
    """TDE neural network with Softshrink + L1 for sparsity"""
    
    def __init__(self, time_steps, n_features, 
                 num_attention_heads=4, 
                 num_conv_layers=2,
                 num_filters=64, 
                 kernel_size=3, 
                 dropout_rate=0.2,
                 softshrink_lambda=0.001):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(n_features, num_filters),
            nn.GELU(),
            nn.LayerNorm(num_filters),
            nn.Dropout(dropout_rate)
        )
        
        # Convolution layers
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.LayerNorm([num_filters, time_steps]),
                nn.Dropout(dropout_rate)
            ])
        self.time_conv = nn.Sequential(*conv_layers)
        
        # Attention
        self.attention = nn.MultiheadAttention(num_filters, num_attention_heads, 
                                                dropout=dropout_rate, batch_first=True)
        self.attn_norm = nn.LayerNorm(num_filters)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Conv1d(num_filters, num_filters // 2, 1),
            nn.GELU(),
            nn.Conv1d(num_filters // 2, n_features, 1)
        )
        self.softshrink = nn.Softshrink(lambd=softshrink_lambda)
        
        # Init
        for m in self.output_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.input_embed(x)
        h = h.permute(0, 2, 1)
        h = self.time_conv(h)
        h_attn = h.permute(0, 2, 1)
        attn_out, _ = self.attention(h_attn, h_attn, h_attn)
        h_attn = self.attn_norm(h_attn + attn_out)
        h = h_attn.permute(0, 2, 1)
        output = self.output_proj(h)
        output = self.softshrink(output)
        return output.permute(0, 2, 1)


# ============================
# FASTSHAP NETWORK (Simple MLP)
# ============================

class FastSHAPNetwork(nn.Module):
    """
    FastSHAP neural network - Simple MLP (Jethani et al., 2022)
    No temporal structure - treats input as flat vector
    """
    
    def __init__(self, input_dim, 
                 hidden_dims=[256, 256],
                 dropout_rate=0.2,
                 activation='gelu'):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        act_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn,
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Init output with small weights
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x):
        """
        x: (batch, input_dim) - flattened input
        output: (batch, input_dim) - SHAP values
        """
        return self.network(x)


# ============================
# TDE TRAINER CLASS
# ============================

class TemporalDeepExplainer:
    """TDE Trainer - First Horizon Only"""
    
    def __init__(self,
                 n_epochs=100, batch_size=256, patience=15, verbose=True, min_lr=1e-6,
                 l1_lambda=0.01, smoothness_lambda=0.1, efficiency_lambda=0.1, weight_decay=1e-4,
                 num_attention_heads=4, num_conv_layers=2, num_filters=64, kernel_size=3,
                 dropout_rate=0.2, softshrink_lambda=0.001,
                 optimizer_type='adam', learning_rate=1e-3,
                 window_size=6, paired_sampling=True, samples_per_feature=1, masking_mode='window'):
        
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        self.l1_lambda = l1_lambda
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda
        self.weight_decay = weight_decay
        
        self.num_attention_heads = num_attention_heads
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.softshrink_lambda = softshrink_lambda
        
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        self.window_size = window_size
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        self.masking_mode = masking_mode
        
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.n_windows = None
        self.model_predict_func = None
        
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 
                       'mse': [], 'eff': [], 'nonzero_pct': []}
        
        self._init_params = {
            'n_epochs': n_epochs, 'batch_size': batch_size, 'patience': patience,
            'verbose': verbose, 'min_lr': min_lr, 'l1_lambda': l1_lambda,
            'smoothness_lambda': smoothness_lambda, 'efficiency_lambda': efficiency_lambda,
            'weight_decay': weight_decay, 'num_attention_heads': num_attention_heads,
            'num_conv_layers': num_conv_layers, 'num_filters': num_filters,
            'kernel_size': kernel_size, 'dropout_rate': dropout_rate,
            'softshrink_lambda': softshrink_lambda, 'optimizer_type': optimizer_type,
            'learning_rate': learning_rate, 'window_size': window_size,
            'paired_sampling': paired_sampling, 'samples_per_feature': samples_per_feature,
            'masking_mode': masking_mode
        }
    
    def _setup(self, X_train, model_predict_func, feature_names):
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.n_windows = (self.time_steps + self.window_size - 1) // self.window_size
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        baseline_np = self.baseline.unsqueeze(0).cpu().numpy()
        base_raw = model_predict_func(baseline_np)
        self.base_pred = torch.tensor(float(np.atleast_1d(base_raw).flatten()[0]),
                                       dtype=torch.float32, device=self.device)
        
        self.explainer = TemporalExplainerNetwork(
            time_steps=self.time_steps, n_features=self.n_features,
            num_attention_heads=self.num_attention_heads,
            num_conv_layers=self.num_conv_layers,
            num_filters=self.num_filters, kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate, softshrink_lambda=self.softshrink_lambda
        ).to(self.device)
        
        if self.verbose:
            n_params = sum(p.numel() for p in self.explainer.parameters())
            print(f"  [TDE] Explainer: {n_params:,} params, shape: ({self.time_steps}, {self.n_features})")
    
    def _compute_shapley_kernel(self, d):
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float32)
        binom_coeffs = torch.tensor([comb(d, int(k.item()), exact=True) for k in k_values],
                                     device=self.device, dtype=torch.float32)
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        probs = weights / weights.sum()
        return weights, probs
    
    def _generate_masks(self, batch_size, d, probs):
        if self.masking_mode == 'window':
            return self._generate_window_masks(batch_size)
        elif self.masking_mode == 'feature':
            return self._generate_feature_masks(batch_size, probs)
        else:
            return self._generate_element_masks(batch_size, d, probs)
    
    def _generate_window_masks(self, batch_size):
        total = batch_size * self.samples_per_feature
        masks = torch.ones(total, self.time_steps, self.n_features, device=self.device)
        for i in range(total):
            n_mask = torch.randint(1, max(2, self.n_windows), (1,)).item()
            n_feat = torch.randint(1, self.n_features + 1, (1,)).item()
            feats = torch.randperm(self.n_features, device=self.device)[:n_feat]
            wins = torch.randperm(self.n_windows, device=self.device)[:n_mask]
            for w in wins:
                start = int(w) * self.window_size
                end = min((int(w) + 1) * self.window_size, self.time_steps)
                for f in feats:
                    masks[i, start:end, int(f)] = 0.0
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks
    
    def _generate_feature_masks(self, batch_size, probs):
        _, probs_f = self._compute_shapley_kernel(self.n_features)
        total = batch_size * self.samples_per_feature
        k_idx = torch.multinomial(probs_f, total, replacement=True)
        k_samples = torch.arange(1, self.n_features, device=self.device)[k_idx]
        rand = torch.rand(total, self.n_features, device=self.device)
        sorted_idx = torch.argsort(rand, dim=1)
        masks = (sorted_idx < k_samples.unsqueeze(1)).float()
        masks = masks.unsqueeze(1).repeat(1, self.time_steps, 1)
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks
    
    def _generate_element_masks(self, batch_size, d, probs):
        total = batch_size * self.samples_per_feature
        k_idx = torch.multinomial(probs, total, replacement=True)
        k_samples = torch.arange(1, d, device=self.device)[k_idx]
        rand = torch.rand(total, d, device=self.device)
        sorted_idx = torch.argsort(rand, dim=1)
        masks = (sorted_idx < k_samples.unsqueeze(1)).float()
        masks = masks.view(-1, self.time_steps, self.n_features)
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks
    
    def _get_predictions(self, inputs):
        with torch.no_grad():
            preds = self.model_predict_func(inputs.cpu().numpy())
            return torch.tensor(np.atleast_1d(preds).flatten(), dtype=torch.float32, device=self.device)
    
    def _process_batch(self, X_batch, d, probs, optimizer):
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device)
        
        expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks = self._generate_masks(batch_size, d, probs)
        
        total = masks.size(0)
        repeat = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat, 1, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).repeat(total, 1, 1)
        
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        preds_masked = self._get_predictions(masked)
        phi = self.explainer(X_paired)
        
        # Fidelity
        masked_sum = (masks * phi).sum(dim=(1, 2))
        mse_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
        
        # Efficiency
        preds_orig = self._get_predictions(X_paired)
        phi_sum = phi.sum(dim=(1, 2))
        eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
        
        # Smoothness
        smooth_loss = self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean() if phi.size(1) > 1 else 0.0
        
        # L1
        l1_loss = self.l1_lambda * torch.abs(phi).mean()
        
        loss = mse_loss + eff_loss + smooth_loss + l1_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            nonzero = (torch.abs(phi) > 1e-6).float().mean().item() * 100
        
        return loss.item(), mse_loss.item(), eff_loss.item() if isinstance(eff_loss, float) else eff_loss, nonzero
    
    def _validate(self, X_val):
        self.explainer.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_val)), batch_size=self.batch_size, shuffle=False)
        total_loss, total_nz, n = 0.0, 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi = self.explainer(X_batch)
                preds = self._get_predictions(X_batch)
                eff_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                smooth = ((phi[:, 1:, :] - phi[:, :-1, :]) ** 2).mean() if phi.size(1) > 1 else 0.0
                total_loss += (eff_err + self.smoothness_lambda * smooth).item()
                total_nz += (torch.abs(phi) > 1e-6).float().mean().item() * 100
                n += 1
        
        self.explainer.train()
        return total_loss / max(n, 1), total_nz / max(n, 1)
    
    def train(self, X_train, X_val, model_predict_func, feature_names):
        self._setup(X_train, model_predict_func, feature_names)
        
        d = self.n_features if self.masking_mode == 'feature' else (
            self.n_windows * self.n_features if self.masking_mode == 'window' else self.time_steps * self.n_features)
        _, probs = self._compute_shapley_kernel(d)
        
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=self.batch_size, shuffle=True)
        
        opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(self.optimizer_type, torch.optim.Adam)
        optimizer = opt_cls(self.explainer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.patience // 2, factor=0.5, min_lr=self.min_lr)
        
        best_val, best_weights, no_improve = float('inf'), None, 0
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            ep_loss, ep_mse, ep_eff, ep_nz, n = 0, 0, 0, 0, 0
            
            for (X_batch,) in loader:
                loss, mse, eff, nz = self._process_batch(X_batch, d, probs, optimizer)
                ep_loss += loss
                ep_mse += mse
                ep_eff += eff
                ep_nz += nz
                n += 1
            
            ep_loss /= n
            ep_mse /= n
            ep_eff /= n
            ep_nz /= n
            
            val_loss, val_nz = self._validate(X_val)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(ep_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            self.history['mse'].append(ep_mse)
            self.history['eff'].append(ep_eff)
            self.history['nonzero_pct'].append(ep_nz)
            
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {ep_loss:.4f} | Val: {val_loss:.4f} | NZ: {ep_nz:.0f}%")
            
            if no_improve >= self.patience and epoch >= self.n_epochs // 3:
                if self.verbose:
                    print(f"  [EARLY STOP] epoch {epoch+1}")
                break
        
        if best_weights:
            self.explainer.load_state_dict(best_weights)
        self.best_loss = best_val
        return best_val
    
    def explain(self, instance, enforce_efficiency=False):
        """Get SHAP values - enforce_efficiency=False for true efficiency measurement"""
        if self.explainer is None:
            raise ValueError("Not trained")
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        instance = instance.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi = self.explainer(instance).cpu().numpy()[0]
        
        if enforce_efficiency:
            pred = self._get_predictions(instance).item()
            expected = pred - self.base_pred.item()
            current = np.sum(phi)
            if abs(current) > 1e-10:
                phi = phi * (expected / current)
        
        return phi
    
    def explain_batch(self, instances, enforce_efficiency=False):
        if isinstance(instances, np.ndarray):
            instances = torch.FloatTensor(instances)
        instances = instances.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi = self.explainer(instances).cpu().numpy()
        
        if enforce_efficiency:
            preds = self._get_predictions(instances).cpu().numpy()
            base = self.base_pred.item()
            for i in range(len(phi)):
                expected = preds[i] - base
                current = np.sum(phi[i])
                if abs(current) > 1e-10:
                    phi[i] = phi[i] * (expected / current)
        
        return phi
    
    def save(self, path, filename="tde_explainer"):
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
        save_path = os.path.join(path, f"{filename}.pt")
        torch.save(state, save_path)
        return save_path
    
    @classmethod
    def load(cls, path, filename="tde_explainer", device_override=None):
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
            time_steps=exp.time_steps, n_features=exp.n_features,
            num_attention_heads=params.get('num_attention_heads', 4),
            num_conv_layers=params.get('num_conv_layers', 2),
            num_filters=params.get('num_filters', 64),
            kernel_size=params.get('kernel_size', 3),
            dropout_rate=params.get('dropout_rate', 0.2),
            softshrink_lambda=params.get('softshrink_lambda', 0.001)
        ).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        return exp


# ============================
# FASTSHAP TRAINER CLASS
# ============================

class FastSHAPExplainer:
    """
    FastSHAP Trainer (Jethani et al., 2022)
    Simple MLP with element-wise masking - First Horizon Only
    """
    
    def __init__(self,
                 n_epochs=100, batch_size=256, patience=15, verbose=True, min_lr=1e-6,
                 l1_lambda=0.01, efficiency_lambda=0.1, weight_decay=1e-4,
                 hidden_dims=[256, 256], dropout_rate=0.2, activation='gelu',
                 optimizer_type='adam', learning_rate=1e-3,
                 paired_sampling=True, samples_per_feature=1):
        
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        self.l1_lambda = l1_lambda
        self.efficiency_lambda = efficiency_lambda
        self.weight_decay = weight_decay
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        
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
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mse': [], 'eff': []}
        
        self._init_params = {
            'n_epochs': n_epochs, 'batch_size': batch_size, 'patience': patience,
            'verbose': verbose, 'min_lr': min_lr, 'l1_lambda': l1_lambda,
            'efficiency_lambda': efficiency_lambda, 'weight_decay': weight_decay,
            'hidden_dims': hidden_dims, 'dropout_rate': dropout_rate,
            'activation': activation, 'optimizer_type': optimizer_type,
            'learning_rate': learning_rate, 'paired_sampling': paired_sampling,
            'samples_per_feature': samples_per_feature
        }
    
    def _setup(self, X_train, model_predict_func, feature_names):
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.input_dim = self.time_steps * self.n_features
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        # Flatten for FastSHAP
        X_flat = X_train.reshape(len(X_train), -1)
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        baseline_np = self.baseline.unsqueeze(0).cpu().numpy().reshape(1, self.time_steps, self.n_features)
        base_raw = model_predict_func(baseline_np)
        self.base_pred = torch.tensor(float(np.atleast_1d(base_raw).flatten()[0]),
                                       dtype=torch.float32, device=self.device)
        
        self.explainer = FastSHAPNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        ).to(self.device)
        
        if self.verbose:
            n_params = sum(p.numel() for p in self.explainer.parameters())
            print(f"  [FastSHAP] Explainer: {n_params:,} params, input_dim: {self.input_dim}")
    
    def _compute_shapley_kernel(self, d):
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float32)
        binom_coeffs = torch.tensor([comb(d, int(k.item()), exact=True) for k in k_values],
                                     device=self.device, dtype=torch.float32)
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        probs = weights / weights.sum()
        return weights, probs
    
    def _generate_element_masks(self, batch_size, probs):
        """Simple element-wise masking for FastSHAP"""
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
        """Get predictions from flattened inputs"""
        with torch.no_grad():
            inputs_3d = inputs_flat.cpu().numpy().reshape(-1, self.time_steps, self.n_features)
            preds = self.model_predict_func(inputs_3d)
            return torch.tensor(np.atleast_1d(preds).flatten(), dtype=torch.float32, device=self.device)
    
    def _process_batch(self, X_batch_flat, probs, optimizer):
        batch_size = X_batch_flat.size(0)
        X_batch_flat = X_batch_flat.to(self.device)
        
        expanded = X_batch_flat.repeat(self.samples_per_feature, 1)
        masks = self._generate_element_masks(batch_size, probs)
        
        total = masks.size(0)
        repeat = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).repeat(total, 1)
        
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        preds_masked = self._get_predictions(masked)
        phi = self.explainer(X_paired)
        
        # Fidelity
        masked_sum = (masks * phi).sum(dim=1)
        mse_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
        
        # Efficiency
        preds_orig = self._get_predictions(X_paired)
        phi_sum = phi.sum(dim=1)
        eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
        
        # L1
        l1_loss = self.l1_lambda * torch.abs(phi).mean()
        
        loss = mse_loss + eff_loss + l1_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item(), mse_loss.item(), eff_loss.item()
    
    def _validate(self, X_val_flat):
        self.explainer.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_flat)), batch_size=self.batch_size, shuffle=False)
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
        self._setup(X_train, model_predict_func, feature_names)
        
        # Flatten data
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        
        _, probs = self._compute_shapley_kernel(self.input_dim)
        
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_flat)), batch_size=self.batch_size, shuffle=True)
        
        opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(self.optimizer_type, torch.optim.Adam)
        optimizer = opt_cls(self.explainer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.patience // 2, factor=0.5, min_lr=self.min_lr)
        
        best_val, best_weights, no_improve = float('inf'), None, 0
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            ep_loss, ep_mse, ep_eff, n = 0, 0, 0, 0
            
            for (X_batch,) in loader:
                loss, mse, eff = self._process_batch(X_batch, probs, optimizer)
                ep_loss += loss
                ep_mse += mse
                ep_eff += eff
                n += 1
            
            ep_loss /= n
            ep_mse /= n
            ep_eff /= n
            
            val_loss = self._validate(X_val_flat)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(ep_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            self.history['mse'].append(ep_mse)
            self.history['eff'].append(ep_eff)
            
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {ep_loss:.4f} | Val: {val_loss:.4f}")
            
            if no_improve >= self.patience and epoch >= self.n_epochs // 3:
                if self.verbose:
                    print(f"  [EARLY STOP] epoch {epoch+1}")
                break
        
        if best_weights:
            self.explainer.load_state_dict(best_weights)
        self.best_loss = best_val
        return best_val
    
    def explain(self, instance, enforce_efficiency=False):
        """Get SHAP values - returns shape (time_steps, n_features)"""
        if self.explainer is None:
            raise ValueError("Not trained")
        
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
        
        # Reshape to (time_steps, n_features)
        phi = phi_flat.reshape(self.time_steps, self.n_features)
        
        if enforce_efficiency:
            inst_3d = instance.cpu().numpy().reshape(1, self.time_steps, self.n_features)
            pred = self.model_predict_func(inst_3d).flatten()[0]
            expected = pred - self.base_pred.item()
            current = np.sum(phi)
            if abs(current) > 1e-10:
                phi = phi * (expected / current)
        
        return phi
    
    def explain_batch(self, instances, enforce_efficiency=False):
        if isinstance(instances, np.ndarray):
            instances = torch.FloatTensor(instances)
        
        batch_size = instances.size(0)
        instances_flat = instances.reshape(batch_size, -1).to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi_flat = self.explainer(instances_flat).cpu().numpy()
        
        phi = phi_flat.reshape(batch_size, self.time_steps, self.n_features)
        
        if enforce_efficiency:
            inst_3d = instances.cpu().numpy()
            preds = self.model_predict_func(inst_3d).flatten()
            base = self.base_pred.item()
            for i in range(len(phi)):
                expected = preds[i] - base
                current = np.sum(phi[i])
                if abs(current) > 1e-10:
                    phi[i] = phi[i] * (expected / current)
        
        return phi
    
    def save(self, path, filename="fastshap_explainer"):
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
        save_path = os.path.join(path, f"{filename}.pt")
        torch.save(state, save_path)
        return save_path
    
    @classmethod
    def load(cls, path, filename="fastshap_explainer", device_override=None):
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
            input_dim=exp.input_dim,
            hidden_dims=params.get('hidden_dims', [256, 256]),
            dropout_rate=params.get('dropout_rate', 0.2),
            activation=params.get('activation', 'gelu')
        ).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        return exp


# ============================
# TRADITIONAL SHAP METHODS
# ============================

class TraditionalSHAPMethods:
    """Gradient SHAP and Deep SHAP - First Horizon Only"""
    
    def __init__(self, model, background, time_steps, n_features, device=device):
        self.device = device
        self.time_steps = time_steps
        self.n_features = n_features
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(device)
        self.wrapped_model.eval()
        
        if isinstance(background, np.ndarray):
            self.background_tensor = torch.FloatTensor(background).to(device)
        else:
            self.background_tensor = background.to(device)
    
    def gradient_shap(self, instance):
        try:
            if isinstance(instance, np.ndarray):
                instance = torch.FloatTensor(instance)
            if instance.ndim == 2:
                instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            
            explainer = shap.GradientExplainer(self.wrapped_model, self.background_tensor)
            shap_vals = explainer.shap_values(instance)
            
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
        except Exception as e:
            print(f"    [WARN] Gradient SHAP failed: {e}")
            return None
    
    def deep_shap(self, instance):
        try:
            if isinstance(instance, np.ndarray):
                instance = torch.FloatTensor(instance)
            if instance.ndim == 2:
                instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            
            explainer = shap.DeepExplainer(self.wrapped_model, self.background_tensor)
            shap_vals = explainer.shap_values(instance, check_additivity=False)
            
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
        except Exception as e:
            print(f"    [WARN] Deep SHAP failed: {e}")
            return None


# ============================
# METRICS
# ============================

class ExplainabilityMetrics:
    """Metrics for first horizon - measures RAW efficiency (not post-hoc adjusted)"""
    
    def __init__(self, model, baseline, base_pred, time_steps, n_features, device=device):
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(device)
        self.wrapped_model.eval()
        self.baseline = baseline
        self.base_pred = base_pred
        self.time_steps = time_steps
        self.n_features = n_features
        self.device = device
    
    def _get_prediction(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.wrapped_model(x).cpu().numpy()
        return pred.flatten()[0]
    
    def fidelity(self, instance, shap_vals, top_k_pct=10):
        if shap_vals is None:
            return None
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        baseline = self.baseline.cpu().numpy() if isinstance(self.baseline, torch.Tensor) else self.baseline
        if baseline.ndim == 3:
            baseline = baseline[0]
        if instance.shape != shap_vals.shape or baseline.shape != instance.shape:
            return None
        
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
        if shap_vals is None:
            return None, None
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        
        perturbed = np.clip(instance + np.random.normal(0, noise_std, instance.shape), 0, 1).astype(np.float32)
        shap_pert = shap_func(perturbed)
        if shap_pert is None:
            return None, None
        
        orig, pert = shap_vals.flatten(), shap_pert.flatten()
        mask = np.isfinite(orig) & np.isfinite(pert)
        if np.sum(mask) < 10:
            return None, None
        
        corr, _ = pearsonr(orig[mask], pert[mask])
        mse = np.mean((orig[mask] - pert[mask]) ** 2)
        return (float(corr) if np.isfinite(corr) else None), float(mse)
    
    def sparsity(self, shap_vals, threshold_pct=1):
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals)
        max_val = np.max(abs_shap)
        if max_val == 0:
            return 100.0
        threshold = max_val * threshold_pct / 100
        return float(np.sum(abs_shap < threshold) / abs_shap.size * 100)
    
    def complexity(self, shap_vals):
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals).flatten() + 1e-10
        probs = abs_shap / np.sum(abs_shap)
        return float(-np.sum(probs * np.log(probs)))
    
    def efficiency_error(self, instance, shap_vals):
        """Measures RAW efficiency - how well SHAP sum matches prediction difference"""
        if shap_vals is None:
            return None
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        
        pred = self._get_prediction(instance)
        expected = pred - self.base_pred
        actual = np.sum(shap_vals)
        
        if abs(expected) < 1e-10:
            return abs(actual)
        return abs(actual - expected) / abs(expected)


# ============================
# VISUALIZATION
# ============================

def generate_shap_heatmap(shap_vals, feature_names, output_path, method_name):
    if shap_vals is None:
        return False
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        vmax = np.max(np.abs(shap_vals))
        im = ax.imshow(shap_vals.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature')
        ax.set_title(f'{method_name} SHAP Values (First Horizon)')
        ax.set_yticks(range(len(feature_names[:shap_vals.shape[1]])))
        ax.set_yticklabels(feature_names[:shap_vals.shape[1]], fontsize=8)
        plt.colorbar(im, ax=ax, label='SHAP Value')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"    [WARN] Heatmap failed: {e}")
        return False


def plot_convergence(history, save_path, title="Training Convergence"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(shap_vals, feature_names, save_path, title="Feature Importance"):
    mean_abs = np.mean(np.abs(shap_vals), axis=(0, 1)) if shap_vals.ndim == 3 else np.mean(np.abs(shap_vals), axis=0)
    idx = np.argsort(mean_abs)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_names)), mean_abs[idx])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_optuna_importance(study, save_path):
    """Plot hyperparameter importance from Optuna study"""
    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
        
        if not importances:
            return False
        
        params = list(importances.keys())
        values = list(importances.values())
        sorted_pairs = sorted(zip(params, values), key=lambda x: x[1], reverse=True)
        params, values = zip(*sorted_pairs)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(params)), values)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"    [WARN] Param importance plot failed: {e}")
        return False


# ============================
# HYPERPARAMETER OPTIMIZATION
# ============================

def create_tde_objective(X_train, X_val, model_predict_func, feature_names, window_size, n_epochs):
    def objective(trial):
        params = {
            'l1_lambda': trial.suggest_float('l1_lambda', 0.001, 0.5, log=True),
            'smoothness_lambda': trial.suggest_float('smoothness_lambda', 0.001, 0.5),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.001, 0.5),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4, 8]),
            'num_conv_layers': trial.suggest_int('num_conv_layers', 1, 3, 6, 9),
            'num_filters': trial.suggest_categorical('num_filters', [32, 64, 128]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 6, 9]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'softshrink_lambda': trial.suggest_float('softshrink_lambda', 0.0005, 0.01, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'masking_mode': trial.suggest_categorical('masking_mode', ['window', 'feature']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 5),
        }
        try:
            tde = TemporalDeepExplainer(n_epochs=n_epochs, patience=8, verbose=False,
                                         window_size=window_size, paired_sampling=True, **params)
            val_loss = tde.train(X_train, X_val, model_predict_func, feature_names)
            del tde
            torch.cuda.empty_cache()
            return val_loss
        except Exception as e:
            return float('inf')
    return objective


def create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs):
    def objective(trial):
        n_layers = trial.suggest_int('n_hidden_layers', 1, 4)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        hidden_dims = [hidden_dim] * n_layers
        
        params = {
            'l1_lambda': trial.suggest_float('l1_lambda', 0.001, 0.5, log=True),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.3),
            'hidden_dims': hidden_dims,
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'activation': trial.suggest_categorical('activation', ['gelu', 'relu']),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 1, 2),
        }
        try:
            fs = FastSHAPExplainer(n_epochs=n_epochs, patience=8, verbose=False,
                                    paired_sampling=True, **params)
            val_loss = fs.train(X_train, X_val, model_predict_func, feature_names)
            del fs
            torch.cuda.empty_cache()
            return val_loss
        except Exception as e:
            return float('inf')
    return objective


def run_tde_optimization(X_train, X_val, model_predict_func, feature_names,
                          window_size, n_trials, n_epochs,
                          primary_use, option_number, model_name):
    print(f"\n  [TDE OPT] {n_trials} trials, {n_epochs} epochs/trial")
    
    optuna_db = get_optuna_db_path(primary_use, option_number, model_name, "tde")
    storage = f"sqlite:///{optuna_db}"
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42),
                                 study_name=f"tde_{model_name}", storage=storage, load_if_exists=True)
    
    objective = create_tde_objective(X_train, X_val, model_predict_func, feature_names, window_size, n_epochs)
    
    def callback(study, trial):
        save_tde_trial(primary_use, option_number, model_name, trial.number, trial.params, trial.value, len(X_train))
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)
    opt_time = time.time() - start
    
    print(f"\n  [TDE] Best: {study.best_value:.6f}")
    return study, opt_time


def run_fastshap_optimization(X_train, X_val, model_predict_func, feature_names,
                               n_trials, n_epochs,
                               primary_use, option_number, model_name):
    print(f"\n  [FastSHAP OPT] {n_trials} trials, {n_epochs} epochs/trial")
    
    optuna_db = get_optuna_db_path(primary_use, option_number, model_name, "fastshap")
    storage = f"sqlite:///{optuna_db}"
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42),
                                 study_name=f"fastshap_{model_name}", storage=storage, load_if_exists=True)
    
    objective = create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs)
    
    def callback(study, trial):
        save_fastshap_trial(primary_use, option_number, model_name, trial.number, trial.params, trial.value, len(X_train))
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)
    opt_time = time.time() - start
    
    print(f"\n  [FastSHAP] Best: {study.best_value:.6f}")
    return study, opt_time


# ============================
# DATABASE SAVE FUNCTIONS
# ============================

def save_tde_trial(primary_use, option_number, model_name, trial_num, params, loss, n_train):
    conn = sqlite3.connect(TDE_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO tde_hyperparameter_trials
        (primary_use, option_number, model_name, trial_number, hyperparameters, validation_loss, n_training_samples, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, trial_num, json.dumps(params), loss, n_train, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_fastshap_trial(primary_use, option_number, model_name, trial_num, params, loss, n_train):
    conn = sqlite3.connect(FASTSHAP_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO fastshap_hyperparameter_trials
        (primary_use, option_number, model_name, trial_number, hyperparameters, validation_loss, n_training_samples, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, trial_num, json.dumps(params), loss, n_train, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_tde_metadata(primary_use, option_number, model_name, best_params, best_loss, final_loss,
                       n_train, time_steps, n_features, opt_time, train_time, n_trials, path, feature_names):
    conn = sqlite3.connect(TDE_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO tde_model_metadata
        (primary_use, option_number, model_name, best_hyperparameters, best_validation_loss, final_training_loss,
         n_training_samples, time_steps, n_features, optimization_time, training_time, n_trials, explainer_path, feature_names, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, json.dumps(best_params), best_loss, final_loss,
          n_train, time_steps, n_features, opt_time, train_time, n_trials, path, json.dumps(feature_names), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_fastshap_metadata(primary_use, option_number, model_name, best_params, best_loss, final_loss,
                            n_train, input_dim, opt_time, train_time, n_trials, path, feature_names):
    conn = sqlite3.connect(FASTSHAP_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO fastshap_model_metadata
        (primary_use, option_number, model_name, best_hyperparameters, best_validation_loss, final_training_loss,
         n_training_samples, input_dim, optimization_time, training_time, n_trials, explainer_path, feature_names, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, json.dumps(best_params), best_loss, final_loss,
          n_train, input_dim, opt_time, train_time, n_trials, path, json.dumps(feature_names), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_comparison(db_path, primary_use, option_number, model_name, sample_idx, method,
                    fidelity, rel_corr, rel_mse, sparsity, complexity, eff_err, comp_time):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    table = "tde_comparison_results" if "tde" in db_path else "fastshap_comparison_results"
    cursor.execute(f'''
        INSERT OR REPLACE INTO {table}
        (primary_use, option_number, model_name, sample_idx, method, fidelity, reliability_correlation,
         reliability_mse, sparsity, complexity, efficiency_error, computation_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, sample_idx, method, fidelity, rel_corr, rel_mse,
          sparsity, complexity, eff_err, comp_time, datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ============================
# DATA LOADING
# ============================

def load_dataset(primary_use, option_number):
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path="energy_data.db", primary_use=primary_use, option_number=option_number,
        scaled=True, scale_type="both"
    )


def get_datasets():
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('SELECT DISTINCT primary_use, option_number FROM prediction_performance ORDER BY primary_use, option_number', conn)
    conn.close()
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])} for _, r in df.iterrows()]


def get_models(primary_use, option_number):
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('SELECT DISTINCT model_name FROM prediction_performance WHERE primary_use = ? AND option_number = ? ORDER BY model_name',
                            conn, params=(primary_use, option_number))
    conn.close()
    return df['model_name'].tolist()


# ============================
# USER INPUT
# ============================

def get_user_inputs():
    print("\n" + "="*80)
    print("TDE & FastSHAP Training System")
    print("="*80)
    print(f"  Mode: {'DEBUG' if DEBUG_MODE else 'PRODUCTION'}")
    print(f"  TDE: Temporal masking + Attention")
    print(f"  FastSHAP: Simple MLP + Element masking")
    print(f"  Target: FIRST HORIZON ONLY")
    print("="*80)
    
    datasets = get_datasets()
    if not datasets:
        print("\n[ERROR] No trained models!")
        return None
    
    uses = sorted(set(d['primary_use'] for d in datasets))
    print(f"\n[LIST] Primary Uses:")
    for i, u in enumerate(uses):
        print(f"  {i}: {u}")
    
    while True:
        try:
            selected_use = uses[int(input(f"\n--> Select [0-{len(uses)-1}]: ").strip())]
            break
        except:
            print("  Invalid")
    
    use_ds = [d for d in datasets if d['primary_use'] == selected_use]
    selected_ds = use_ds[0] if len(use_ds) == 1 else use_ds[int(input(f"--> Option [0-{len(use_ds)-1}]: ").strip())]
    
    models = get_models(selected_ds['primary_use'], selected_ds['option_number'])
    print(f"\n[LIST] Models:")
    for i, m in enumerate(models):
        print(f"  {i}: {m}")
    
    inp = input(f"\n--> Select or 'all': ").strip().lower()
    selected_models = models if inp == 'all' else [models[int(inp)]]
    
    print("\n[LIST] Explainer Types:")
    print("  0: TDE only")
    print("  1: FastSHAP only")
    print("  2: Both")
    explainer_choice = int(input(f"\n--> Select [0-2]: ").strip() or "2")
    
    window_size = int(input(f"\n--> Window size [{DEFAULT_WINDOW_SIZE}]: ").strip() or DEFAULT_WINDOW_SIZE)
    n_trials = int(input(f"\n--> Trials [{DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS}]: ").strip() or (DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS))
    n_test = int(input(f"\n--> Test samples [5]: ").strip() or "5")
    
    return {
        'primary_use': selected_ds['primary_use'],
        'option_number': selected_ds['option_number'],
        'models': selected_models,
        'explainer_types': ['tde'] if explainer_choice == 0 else (['fastshap'] if explainer_choice == 1 else ['tde', 'fastshap']),
        'window_size': window_size,
        'n_trials': n_trials,
        'n_test_samples': n_test
    }


# ============================
# MAIN TRAINING
# ============================

def train_and_compare(primary_use, option_number, model_name, container,
                      explainer_types, window_size, n_trials, n_test_samples, logger):
    logger.info(f"\n{'='*80}")
    logger.info(f"[MODEL] {model_name}")
    logger.info(f"{'='*80}")
    
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
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
    
    # Prepare data
    X_all = np.concatenate([container.X_train, container.X_val], axis=0)
    frac = DEBUG_TRAINING_FRACTION if DEBUG_MODE else PROD_TRAINING_FRACTION
    np.random.seed(42)
    X_all = X_all[np.random.choice(len(X_all), int(len(X_all) * frac), replace=False)]
    n_val = int(len(X_all) * VALIDATION_SPLIT)
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    
    logger.info(f"  [DATA] Train: {len(X_train)}, Val: {len(X_val)}")
    
    trial_epochs = DEBUG_TRIAL_EPOCHS if DEBUG_MODE else PROD_TRIAL_EPOCHS
    final_epochs = DEBUG_FINAL_EPOCHS if DEBUG_MODE else PROD_FINAL_EPOCHS
    
    explainers = {}
    
    # ===== TDE =====
    if 'tde' in explainer_types:
        logger.info(f"\n  [TDE] Training...")
        study, opt_time = run_tde_optimization(X_train, X_val, predict_first_horizon, container.feature_names,
                                                window_size, n_trials, trial_epochs, primary_use, option_number, model_name)
        
        tde = TemporalDeepExplainer(n_epochs=final_epochs, patience=15, verbose=True,
                                     window_size=window_size, paired_sampling=True, **study.best_params)
        start = time.time()
        final_loss = tde.train(X_train, X_val, predict_first_horizon, container.feature_names)
        train_time = time.time() - start
        
        tde_dir = model_dir / "tde"
        tde_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = tde_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        tde.save(str(tde_dir))
        save_tde_metadata(primary_use, option_number, model_name, study.best_params, study.best_value,
                          final_loss, len(X_train), time_steps, n_features, opt_time, train_time, n_trials,
                          str(tde_dir), container.feature_names)
        
        plot_convergence(tde.history, plots_dir / "convergence.png", "TDE Convergence")
        plot_optuna_importance(study, plots_dir / "param_importance.png")
        
        explainers['TDE'] = lambda x: tde.explain(x, enforce_efficiency=False)
        logger.info(f"  [TDE] Done. Loss: {final_loss:.6f}")
    
    # ===== FastSHAP =====
    if 'fastshap' in explainer_types:
        logger.info(f"\n  [FastSHAP] Training...")
        study, opt_time = run_fastshap_optimization(X_train, X_val, predict_first_horizon, container.feature_names,
                                                     n_trials, trial_epochs, primary_use, option_number, model_name)
        
        # Reconstruct hidden_dims from best params
        best_params = study.best_params.copy()
        n_layers = best_params.pop('n_hidden_layers')
        hidden_dim = best_params.pop('hidden_dim')
        best_params['hidden_dims'] = [hidden_dim] * n_layers
        
        fs = FastSHAPExplainer(n_epochs=final_epochs, patience=15, verbose=True,
                                paired_sampling=True, **best_params)
        start = time.time()
        final_loss = fs.train(X_train, X_val, predict_first_horizon, container.feature_names)
        train_time = time.time() - start
        
        fs_dir = model_dir / "fastshap"
        fs_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = fs_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        fs.save(str(fs_dir))
        save_fastshap_metadata(primary_use, option_number, model_name, best_params, study.best_value,
                                final_loss, len(X_train), time_steps * n_features, opt_time, train_time, n_trials,
                                str(fs_dir), container.feature_names)
        
        plot_convergence(fs.history, plots_dir / "convergence.png", "FastSHAP Convergence")
        plot_optuna_importance(study, plots_dir / "param_importance.png")
        
        explainers['FastSHAP'] = lambda x: fs.explain(x, enforce_efficiency=False)
        logger.info(f"  [FastSHAP] Done. Loss: {final_loss:.6f}")
    
    # ===== Comparison =====
    logger.info(f"\n  [COMPARE] Testing {n_test_samples} samples...")
    
    bg_idx = np.random.choice(len(X_train), min(50, len(X_train)), replace=False)
    background = X_train[bg_idx]
    
    trad = TraditionalSHAPMethods(model, background, time_steps, n_features, device)
    explainers['Gradient_SHAP'] = trad.gradient_shap
    explainers['Deep_SHAP'] = trad.deep_shap
    
    # Use TDE's baseline/base_pred if available, else compute
    if 'tde' in explainer_types:
        baseline_np = tde.baseline.cpu().numpy()
        base_pred = tde.base_pred.item()
    elif 'fastshap' in explainer_types:
        baseline_np = fs.baseline.cpu().numpy().reshape(time_steps, n_features)
        base_pred = fs.base_pred.item()
    else:
        baseline_np = np.median(X_train, axis=0)
        base_pred = predict_first_horizon(baseline_np[np.newaxis])[0]
    
    metrics = ExplainabilityMetrics(model, baseline_np, base_pred, time_steps, n_features, device)
    
    X_test = container.X_test[:n_test_samples]
    
    logger.info(f"\n  {'Method':<15} {'Fidelity':<10} {'Reliability':<12} {'Sparsity':<10} {'Efficiency':<12}")
    logger.info(f"  {'-'*60}")
    
    all_results = {m: [] for m in explainers}
    
    for idx in range(len(X_test)):
        sample = X_test[idx]
        logger.info(f"\n  Sample {idx}:")
        
        for method, func in explainers.items():
            start = time.time()
            try:
                shap_vals = func(sample)
            except Exception as e:
                logger.info(f"    {method:<15} FAILED: {e}")
                continue
            
            comp_time = time.time() - start
            
            if shap_vals is None or shap_vals.shape != (time_steps, n_features):
                logger.info(f"    {method:<15} FAILED (shape)")
                continue
            
            nz = (np.abs(shap_vals) > 1e-6).mean() * 100
            fid = metrics.fidelity(sample, shap_vals)
            rel, rel_mse = metrics.reliability(sample, shap_vals, func)
            spa = metrics.sparsity(shap_vals)
            com = metrics.complexity(shap_vals)
            eff = metrics.efficiency_error(sample, shap_vals)
            
            logger.info(f"    {method:<15} {fid or 0:.4f}     {rel or 0:.4f}       {spa or 0:.1f}%      {eff or 0:.4f}       ({nz:.0f}% nz)")
            
            all_results[method].append({'fidelity': fid, 'reliability': rel, 'sparsity': spa, 'efficiency': eff})
            
            db_path = TDE_DB if method in ['TDE', 'Gradient_SHAP', 'Deep_SHAP'] else FASTSHAP_DB
            save_comparison(db_path, primary_use, option_number, model_name, idx, method, fid, rel, rel_mse, spa, com, eff, comp_time)
            
            # Save heatmaps
            if 'tde' in explainer_types and method in ['TDE', 'Gradient_SHAP', 'Deep_SHAP']:
                hm_path = model_dir / "tde" / "plots" / f"heatmap_s{idx}_{method}.pdf"
                generate_shap_heatmap(shap_vals, container.feature_names, str(hm_path), method)
            if 'fastshap' in explainer_types and method == 'FastSHAP':
                hm_path = model_dir / "fastshap" / "plots" / f"heatmap_s{idx}_{method}.pdf"
                generate_shap_heatmap(shap_vals, container.feature_names, str(hm_path), method)
    
    # Summary
    logger.info(f"\n  {'='*60}")
    logger.info(f"  [SUMMARY]")
    for method, results in all_results.items():
        if results:
            avg_fid = np.mean([r['fidelity'] for r in results if r['fidelity'] is not None])
            avg_rel = np.mean([r['reliability'] for r in results if r['reliability'] is not None])
            avg_spa = np.mean([r['sparsity'] for r in results if r['sparsity'] is not None])
            avg_eff = np.mean([r['efficiency'] for r in results if r['efficiency'] is not None])
            logger.info(f"  {method:<15} Fid: {avg_fid:.4f}, Rel: {avg_rel:.4f}, Spa: {avg_spa:.1f}%, Eff: {avg_eff:.4f}")
    
    del model
    torch.cuda.empty_cache()
    
    return all_results


# ============================
# MAIN FUNCTION
# ============================

def main():
    """Main entry point"""
    
    # Initialize databases
    init_tde_database()
    init_fastshap_database()
    
    # Get user inputs
    config = get_user_inputs()
    if config is None:
        return
    
    primary_use = config['primary_use']
    option_number = config['option_number']
    models = config['models']
    explainer_types = config['explainer_types']
    window_size = config['window_size']
    n_trials = config['n_trials']
    n_test_samples = config['n_test_samples']
    
    # Setup logging
    log_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"explainer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(str(log_path))
    
    logger.info("="*80)
    logger.info("TDE & FastSHAP TRAINING - FIRST HORIZON EXPLAINABILITY")
    logger.info("="*80)
    logger.info(f"  Primary Use: {primary_use}")
    logger.info(f"  Option: {option_number}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Explainer Types: {explainer_types}")
    logger.info(f"  Window Size: {window_size}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Test Samples: {n_test_samples}")
    logger.info(f"  Mode: {'DEBUG' if DEBUG_MODE else 'PRODUCTION'}")
    logger.info(f"  Device: {device}")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"\n[LOAD] Dataset...")
    try:
        container = load_dataset(primary_use, option_number)
        logger.info(f"  Train: {container.X_train.shape}")
        logger.info(f"  Val: {container.X_val.shape}")
        logger.info(f"  Test: {container.X_test.shape}")
        logger.info(f"  Features: {container.feature_names}")
    except Exception as e:
        logger.error(f"  Failed to load dataset: {e}")
        return
    
    # Process each model
    all_model_results = {}
    
    for model_name in models:
        try:
            results = train_and_compare(
                primary_use, option_number, model_name, container,
                explainer_types, window_size, n_trials, n_test_samples, logger
            )
            if results:
                all_model_results[model_name] = results
        except Exception as e:
            logger.error(f"\n[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"  Models processed: {len(all_model_results)}/{len(models)}")
    logger.info(f"  Results saved to: {RESULTS_BASE_DIR}/{primary_use}/option_{option_number}/")
    logger.info(f"  TDE Database: {TDE_DB}")
    logger.info(f"  FastSHAP Database: {FASTSHAP_DB}")
    logger.info(f"  Optuna DBs: {RESULTS_BASE_DIR}/.../tde/optuna_study.db")
    logger.info(f"  Log file: {log_path}")
    logger.info("="*80)
    
    return all_model_results


if __name__ == "__main__":
    main()