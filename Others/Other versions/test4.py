"""
TDE & FastSHAP Training System v5.2

ARCHITECTURE UPDATES:
- TDE: Added Multi-Head Attention after convolutions
- TDE: Added L2 regularization
- TDE: Extended L1 search space [0.0001, 1.0]

MASKING STRATEGIES:
- TDE: Window/Feature masking (temporal structure)
- FastSHAP: Element-wise (time×feature) masking

ARCHITECTURE:
- TDE: Dilated Conv → Multi-Head Attention → Direct Input Connection → Soft Threshold
- FastSHAP: Pure simple MLP

SPARSITY CONTROL FOR TDE:
- Adaptive threshold based on SHAP magnitude distribution
- Target sparsity ~70% to match traditional SHAP methods
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

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

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

DEBUG_MODE = True
DEBUG_TRAINING_FRACTION = 0.15
DEBUG_TRIAL_EPOCHS = 10
DEBUG_FINAL_EPOCHS = 50
DEBUG_N_TRIALS = 10

PROD_TRAINING_FRACTION = 0.30
PROD_TRIAL_EPOCHS = 20
PROD_FINAL_EPOCHS = 100
PROD_N_TRIALS = 30

PAIRED_SAMPLING = True
VALIDATION_SPLIT = 0.20
DEFAULT_WINDOW_SIZE = 6
NOISE_STD = 0.01
EARLY_STOP_PATIENCE = 5

# Target sparsity for TDE (to match traditional SHAP ~70%)
TARGET_SPARSITY = 0.70

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


def get_optuna_db_path(primary_use, option_number, model_name, explainer_type):
    """Get path for Optuna study database"""
    optuna_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / explainer_type
    optuna_dir.mkdir(parents=True, exist_ok=True)
    return str(optuna_dir / "optuna_study.db")


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
# TDE NETWORK v5.2 - WITH MULTI-HEAD ATTENTION
# ============================

class TemporalExplainerNetwork(nn.Module):
    """
    TDE v5.2 Fixed - Attention as gating mechanism, not in main path
    
    Architecture:
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
        
        # ========================================
        # MAIN PATH: Dilated Temporal Convolutions
        # ========================================
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
        
        # ========================================
        # SIDE PATH: Attention Gate (Optional)
        # ========================================
        # This learns which (time, feature) positions are globally important
        # Output is sigmoid → weights in [0, 1]
        # Does NOT use LayerNorm to preserve magnitude information
        if use_attention_gate:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            # Gate projection: maps attention output to [0,1] weights
            self.gate_proj = nn.Sequential(
                nn.Conv1d(hidden_dim, n_features, 1),
                nn.Sigmoid()  # Bounded [0, 1] for gating
            )
        
        # ========================================
        # DIRECT INPUT CONNECTION (Critical)
        # ========================================
        self.input_weight = nn.Parameter(torch.zeros(time_steps, n_features))
        
        # Small initialization
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, baseline=None):
        """
        x: (batch, time_steps, n_features)
        """
        # Main conv path
        h = x.permute(0, 2, 1)  # (batch, feat, time)
        h = self.conv(h)  # (batch, hidden, time)
        
        # Output projection
        conv_out = self.output_proj(h)  # (batch, n_feat, time)
        
        # Attention gating (if enabled)
        if self.use_attention_gate:
            # Attention on conv features
            h_att = h.permute(0, 2, 1)  # (batch, time, hidden)
            attn_out, _ = self.attention(h_att, h_att, h_att)
            attn_out = attn_out.permute(0, 2, 1)  # (batch, hidden, time)
            
            # Generate gate weights [0, 1]
            gate = self.gate_proj(attn_out)  # (batch, n_feat, time)
            
            # Apply gating - preserves conv_out magnitude but modulates by attention
            conv_out = conv_out * gate
        
        conv_out = conv_out.permute(0, 2, 1)  # (batch, time, feat)
        
        # Direct input contribution
        if baseline is not None:
            if baseline.dim() == 2:
                baseline = baseline.unsqueeze(0)
            diff = x - baseline
        else:
            diff = x
        
        input_contrib = diff * torch.tanh(self.input_weight).unsqueeze(0)
        
        # Combine
        output = conv_out + input_contrib
        
        # Soft thresholding for sparsity
        output = torch.sign(output) * torch.relu(torch.abs(output) - self.sparsity_threshold)
        
        return output


# ============================
# FASTSHAP NETWORK - PURE SIMPLE MLP
# ============================

class FastSHAPNetwork(nn.Module):
    """
    Pure simple MLP for FastSHAP (baseline method).
    
    Key differences from TDE:
    - No temporal structure awareness
    - Element-wise (flattened) processing
    - No attention mechanism
    - No direct input connection (relies on proper training)
    
    The network learns: flattened_input -> flattened_SHAP_values
    """
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        
        # Simple MLP - NO LayerNorm, NO BatchNorm (can cause issues)
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Proper initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, baseline=None):
        """
        Forward pass.
        
        Args:
            x: Flattened input (batch, input_dim)
            baseline: Not used in pure MLP (kept for interface compatibility)
        
        Returns:
            Flattened SHAP values (batch, input_dim)
        """
        h = self.mlp(x)
        output = self.output_proj(h)
        return output


# ============================
# TDE TRAINER
# ============================

class TemporalDeepExplainer:
    """
    Trainer for Temporal Deep Explainer.
    
    Training approach:
    - Coalition-based loss (Shapley kernel weighting)
    - Efficiency constraint (SHAP values sum to prediction - baseline)
    - Temporal smoothness regularization
    - L1 + L2 regularization for sparsity
    - Target sparsity loss
    
    Masking strategies:
    - Window: Masks contiguous time windows (preserves temporal structure)
    - Feature: Masks entire features across all time steps
    """
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True, min_lr=1e-6,
                 l1_lambda=0.01, l2_lambda=0.01, smoothness_lambda=0.1, efficiency_lambda=0.1,
                 sparsity_lambda=0.1, target_sparsity=0.70,
                 weight_decay=1e-4, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, optimizer_type='adam', learning_rate=1e-3,
                 window_size=6, paired_sampling=True, samples_per_feature=2,
                 masking_mode='window', **kwargs):
        
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
        
        # Compute baseline as median of training data
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        # Compute base prediction f(baseline)
        base_raw = model_predict_func(self.baseline.unsqueeze(0).cpu().numpy())
        self.base_pred = torch.tensor(
            float(np.atleast_1d(base_raw).flatten()[0]),
            dtype=torch.float32, device=self.device
        )
        
        # Initialize explainer network
        self.explainer = TemporalExplainerNetwork(
            self.time_steps, self.n_features, self.hidden_dim, self.n_conv_layers,
            self.kernel_size, self.dropout_rate, self.sparsity_threshold,
            self.n_attention_heads
        ).to(self.device)
    
    def _compute_shapley_kernel(self, d):
        """Compute Shapley kernel weights for coalition sampling"""
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float32)
        binom_coeffs = torch.tensor(
            [comb(d, int(k.item()), exact=True) for k in k_values],
            device=self.device, dtype=torch.float32
        )
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        probs = weights / weights.sum()
        return weights, probs
    
    def _generate_window_masks(self, batch_size):
        """Generate window-based masks for temporal structure preservation"""
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
    
    def _generate_feature_masks(self, batch_size):
        """Generate feature-based masks (entire feature across all time)"""
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
    
    def _generate_masks(self, batch_size):
        """Generate masks based on selected masking mode"""
        if self.masking_mode == 'window':
            return self._generate_window_masks(batch_size)
        return self._generate_feature_masks(batch_size)
    
    def _get_predictions(self, inputs):
        """Get predictions from black-box model"""
        with torch.no_grad():
            preds = self.model_predict_func(inputs.cpu().numpy())
            return torch.tensor(
                np.atleast_1d(preds).flatten(),
                dtype=torch.float32, device=self.device
            )
    
    def _compute_sparsity(self, phi):
        """Compute current sparsity (fraction of near-zero values)"""
        with torch.no_grad():
            abs_phi = torch.abs(phi)
            max_val = abs_phi.max()
            if max_val < 1e-10:
                return 1.0
            threshold = max_val * 0.01  # 1% of max
            sparsity = (abs_phi < threshold).float().mean()
            return sparsity.item()
    
    def _process_batch(self, X_batch, optimizer):
        """Process single training batch"""
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device)
        
        # Expand batch for multiple samples per feature
        expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks = self._generate_masks(batch_size)
        
        total = masks.size(0)
        repeat = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat, 1, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).repeat(total, 1, 1)
        
        # Apply masking: masked = x * mask + baseline * (1 - mask)
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        
        # Get predictions for masked inputs
        preds_masked = self._get_predictions(masked)
        
        # Get SHAP values from explainer
        phi = self.explainer(X_paired, self.baseline)
        
        # ========================================
        # LOSS COMPUTATION
        # ========================================
        
        # 1. Coalition loss: SHAP values for included features should sum to prediction change
        masked_sum = (masks * phi).sum(dim=(1, 2))
        coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
        
        # 2. Efficiency loss: All SHAP values should sum to f(x) - f(baseline)
        preds_orig = self._get_predictions(X_paired)
        phi_sum = phi.sum(dim=(1, 2))
        eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
        
        # 3. Temporal smoothness: Adjacent time steps should have similar SHAP values
        if phi.size(1) > 1:
            smooth_loss = self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=self.device)
        
        # 4. L1 regularization: Encourages sparsity
        l1_loss = self.l1_lambda * torch.abs(phi).mean()
        
        # 5. L2 regularization: Prevents extreme values
        l2_loss = self.l2_lambda * torch.pow(phi, 2).mean()
        
        # 6. Target sparsity loss: Penalize deviation from target sparsity
        current_sparsity = self._compute_sparsity(phi)
        sparsity_loss = self.sparsity_lambda * (current_sparsity - self.target_sparsity) ** 2
        
        # Total loss
        loss = coalition_loss + eff_loss + smooth_loss + l1_loss + l2_loss + sparsity_loss
        
        # Backpropagation
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
                # Validation loss = efficiency error only
                eff_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                total_loss += eff_err.item()
                n += 1
        
        self.explainer.train()
        return total_loss / max(n, 1)
    
    def train(self, X_train, X_val, model_predict_func, feature_names):
        """
        Train the TDE explainer.
        
        Args:
            X_train: Training data (n_samples, time_steps, n_features)
            X_val: Validation data
            model_predict_func: Function to get predictions from black-box model
            feature_names: List of feature names
        
        Returns:
            Best validation loss
        """
        self._setup(X_train, model_predict_func, feature_names)
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=self.batch_size, shuffle=True
        )
        
        # Setup optimizer
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
                    print(f"    [STOP] e{epoch+1}")
                break
        
        if best_weights:
            self.explainer.load_state_dict(best_weights)
        self.best_loss = best_val
        return best_val
    
    def explain(self, instance):
        """
        Generate SHAP values for a single instance.
        
        Args:
            instance: Input sample (time_steps, n_features) or (1, time_steps, n_features)
        
        Returns:
            SHAP values (time_steps, n_features)
        """
        if self.explainer is None:
            raise ValueError("Not trained")
        
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
    """
    Trainer for FastSHAP (baseline comparison method).
    
    Uses element-wise masking on flattened input.
    Pure MLP architecture without temporal awareness.
    """
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
    
    def _compute_shapley_kernel(self, d):
        """Compute Shapley kernel weights"""
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float32)
        binom_coeffs = torch.tensor(
            [comb(d, int(k.item()), exact=True) for k in k_values],
            device=self.device, dtype=torch.float32
        )
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        probs = weights / weights.sum()
        return weights, probs
    
    def _generate_element_masks(self, batch_size, probs):
        """Generate element-wise masks (each element independently masked)"""
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
    
    def _process_batch(self, X_batch_flat, probs, optimizer):
        """Process single training batch"""
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
        
        # Coalition loss
        masked_sum = (masks * phi).sum(dim=1)
        coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
        
        # Efficiency loss
        preds_orig = self._get_predictions(X_paired)
        phi_sum = phi.sum(dim=1)
        eff_loss = self.efficiency_lambda * ((phi_sum - (preds_orig - self.base_pred)) ** 2).mean()
        
        # L1 regularization
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
        _, probs = self._compute_shapley_kernel(self.input_dim)
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_flat)),
            batch_size=self.batch_size, shuffle=True
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
                ep_loss += self._process_batch(X_batch, probs, optimizer)
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
                    print(f"    [STOP] e{epoch+1}")
                break
        
        if best_weights:
            self.explainer.load_state_dict(best_weights)
        self.best_loss = best_val
        return best_val
    
    def explain(self, instance):
        """Generate SHAP values for a single instance"""
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
# LOADING HELPER FUNCTIONS
# ============================

def load_tde_for_inference(primary_use, option_number, model_name, model_predict_func=None, device_override=None):
    """Load trained TDE for inference"""
    path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / "tde"
    if not path.exists():
        raise FileNotFoundError(f"TDE not found: {path}")
    tde = TemporalDeepExplainer.load(str(path), device_override=device_override)
    if model_predict_func:
        tde.model_predict_func = model_predict_func
    return tde


def load_fastshap_for_inference(primary_use, option_number, model_name, model_predict_func=None, device_override=None):
    """Load trained FastSHAP for inference"""
    path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / "fastshap"
    if not path.exists():
        raise FileNotFoundError(f"FastSHAP not found: {path}")
    fs = FastSHAPExplainer.load(str(path), device_override=device_override)
    if model_predict_func:
        fs.model_predict_func = model_predict_func
    return fs


def load_explainer_with_model(primary_use, option_number, model_name, explainer_type='tde'):
    """Load both model and explainer together"""
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_complete_model(str(model_path), device=device)
    
    config_path = model_dir / "model_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    time_steps, n_features = config['seq_length'], config['n_features']
    
    def predict_first_horizon(X):
        if X.ndim == 2:
            X = X.reshape(-1, time_steps, n_features)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        return pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()
    
    if explainer_type.lower() == 'tde':
        explainer = load_tde_for_inference(primary_use, option_number, model_name, predict_first_horizon)
    else:
        explainer = load_fastshap_for_inference(primary_use, option_number, model_name, predict_first_horizon)
    
    return model, explainer, predict_first_horizon


# ============================
# TRADITIONAL SHAP METHODS
# ============================

class TraditionalSHAPMethods:
    """Wrapper for traditional SHAP methods (Gradient, Deep)"""
    def __init__(self, model, background, time_steps, n_features, device=device):
        self.device = device
        self.time_steps = time_steps
        self.n_features = n_features
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(device)
        self.wrapped_model.eval()
        self.background_tensor = torch.FloatTensor(background).to(device)
    
    def gradient_shap(self, instance):
        """Compute Gradient SHAP values"""
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
        except Exception:
            return None
    
    def deep_shap(self, instance):
        """Compute Deep SHAP values"""
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
        except Exception:
            return None


# ============================
# EXPLAINABILITY METRICS
# ============================

class ExplainabilityMetrics:
    """Compute various explainability metrics"""
    def __init__(self, model, baseline, base_pred, time_steps, n_features, device=device):
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(device)
        self.wrapped_model.eval()
        self.baseline = baseline
        self.base_pred = base_pred
        self.time_steps = time_steps
        self.n_features = n_features
        self.device = device
    
    def _get_prediction(self, x):
        """Get single prediction"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            return self.wrapped_model(x).cpu().numpy().flatten()[0]
    
    def fidelity(self, instance, shap_vals, top_k_pct=10):
        """
        Fidelity: How much does prediction change when top-k important features are masked?
        Higher fidelity = SHAP values identify truly important features
        """
        if shap_vals is None:
            return None
        
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        
        baseline = self.baseline.cpu().numpy() if isinstance(self.baseline, torch.Tensor) else self.baseline
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
        """
        Reliability: How stable are SHAP values under small input perturbations?
        Higher correlation = more stable/reliable explanations
        """
        if shap_vals is None:
            return None, None
        
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        
        perturbed = np.clip(
            instance + np.random.normal(0, noise_std, instance.shape),
            0, 1
        ).astype(np.float32)
        
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
        """
        Sparsity: What fraction of SHAP values are near-zero?
        Higher sparsity = more interpretable (fewer important features)
        """
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals)
        max_val = np.max(abs_shap)
        if max_val == 0:
            return 100.0
        threshold = max_val * threshold_pct / 100
        return float(np.sum(abs_shap < threshold) / abs_shap.size * 100)
    
    def complexity(self, shap_vals):
        """
        Complexity: Entropy of SHAP value distribution
        Lower complexity = simpler explanation
        """
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals).flatten() + 1e-10
        probs = abs_shap / np.sum(abs_shap)
        return float(-np.sum(probs * np.log(probs)))
    
    def efficiency_error(self, instance, shap_vals):
        """
        Efficiency Error: How well do SHAP values sum to f(x) - f(baseline)?
        Lower = better adherence to SHAP efficiency property
        """
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
    """Generate SHAP value heatmap"""
    if shap_vals is None:
        return False
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        vmax = np.max(np.abs(shap_vals))
        im = ax.imshow(shap_vals.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature')
        ax.set_title(f'{method_name} SHAP Values')
        ax.set_yticks(range(len(feature_names[:shap_vals.shape[1]])))
        ax.set_yticklabels(feature_names[:shap_vals.shape[1]], fontsize=8)
        plt.colorbar(im, ax=ax, label='SHAP Value')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception:
        return False


def plot_convergence(history, save_path, title="Convergence"):
    """Plot training convergence"""
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

def create_tde_objective(X_train, X_val, model_predict_func, feature_names, window_size, n_epochs):
    """Create Optuna objective for TDE hyperparameter optimization"""
    def objective(trial):
        params = {
            # Regularization - EXTENDED L1 range
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
            'masking_mode': trial.suggest_categorical('masking_mode', ['window', 'feature']),
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
    
    study_name = f"{explainer_type}_{model_name}_v52"
    
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


# ============================
# USER INPUT
# ============================

def get_user_inputs():
    """Get user configuration through interactive prompts"""
    print("\n" + "="*60)
    print("TDE & FastSHAP Training v5.2")
    print("="*60)
    print("TDE: Dilated Conv → Multi-Head Attention → Direct Input → Soft Threshold")
    print("FastSHAP: Pure MLP with element-wise masking")
    print("="*60)
    
    datasets = get_datasets()
    if not datasets:
        print("No datasets found!")
        return None
    
    uses = sorted(set(d['primary_use'] for d in datasets))
    print(f"\nAvailable Primary Uses:")
    for i, u in enumerate(uses):
        print(f"  {i}: {u}")
    
    selected_use = uses[int(input(f"\n--> Select primary use [0-{len(uses)-1}]: ").strip())]
    use_ds = [d for d in datasets if d['primary_use'] == selected_use]
    
    if len(use_ds) == 1:
        selected_ds = use_ds[0]
    else:
        print(f"\nAvailable Options:")
        for i, d in enumerate(use_ds):
            print(f"  {i}: Option {d['option_number']}")
        selected_ds = use_ds[int(input(f"--> Select option [0-{len(use_ds)-1}]: ").strip())]
    
    models = get_models(selected_ds['primary_use'], selected_ds['option_number'])
    print(f"\nAvailable Models:")
    for i, m in enumerate(models):
        print(f"  {i}: {m}")
    
    inp = input(f"\n--> Select model or 'all': ").strip().lower()
    selected_models = models if inp == 'all' else [models[int(inp)]]
    
    print("\nExplainer Type:")
    print("  0: TDE only")
    print("  1: FastSHAP only")
    print("  2: Both")
    explainer_choice = int(input("--> Select [0-2]: ").strip() or "2")
    
    window_size = int(input(f"\n--> Window size [{DEFAULT_WINDOW_SIZE}]: ").strip() or DEFAULT_WINDOW_SIZE)
    n_trials = int(input(f"--> Number of trials [{DEBUG_N_TRIALS}]: ").strip() or DEBUG_N_TRIALS)
    n_test = int(input("--> Test samples for comparison [5]: ").strip() or "5")
    
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
# MAIN TRAINING FUNCTION
# ============================

def train_and_compare(primary_use, option_number, model_name, container,
                      explainer_types, window_size, n_trials, n_test_samples, logger):
    """Train explainers and compare with traditional SHAP methods"""
    logger.info(f"\n[MODEL] {model_name}")
    
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
    if not model_path.exists():
        logger.error(f"  Model not found: {model_path}")
        return None
    
    # Load model
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
    np.random.seed(42)
    X_all = X_all[np.random.choice(len(X_all), int(len(X_all) * frac), replace=False)]
    n_val = int(len(X_all) * VALIDATION_SPLIT)
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    
    logger.info(f"  Data: Train={len(X_train)} Val={len(X_val)}")
    
    trial_epochs = DEBUG_TRIAL_EPOCHS if DEBUG_MODE else PROD_TRIAL_EPOCHS
    final_epochs = DEBUG_FINAL_EPOCHS if DEBUG_MODE else PROD_FINAL_EPOCHS
    
    explainers = {}
    
    # Train TDE
    if 'tde' in explainer_types:
        logger.info(f"  [TDE] Optimizing hyperparameters...")
        study, opt_time = run_optimization(
            'tde', X_train, X_val, predict_first_horizon, container.feature_names,
            n_trials, trial_epochs, primary_use, option_number, model_name, window_size
        )
        
        logger.info(f"  [TDE] Final training with best params...")
        tde = TemporalDeepExplainer(
            n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE, verbose=True,
            window_size=window_size, paired_sampling=True, **study.best_params
        )
        start = time.time()
        final_loss = tde.train(X_train, X_val, predict_first_horizon, container.feature_names)
        train_time = time.time() - start
        
        tde_dir = model_dir / "tde"
        tde_dir.mkdir(parents=True, exist_ok=True)
        (tde_dir / "plots").mkdir(exist_ok=True)
        
        tde.save(str(tde_dir))
        save_explainer_metadata(
            primary_use, option_number, model_name, 'TDE', study.best_params, study.best_value,
            final_loss, len(X_train), time_steps, n_features, opt_time, train_time, n_trials,
            str(tde_dir), container.feature_names
        )
        plot_convergence(tde.history, tde_dir / "plots" / "convergence.png", "TDE Convergence")
        
        explainers['TDE'] = lambda x: tde.explain(x)
        logger.info(f"  [TDE] Final Loss: {final_loss:.6f}")
    
    # Train FastSHAP
    if 'fastshap' in explainer_types:
        logger.info(f"  [FastSHAP] Optimizing hyperparameters...")
        study, opt_time = run_optimization(
            'fastshap', X_train, X_val, predict_first_horizon, container.feature_names,
            n_trials, trial_epochs, primary_use, option_number, model_name
        )
        
        logger.info(f"  [FastSHAP] Final training with best params...")
        fs = FastSHAPExplainer(
            n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE, verbose=True,
            paired_sampling=True, **study.best_params
        )
        start = time.time()
        final_loss = fs.train(X_train, X_val, predict_first_horizon, container.feature_names)
        train_time = time.time() - start
        
        fs_dir = model_dir / "fastshap"
        fs_dir.mkdir(parents=True, exist_ok=True)
        (fs_dir / "plots").mkdir(exist_ok=True)
        
        fs.save(str(fs_dir))
        save_explainer_metadata(
            primary_use, option_number, model_name, 'FastSHAP', study.best_params, study.best_value,
            final_loss, len(X_train), time_steps, n_features, opt_time, train_time, n_trials,
            str(fs_dir), container.feature_names
        )
        plot_convergence(fs.history, fs_dir / "plots" / "convergence.png", "FastSHAP Convergence")
        
        explainers['FastSHAP'] = lambda x: fs.explain(x)
        logger.info(f"  [FastSHAP] Final Loss: {final_loss:.6f}")
    
    # Comparison with traditional SHAP
    logger.info(f"\n  [COMPARE] Evaluating on {n_test_samples} test samples...")
    
    bg_idx = np.random.choice(len(X_train), min(50, len(X_train)), replace=False)
    background = X_train[bg_idx]
    
    trad = TraditionalSHAPMethods(model, background, time_steps, n_features, device)
    explainers['Gradient_SHAP'] = trad.gradient_shap
    explainers['Deep_SHAP'] = trad.deep_shap
    
    # Get baseline for metrics
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
    
    logger.info(f"\n  {'Method':<12} {'Fidelity':>10} {'Reliability':>12} {'Sparsity':>10} {'Efficiency':>12}")
    logger.info(f"  {'-'*56}")
    
    all_results = {m: [] for m in explainers}
    
    for idx in range(len(X_test)):
        sample = X_test[idx]
        
        for method, func in explainers.items():
            start = time.time()
            try:
                shap_vals = func(sample)
            except Exception:
                continue
            comp_time = time.time() - start
            
            if shap_vals is None or shap_vals.shape != (time_steps, n_features):
                continue
            
            fid = metrics.fidelity(sample, shap_vals)
            rel, rel_mse = metrics.reliability(sample, shap_vals, func)
            spa = metrics.sparsity(shap_vals)
            com = metrics.complexity(shap_vals)
            eff = metrics.efficiency_error(sample, shap_vals)
            
            all_results[method].append({
                'fidelity': fid, 'reliability': rel, 'sparsity': spa, 'efficiency': eff
            })
            
            save_comparison(
                primary_use, option_number, model_name, idx, method,
                fid, rel, rel_mse, spa, com, eff, comp_time
            )
            
            # Generate heatmaps
            if method in ['TDE', 'Gradient_SHAP', 'Deep_SHAP'] and 'tde' in explainer_types:
                generate_shap_heatmap(
                    shap_vals, container.feature_names,
                    model_dir / "tde" / "plots" / f"heatmap_sample{idx}_{method}.pdf", method
                )
            if method == 'FastSHAP' and 'fastshap' in explainer_types:
                generate_shap_heatmap(
                    shap_vals, container.feature_names,
                    model_dir / "fastshap" / "plots" / f"heatmap_sample{idx}_{method}.pdf", method
                )
    
    # Print summary
    for method, results in all_results.items():
        if results:
            avg_fid = np.mean([r['fidelity'] for r in results if r['fidelity'] is not None])
            avg_rel = np.mean([r['reliability'] for r in results if r['reliability'] is not None])
            avg_spa = np.mean([r['sparsity'] for r in results if r['sparsity'] is not None])
            avg_eff = np.mean([r['efficiency'] for r in results if r['efficiency'] is not None])
            logger.info(f"  {method:<12} {avg_fid:>10.4f} {avg_rel:>12.4f} {avg_spa:>9.1f}% {avg_eff:>12.4f}")
    
    del model
    torch.cuda.empty_cache()
    return all_results


# ============================
# MAIN FUNCTION
# ============================

def main():
    """Main entry point"""
    init_database()
    
    config = get_user_inputs()
    if config is None:
        return
    
    primary_use = config['primary_use']
    option_number = config['option_number']
    
    for model_name in config['models']:
        model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = model_dir / f"explainer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger(str(log_path))
        
        logger.info("="*60)
        logger.info(f"TDE & FastSHAP Training v5.2 - {model_name}")
        logger.info("="*60)
        
        try:
            container = load_dataset(primary_use, option_number)
            train_and_compare(
                primary_use, option_number, model_name, container,
                config['explainer_types'], config['window_size'],
                config['n_trials'], config['n_test_samples'], logger
            )
        except Exception as e:
            logger.error(f"  [ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"\nLog saved to: {log_path}")
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()