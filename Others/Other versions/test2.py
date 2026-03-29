"""
TDE (Temporal Deep Explainer) - Extended with XAI Comparison
============================================================
Trains TDE and then compares with traditional SHAP methods:
- Gradient SHAP
- Deep SHAP  
- Sampling SHAP

Comparison metrics:
- Fidelity
- Reliability (correlation under noise perturbation)
- Sparsity
- Complexity
- Efficiency Error
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
from scipy.stats import pearsonr, spearmanr, kendalltau

warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP
import shap

# Local imports
from dl import load_complete_model


# ============================
# CONFIGURATION
# ============================

BENCHMARK_DB = "benchmark_results.db"
TDE_DB = "tde_results.db"
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
PREDICTION_HORIZON = 0  # Which horizon to explain (0 = first)
DEFAULT_WINDOW_SIZE = 6

# Reliability testing
NOISE_STD = 0.01
N_PERTURBATIONS = 5  # Number of perturbations for reliability testing
TOP_K_PCT = 10.0  # Top K% for fidelity and reliability metrics

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")


# ============================
# LOGGING SETUP
# ============================

def setup_logger(log_path):
    """Setup logger - Windows compatible"""
    logger = logging.getLogger('TDE_Comparison')
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
    """Initialize TDE database with comparison tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Hyperparameter trials
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
    
    # Model metadata
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
            n_windows INTEGER,
            window_size INTEGER,
            optimization_time REAL,
            training_time REAL,
            n_trials INTEGER,
            explainer_path TEXT,
            feature_names TEXT,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, model_name)
        )
    ''')
    
    # Comparison results - per sample per method
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tde_comparison_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            method TEXT NOT NULL,
            fidelity REAL,
            reliability_pearson REAL,
            reliability_spearman REAL,
            reliability_kendall REAL,
            reliability_topk_overlap REAL,
            reliability_mse REAL,
            sparsity REAL,
            complexity REAL,
            efficiency_error REAL,
            nonzero_pct REAL,
            computation_time REAL,
            shap_values_json TEXT,
            timestamp TEXT NOT NULL,
            UNIQUE(primary_use, option_number, model_name, sample_idx, method)
        )
    ''')
    
    # Comparison summary - aggregated per method
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tde_comparison_summary (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            method TEXT NOT NULL,
            n_samples INTEGER,
            avg_fidelity REAL,
            std_fidelity REAL,
            avg_reliability_pearson REAL,
            std_reliability_pearson REAL,
            avg_reliability_topk REAL,
            std_reliability_topk REAL,
            avg_sparsity REAL,
            std_sparsity REAL,
            avg_complexity REAL,
            std_complexity REAL,
            avg_efficiency_error REAL,
            std_efficiency_error REAL,
            avg_computation_time REAL,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, model_name, method)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized: {db_path}")


# ============================
# TEMPORAL EXPLAINER NETWORK
# ============================

class TemporalExplainer(nn.Module):
    """
    Neural network for temporal SHAP value estimation
    No Softshrink - sparsity achieved via L1 regularization
    """
    
    def __init__(self, time_steps, n_features, 
                 num_attention_heads=4, 
                 num_conv_layers=2,
                 num_filters=64, 
                 kernel_size=3, 
                 dropout_rate=0.2):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.num_conv_layers = num_conv_layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(n_features, num_filters),
            nn.GELU(),
            nn.LayerNorm(num_filters),
            nn.Dropout(dropout_rate)
        )
        
        # Dynamic convolution layers
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(num_filters, num_filters, kernel_size, 
                                        padding=kernel_size // 2))
            conv_layers.append(nn.GELU())
            conv_layers.append(nn.LayerNorm([num_filters, time_steps]))
            conv_layers.append(nn.Dropout(dropout_rate))
        
        self.time_conv = nn.Sequential(*conv_layers)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            num_filters, num_attention_heads, 
            dropout=dropout_rate, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(num_filters)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(num_filters, num_filters // 2, 1),
            nn.GELU(),
            nn.Conv1d(num_filters // 2, n_features, 1)
        )
        
        # Initialize with small weights
        for m in self.output_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (batch, time_steps, n_features)
        output: (batch, time_steps, n_features) - SHAP values
        """
        # Input embedding
        h = self.input_embed(x)
        
        # Conv1d expects (batch, channels, seq)
        h = h.permute(0, 2, 1)
        
        # Temporal convolutions
        h = self.time_conv(h)
        
        # Multi-head attention
        h_attn = h.permute(0, 2, 1)
        attn_out, _ = self.attention(h_attn, h_attn, h_attn)
        h_attn = self.attn_norm(h_attn + attn_out)
        
        # Back to conv format
        h = h_attn.permute(0, 2, 1)
        
        # Output projection
        output = self.output_proj(h)
        
        # Back to (batch, time, features)
        output = output.permute(0, 2, 1)
        
        return output


# ============================
# TDE CLASS
# ============================

class TemporalDeepExplainer:
    """TDE with improved loss function"""
    
    def __init__(self,
                 n_epochs=100,
                 batch_size=256,
                 patience=15,
                 verbose=True,
                 min_lr=1e-6,
                 
                 l1_lambda=0.01,
                 smoothness_lambda=0.1,
                 efficiency_lambda=0.1,
                 variance_lambda=0.01,
                 weight_decay=1e-4,
                 
                 num_attention_heads=4,
                 num_conv_layers=2,
                 num_filters=64,
                 kernel_size=3,
                 dropout_rate=0.2,
                 
                 optimizer_type='adam',
                 learning_rate=1e-3,
                 
                 window_size=6,
                 paired_sampling=True,
                 samples_per_feature=1,
                 masking_mode='window'):
        
        self.device = device
        
        # Training
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        # Regularization
        self.l1_lambda = l1_lambda
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda
        self.variance_lambda = variance_lambda
        self.weight_decay = weight_decay
        
        # Architecture
        self.num_attention_heads = num_attention_heads
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Optimizer
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        # Masking
        self.window_size = window_size
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        self.masking_mode = masking_mode
        
        # Model state
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.n_windows = None
        self.model_predict_func = None
        
        # Training state
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 
                       'mse': [], 'eff': [], 'nonzero_pct': []}
        
        # Store init params
        self._init_params = {
            'n_epochs': n_epochs, 'batch_size': batch_size,
            'patience': patience, 'verbose': verbose, 'min_lr': min_lr,
            'l1_lambda': l1_lambda, 'smoothness_lambda': smoothness_lambda,
            'efficiency_lambda': efficiency_lambda, 'variance_lambda': variance_lambda,
            'weight_decay': weight_decay,
            'num_attention_heads': num_attention_heads,
            'num_conv_layers': num_conv_layers, 'num_filters': num_filters,
            'kernel_size': kernel_size, 'dropout_rate': dropout_rate,
            'optimizer_type': optimizer_type, 'learning_rate': learning_rate,
            'window_size': window_size, 'paired_sampling': paired_sampling,
            'samples_per_feature': samples_per_feature, 'masking_mode': masking_mode
        }
    
    def _setup(self, X_train, model_predict_func, feature_names):
        """Initialize components"""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.n_windows = (self.time_steps + self.window_size - 1) // self.window_size
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        # Baseline = median of training data
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        # Base prediction
        baseline_np = self.baseline.unsqueeze(0).cpu().numpy()
        base_raw = model_predict_func(baseline_np)
        self.base_pred = torch.tensor(
            float(np.atleast_1d(base_raw).flatten()[0]),
            dtype=torch.float32, device=self.device
        )
        
        # Initialize explainer network
        self.explainer = TemporalExplainer(
            time_steps=self.time_steps,
            n_features=self.n_features,
            num_attention_heads=self.num_attention_heads,
            num_conv_layers=self.num_conv_layers,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        if self.verbose:
            n_params = sum(p.numel() for p in self.explainer.parameters())
            print(f"  [INIT] Explainer: {n_params:,} parameters")
            print(f"  [INIT] Shape: ({self.time_steps}, {self.n_features})")
            print(f"  [INIT] Windows: {self.n_windows} (size={self.window_size})")
            print(f"  [INIT] Masking mode: {self.masking_mode}")
            print(f"  [INIT] Base pred: {self.base_pred.item():.4f}")
    
    def _compute_shapley_kernel(self, d):
        """Calculate Shapley kernel weights"""
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
    
    def _generate_masks(self, batch_size, d, probs):
        """Generate masks based on masking_mode"""
        if self.masking_mode == 'window':
            return self._generate_window_masks(batch_size)
        elif self.masking_mode == 'feature':
            return self._generate_feature_masks(batch_size, probs)
        else:
            return self._generate_element_masks(batch_size, d, probs)
    
    def _generate_window_masks(self, batch_size):
        """Window-based masking"""
        total_samples = batch_size * self.samples_per_feature
        masks = torch.ones(total_samples, self.time_steps, self.n_features, device=self.device)
        
        for i in range(total_samples):
            n_mask = torch.randint(1, max(2, self.n_windows), (1,)).item()
            n_feat_mask = torch.randint(1, self.n_features + 1, (1,)).item()
            feat_indices = torch.randperm(self.n_features, device=self.device)[:n_feat_mask]
            win_indices = torch.randperm(self.n_windows, device=self.device)[:n_mask]
            
            for w in win_indices:
                start_t = int(w) * self.window_size
                end_t = min((int(w) + 1) * self.window_size, self.time_steps)
                for f in feat_indices:
                    masks[i, start_t:end_t, int(f)] = 0.0
        
        if self.paired_sampling:
            paired = 1.0 - masks
            masks = torch.cat([masks, paired], dim=0)
        
        return masks
    
    def _generate_feature_masks(self, batch_size, probs):
        """Feature-level masking"""
        d_features = self.n_features
        _, probs_feature = self._compute_shapley_kernel(d_features)
        
        total_samples = batch_size * self.samples_per_feature
        
        k_indices = torch.multinomial(probs_feature, total_samples, replacement=True)
        k_samples = torch.arange(1, d_features, device=self.device, dtype=torch.int64)[k_indices]
        
        rand = torch.rand(total_samples, d_features, device=self.device)
        sorted_indices = torch.argsort(rand, dim=1)
        
        masks = (sorted_indices < k_samples.unsqueeze(1)).float()
        masks = masks.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        if self.paired_sampling:
            paired = 1.0 - masks
            masks = torch.cat([masks, paired], dim=0)
        
        return masks
    
    def _generate_element_masks(self, batch_size, d, probs):
        """Element-level masking"""
        total_samples = batch_size * self.samples_per_feature
        
        k_indices = torch.multinomial(probs, total_samples, replacement=True)
        k_samples = torch.arange(1, d, device=self.device, dtype=torch.int64)[k_indices]
        
        rand = torch.rand(total_samples, d, device=self.device)
        sorted_indices = torch.argsort(rand, dim=1)
        
        masks = (sorted_indices < k_samples.unsqueeze(1)).float()
        masks = masks.view(-1, self.time_steps, self.n_features)
        
        if self.paired_sampling:
            paired = 1.0 - masks
            masks = torch.cat([masks, paired], dim=0)
        
        return masks
    
    def _get_predictions(self, inputs):
        """Get model predictions"""
        with torch.no_grad():
            inputs_np = inputs.cpu().numpy()
            preds_raw = self.model_predict_func(inputs_np)
            preds = torch.tensor(
                np.atleast_1d(preds_raw).flatten(),
                dtype=torch.float32, device=self.device
            )
        return preds
    
    def _process_batch(self, X_batch, d, probs, optimizer):
        """Process batch with improved loss function"""
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device)
        
        # Generate expanded batch
        expanded_X = X_batch.repeat(self.samples_per_feature, 1, 1)
        
        # Generate masks
        masks = self._generate_masks(batch_size, d, probs)
        
        # Prepare inputs
        total_samples = masks.size(0)
        repeat_factor = max(1, total_samples // (batch_size * self.samples_per_feature))
        
        X_paired = expanded_X.repeat(repeat_factor, 1, 1)[:total_samples]
        baseline_paired = self.baseline.unsqueeze(0).repeat(total_samples, 1, 1)
        
        # Apply masking
        masked_inputs = X_paired * masks + baseline_paired * (1.0 - masks)
        
        # Get model predictions for masked inputs
        preds_masked = self._get_predictions(masked_inputs)
        
        # Get SHAP values from explainer
        phi = self.explainer(X_paired)
        
        # Loss computation
        masked_phi_sum = (masks * phi).sum(dim=(1, 2))
        target_masked = preds_masked - self.base_pred
        mse_loss = ((masked_phi_sum - target_masked) ** 2).mean()
        
        # Efficiency loss
        preds_original = self._get_predictions(X_paired)
        phi_total_sum = phi.sum(dim=(1, 2))
        target_original = preds_original - self.base_pred
        eff_loss = self.efficiency_lambda * ((phi_total_sum - target_original) ** 2).mean()
        
        # Temporal smoothness
        if phi.size(1) > 1:
            smooth_loss = self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=self.device)
        
        # L1 sparsity
        l1_loss = self.l1_lambda * torch.abs(phi).mean()
        
        # Variance encouragement
        phi_std = phi.std(dim=(1, 2)).mean()
        var_loss = self.variance_lambda * torch.exp(-phi_std * 10)
        
        # Total loss
        loss = mse_loss + eff_loss + smooth_loss + l1_loss + var_loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track non-zero percentage
        with torch.no_grad():
            nonzero_pct = (torch.abs(phi) > 1e-6).float().mean().item() * 100
        
        return loss.item(), mse_loss.item(), eff_loss.item(), smooth_loss.item(), nonzero_pct
    
    def _validate(self, X_val):
        """Validation step"""
        self.explainer.eval()
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=self.batch_size, shuffle=False
        )
        
        total_loss = 0.0
        total_nonzero = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                
                phi = self.explainer(X_batch)
                preds = self._get_predictions(X_batch)
                
                phi_sum = phi.sum(dim=(1, 2))
                expected = preds - self.base_pred
                eff_error = ((phi_sum - expected) ** 2).mean()
                
                if phi.size(1) > 1:
                    smooth = ((phi[:, 1:, :] - phi[:, :-1, :]) ** 2).mean()
                else:
                    smooth = torch.tensor(0.0, device=self.device)
                
                val_loss = eff_error + self.smoothness_lambda * smooth
                total_loss += val_loss.item()
                
                nonzero_pct = (torch.abs(phi) > 1e-6).float().mean().item() * 100
                total_nonzero += nonzero_pct
                
                n_batches += 1
        
        self.explainer.train()
        return total_loss / max(n_batches, 1), total_nonzero / max(n_batches, 1)
    
    def train(self, X_train, X_val, model_predict_func, feature_names):
        """Main training procedure"""
        self._setup(X_train, model_predict_func, feature_names)
        
        # Compute d for Shapley kernel
        if self.masking_mode == 'feature':
            d = self.n_features
        elif self.masking_mode == 'window':
            d = self.n_windows * self.n_features
        else:
            d = self.time_steps * self.n_features
        
        weights, probs = self._compute_shapley_kernel(d)
        
        # Data loader
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=self.batch_size, shuffle=True
        )
        
        # Optimizer
        opt_dict = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }
        opt_cls = opt_dict.get(self.optimizer_type, torch.optim.Adam)
        
        optimizer = opt_cls(
            self.explainer.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', 
            patience=self.patience // 2,
            factor=0.5, min_lr=self.min_lr
        )
        
        # Training loop
        best_val = float('inf')
        best_weights = None
        no_improve = 0
        
        if self.verbose:
            print(f"\n  [TRAIN] Starting training: {self.n_epochs} epochs")
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_eff = 0.0
            epoch_smooth = 0.0
            epoch_nonzero = 0.0
            n_batches = 0
            
            for (X_batch,) in loader:
                loss, mse, eff, smooth, nonzero = self._process_batch(X_batch, d, probs, optimizer)
                epoch_loss += loss
                epoch_mse += mse
                epoch_eff += eff
                epoch_smooth += smooth
                epoch_nonzero += nonzero
                n_batches += 1
            
            epoch_loss /= n_batches
            epoch_mse /= n_batches
            epoch_eff /= n_batches
            epoch_smooth /= n_batches
            epoch_nonzero /= n_batches
            
            # Validation
            val_loss, val_nonzero = self._validate(X_val)
            
            # Update scheduler
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            self.history['mse'].append(epoch_mse)
            self.history['eff'].append(epoch_eff)
            self.history['nonzero_pct'].append(epoch_nonzero)
            
            # Early stopping check
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            # Print progress
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | MSE: {epoch_mse:.4f} | "
                      f"Eff: {epoch_eff:.4f} | Val: {val_loss:.4f} | NonZero: {epoch_nonzero:.1f}% | LR: {lr:.1e}")
            
            # Early stopping
            if no_improve >= self.patience and epoch >= self.n_epochs // 3:
                if self.verbose:
                    print(f"  [EARLY STOP] at epoch {epoch+1}")
                break
        
        # Restore best weights
        if best_weights is not None:
            self.explainer.load_state_dict(best_weights)
        
        self.best_loss = best_val
        
        if self.verbose:
            print(f"  [DONE] Best validation loss: {best_val:.6f}")
        
        return best_val
    
    def explain(self, instance, enforce_efficiency=True):
        """Get SHAP values for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not trained")
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        
        instance = instance.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi = self.explainer(instance).cpu().numpy()[0]
        
        # Enforce efficiency constraint
        if enforce_efficiency:
            pred = self._get_predictions(instance).item()
            expected = pred - self.base_pred.item()
            current = np.sum(phi)
            
            if abs(current) > 1e-10:
                phi = phi * (expected / current)
            elif abs(expected) > 1e-10:
                phi = np.full_like(phi, expected / phi.size)
        
        return phi
    
    def explain_batch(self, instances, enforce_efficiency=True):
        """Get SHAP values for multiple instances"""
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
                elif abs(expected) > 1e-10:
                    phi[i] = np.full_like(phi[i], expected / phi[i].size)
        
        return phi
    
    def save(self, path, filename="tde_explainer"):
        """Save explainer"""
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
        
        if self.verbose:
            print(f"  [SAVE] Saved to {save_path}")
        
        return save_path
    
    @classmethod
    def load(cls, path, filename="tde_explainer", device_override=None):
        """Load saved explainer"""
        dev = device_override or device
        load_path = os.path.join(path, f"{filename}.pt")
        
        state = torch.load(load_path, map_location=dev, weights_only=False)
        
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
        
        exp.explainer = TemporalExplainer(
            time_steps=exp.time_steps,
            n_features=exp.n_features,
            num_attention_heads=params.get('num_attention_heads', 4),
            num_conv_layers=params.get('num_conv_layers', 2),
            num_filters=params.get('num_filters', 64),
            kernel_size=params.get('kernel_size', 3),
            dropout_rate=params.get('dropout_rate', 0.2)
        ).to(dev)
        
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        
        return exp


# ============================
# TRADITIONAL SHAP METHODS
# ============================

class TraditionalSHAPMethods:
    """
    Wrapper for traditional SHAP methods: Gradient, Deep, Sampling
    Following the implementation pattern from xai.py
    """
    
    def __init__(self, model, background, time_steps, n_features, device=device, prediction_horizon=0):
        self.model = model
        self.device = device
        self.prediction_horizon = prediction_horizon
        self.time_steps = time_steps
        self.n_features = n_features
        
        if isinstance(background, np.ndarray):
            self.background = background
            self.background_tensor = torch.FloatTensor(background).to(device)
        else:
            self.background = background.cpu().numpy()
            self.background_tensor = background.to(device)
    
    def _model_predict_flat(self, X):
        """Prediction function for flat input (used by Sampling SHAP)"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        if X.ndim == 2:
            X = X.reshape(-1, self.time_steps, self.n_features)
        with torch.no_grad():
            pred = self.model(X)
        # Take mean across prediction horizon
        return pred.cpu().numpy().mean(axis=1).flatten()
    
    def _normalize_shap_values(self, shap_vals, method_name):
        """Normalize SHAP values to expected shape (time_steps, n_features)"""
        try:
            # Convert to numpy if needed
            if isinstance(shap_vals, list):
                if len(shap_vals) > self.prediction_horizon:
                    shap_vals = shap_vals[self.prediction_horizon]
                else:
                    shap_vals = shap_vals[0]
            
            if isinstance(shap_vals, torch.Tensor):
                shap_vals = shap_vals.cpu().numpy()
            
            shap_vals = np.array(shap_vals, dtype=np.float32)
            
            # Handle various shapes
            if shap_vals.ndim == 4:
                if shap_vals.shape[-1] > shap_vals.shape[1]:
                    shap_vals = shap_vals[0, :, :, self.prediction_horizon]
                else:
                    shap_vals = shap_vals[0, self.prediction_horizon, :, :]
                    
            elif shap_vals.ndim == 3:
                if shap_vals.shape[0] == 1:
                    shap_vals = shap_vals[0]
                elif shap_vals.shape[-1] != self.n_features:
                    shap_vals = shap_vals[:, :, self.prediction_horizon]
            
            # Handle 1D case (flattened)
            if shap_vals.ndim == 1:
                expected_size = self.time_steps * self.n_features
                if shap_vals.size == expected_size:
                    shap_vals = shap_vals.reshape(self.time_steps, self.n_features)
                else:
                    return None
            
            # Handle 2D case with wrong shape
            if shap_vals.ndim == 2 and shap_vals.shape != (self.time_steps, self.n_features):
                if shap_vals.shape == (self.n_features, self.time_steps):
                    shap_vals = shap_vals.T
                elif shap_vals.size == self.time_steps * self.n_features:
                    shap_vals = shap_vals.reshape(self.time_steps, self.n_features)
                else:
                    return None
            
            # Final check
            if shap_vals.shape != (self.time_steps, self.n_features):
                return None
            
            return shap_vals.astype(np.float32)
            
        except Exception as e:
            print(f"    [WARN] {method_name} shape normalization failed: {e}")
            return None
    
    def gradient_shap(self, instance):
        """Compute Gradient SHAP"""
        try:
            if isinstance(instance, np.ndarray):
                instance = torch.FloatTensor(instance)
            if instance.ndim == 2:
                instance = instance.unsqueeze(0)
            
            instance = instance.to(self.device)
            
            explainer = shap.GradientExplainer(self.model, self.background_tensor)
            shap_vals = explainer.shap_values(instance)
            
            return self._normalize_shap_values(shap_vals, "Gradient_SHAP")
            
        except Exception as e:
            print(f"    [WARN] Gradient SHAP failed: {e}")
            return None
    
    def deep_shap(self, instance):
        """Compute Deep SHAP"""
        try:
            if isinstance(instance, np.ndarray):
                instance = torch.FloatTensor(instance)
            if instance.ndim == 2:
                instance = instance.unsqueeze(0)
            
            instance = instance.to(self.device)
            
            explainer = shap.DeepExplainer(self.model, self.background_tensor)
            shap_vals = explainer.shap_values(instance, check_additivity=False)
            
            return self._normalize_shap_values(shap_vals, "Deep_SHAP")
            
        except Exception as e:
            print(f"    [WARN] Deep SHAP failed: {e}")
            return None
    
    def sampling_shap(self, instance):
        """Compute Sampling SHAP (from xai.py)"""
        try:
            if isinstance(instance, np.ndarray):
                sample = instance
            else:
                sample = instance.cpu().numpy()
            
            if sample.ndim == 3:
                sample = sample[0]
            
            # Flatten for Sampling SHAP
            bg_flat = self.background.reshape(len(self.background), -1)
            test_flat = sample.reshape(1, -1)
            
            explainer = shap.explainers.Sampling(self._model_predict_flat, bg_flat)
            explanation = explainer(test_flat)
            
            if hasattr(explanation, 'values'):
                shap_vals = explanation.values
            else:
                shap_vals = explanation
            
            return self._normalize_shap_values(shap_vals, "Sampling_SHAP")
            
        except Exception as e:
            print(f"    [WARN] Sampling SHAP failed: {e}")
            return None


# ============================
# METRICS COMPUTATION
# ============================

class ExplainabilityMetrics:
    """Compute explainability metrics"""
    
    def __init__(self, model, baseline, base_pred, time_steps, n_features, device=device):
        self.model = model
        self.baseline = baseline
        self.base_pred = base_pred
        self.time_steps = time_steps
        self.n_features = n_features
        self.device = device
    
    def _get_prediction(self, x):
        """Get model prediction"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()
        
        if pred.ndim > 1:
            pred = pred[:, PREDICTION_HORIZON]
        return pred.flatten()[0] if pred.size > 0 else float(pred)
    
    def fidelity(self, instance, shap_vals, top_k_pct=TOP_K_PCT):
        """Fidelity: prediction change when masking top-K important features"""
        if shap_vals is None:
            return None
        
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        
        if isinstance(self.baseline, torch.Tensor):
            baseline = self.baseline.cpu().numpy()
        else:
            baseline = self.baseline
        if baseline.ndim == 3:
            baseline = baseline[0]
        
        if instance.shape != shap_vals.shape or baseline.shape != instance.shape:
            return None
        
        orig_pred = self._get_prediction(instance)
        
        abs_shap = np.abs(shap_vals)
        total = abs_shap.size
        k = max(1, int(total * top_k_pct / 100))
        
        flat_shap = abs_shap.flatten()
        top_k_indices = np.argsort(flat_shap)[-k:]
        
        masked = instance.copy()
        for flat_idx in top_k_indices:
            t = flat_idx // self.n_features
            f = flat_idx % self.n_features
            if 0 <= t < self.time_steps and 0 <= f < self.n_features:
                masked[t, f] = baseline[t, f]
        
        masked_pred = self._get_prediction(masked)
        
        return float(abs(orig_pred - masked_pred))
    
    def reliability_single(self, instance, shap_vals, shap_func, noise_std=NOISE_STD):
        """Reliability under single perturbation"""
        if shap_vals is None:
            return {}
        
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        if instance.ndim == 3:
            instance = instance[0]
        
        # Add noise
        noise = np.random.normal(0, noise_std, instance.shape)
        perturbed = np.clip(instance + noise, 0, 1).astype(np.float32)
        
        # Get perturbed SHAP
        shap_perturbed = shap_func(perturbed)
        
        if shap_perturbed is None:
            return {}
        
        orig_flat = shap_vals.flatten()
        pert_flat = shap_perturbed.flatten()
        
        mask = np.isfinite(orig_flat) & np.isfinite(pert_flat)
        if np.sum(mask) < 10:
            return {}
        
        # Compute metrics
        try:
            pearson_corr, _ = pearsonr(orig_flat[mask], pert_flat[mask])
        except:
            pearson_corr = None
        
        try:
            spearman_corr, _ = spearmanr(orig_flat[mask], pert_flat[mask])
        except:
            spearman_corr = None
        
        try:
            kendall_corr, _ = kendalltau(orig_flat[mask], pert_flat[mask])
        except:
            kendall_corr = None
        
        # Top-K overlap
        total = len(orig_flat)
        k = max(1, int(total * TOP_K_PCT / 100))
        
        top_k_orig = set(np.argsort(np.abs(orig_flat))[-k:])
        top_k_pert = set(np.argsort(np.abs(pert_flat))[-k:])
        overlap = len(top_k_orig & top_k_pert) / k * 100
        
        # MSE
        mse = np.mean((orig_flat[mask] - pert_flat[mask]) ** 2)
        
        return {
            'pearson': float(pearson_corr) if pearson_corr is not None and np.isfinite(pearson_corr) else None,
            'spearman': float(spearman_corr) if spearman_corr is not None and np.isfinite(spearman_corr) else None,
            'kendall': float(kendall_corr) if kendall_corr is not None and np.isfinite(kendall_corr) else None,
            'topk_overlap': float(overlap),
            'mse': float(mse)
        }
    
    def reliability(self, instance, shap_vals, shap_func, n_perturbations=N_PERTURBATIONS, noise_std=NOISE_STD):
        """Reliability averaged over multiple perturbations"""
        if shap_vals is None:
            return {}
        
        all_metrics = []
        for _ in range(n_perturbations):
            metrics = self.reliability_single(instance, shap_vals, shap_func, noise_std)
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate
        result = {}
        for key in ['pearson', 'spearman', 'kendall', 'topk_overlap', 'mse']:
            values = [m[key] for m in all_metrics if m.get(key) is not None]
            if values:
                result[key] = float(np.mean(values))
            else:
                result[key] = None
        
        return result
    
    def sparsity(self, shap_vals, threshold_pct=1):
        """Sparsity: percentage of near-zero SHAP values"""
        if shap_vals is None:
            return None
        
        abs_shap = np.abs(shap_vals)
        max_val = np.max(abs_shap)
        
        if max_val == 0:
            return 100.0
        
        threshold = max_val * threshold_pct / 100
        near_zero = np.sum(abs_shap < threshold)
        
        return float(near_zero / abs_shap.size * 100)
    
    def complexity(self, shap_vals):
        """Complexity: entropy of SHAP distribution"""
        if shap_vals is None:
            return None
        
        abs_shap = np.abs(shap_vals).flatten() + 1e-10
        probs = abs_shap / np.sum(abs_shap)
        
        return float(-np.sum(probs * np.log(probs)))
    
    def efficiency_error(self, instance, shap_vals):
        """Efficiency error: |sum(SHAP) - (pred - baseline)| / |pred - baseline|"""
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
    
    def nonzero_percentage(self, shap_vals, threshold=1e-6):
        """Percentage of non-zero SHAP values"""
        if shap_vals is None:
            return None
        
        return float((np.abs(shap_vals) > threshold).mean() * 100)


# ============================
# VISUALIZATION
# ============================

def generate_shap_heatmap(shap_vals, sample_data, feature_names, output_path, method_name, metrics_text=None):
    """Generate SHAP heatmap with optional metrics overlay"""
    try:
        if shap_vals is None:
            return False
        
        seq_len, n_features = shap_vals.shape
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        vmax = np.max(np.abs(shap_vals))
        im = ax.imshow(shap_vals.T, aspect='auto', cmap='RdBu_r', 
                       vmin=-vmax, vmax=vmax)
        
        # Labels
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'{method_name} SHAP Values', fontsize=14)
        
        # Feature names on y-axis
        ax.set_yticks(range(len(feature_names[:n_features])))
        ax.set_yticklabels(feature_names[:n_features], fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SHAP Value', fontsize=10)
        
        # Add metrics text if provided
        if metrics_text:
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"    [WARN] Heatmap failed: {e}")
        return False


def plot_convergence(history, save_path):
    """Plot training convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE and Efficiency
    if 'mse' in history and 'eff' in history:
        axes[0, 1].plot(epochs, history['mse'], 'b-', label='MSE', alpha=0.7)
        axes[0, 1].plot(epochs, history['eff'], 'g-', label='Efficiency', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Component')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Non-zero percentage
    if 'nonzero_pct' in history:
        axes[1, 0].plot(epochs, history['nonzero_pct'], 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Non-Zero %')
        axes[1, 0].set_title('Non-Zero SHAP Values')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
    
    # Learning rate
    axes[1, 1].plot(epochs, history['lr'], 'orange', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_summary(summary_data, save_path):
    """Plot comparison summary across methods"""
    methods = list(summary_data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = [
        ('fidelity', 'Fidelity (Higher = Better)', axes[0, 0]),
        ('reliability_pearson', 'Reliability (Pearson)', axes[0, 1]),
        ('reliability_topk', 'Top-K Overlap (%)', axes[0, 2]),
        ('sparsity', 'Sparsity (%)', axes[1, 0]),
        ('complexity', 'Complexity (Lower = Simpler)', axes[1, 1]),
        ('computation_time', 'Computation Time (s)', axes[1, 2])
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for metric_key, title, ax in metrics:
        values = []
        errors = []
        
        for method in methods:
            data = summary_data[method]
            avg_key = f'avg_{metric_key}'
            std_key = f'std_{metric_key}'
            
            if avg_key in data and data[avg_key] is not None:
                values.append(data[avg_key])
                errors.append(data.get(std_key, 0) or 0)
            else:
                values.append(0)
                errors.append(0)
        
        x = np.arange(len(methods))
        bars = ax.bar(x, values, yerr=errors, capsize=5, color=colors[:len(methods)], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(shap_vals, feature_names, save_path, method_name="TDE"):
    """Plot feature importance"""
    if shap_vals.ndim == 3:
        mean_abs = np.mean(np.abs(shap_vals), axis=(0, 1))
    else:
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
    
    idx = np.argsort(mean_abs)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_names)), mean_abs[idx])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(f'Feature Importance ({method_name})')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================
# DATABASE FUNCTIONS
# ============================

def save_trial(primary_use, option_number, model_name, trial_num, params, loss, n_train):
    """Save trial to database"""
    conn = sqlite3.connect(TDE_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO tde_hyperparameter_trials
        (primary_use, option_number, model_name, trial_number, hyperparameters,
         validation_loss, n_training_samples, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, trial_num,
          json.dumps(params), loss, n_train, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def save_comparison_result(primary_use, option_number, model_name, sample_idx, method,
                           fidelity, reliability, sparsity, complexity, efficiency_error,
                           nonzero_pct, comp_time, shap_vals=None):
    """Save individual comparison result"""
    conn = sqlite3.connect(TDE_DB)
    cursor = conn.cursor()
    
    shap_json = json.dumps(shap_vals.tolist()) if shap_vals is not None else None
    
    cursor.execute('''
        INSERT OR REPLACE INTO tde_comparison_results
        (primary_use, option_number, model_name, sample_idx, method,
         fidelity, reliability_pearson, reliability_spearman, reliability_kendall,
         reliability_topk_overlap, reliability_mse, sparsity, complexity,
         efficiency_error, nonzero_pct, computation_time, shap_values_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, sample_idx, method,
          fidelity,
          reliability.get('pearson') if reliability else None,
          reliability.get('spearman') if reliability else None,
          reliability.get('kendall') if reliability else None,
          reliability.get('topk_overlap') if reliability else None,
          reliability.get('mse') if reliability else None,
          sparsity, complexity, efficiency_error, nonzero_pct, comp_time,
          shap_json, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def save_comparison_summary(primary_use, option_number, model_name, method, summary_data):
    """Save aggregated comparison summary"""
    conn = sqlite3.connect(TDE_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO tde_comparison_summary
        (primary_use, option_number, model_name, method, n_samples,
         avg_fidelity, std_fidelity, avg_reliability_pearson, std_reliability_pearson,
         avg_reliability_topk, std_reliability_topk, avg_sparsity, std_sparsity,
         avg_complexity, std_complexity, avg_efficiency_error, std_efficiency_error,
         avg_computation_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (primary_use, option_number, model_name, method,
          summary_data.get('n_samples', 0),
          summary_data.get('avg_fidelity'),
          summary_data.get('std_fidelity'),
          summary_data.get('avg_reliability_pearson'),
          summary_data.get('std_reliability_pearson'),
          summary_data.get('avg_reliability_topk'),
          summary_data.get('std_reliability_topk'),
          summary_data.get('avg_sparsity'),
          summary_data.get('std_sparsity'),
          summary_data.get('avg_complexity'),
          summary_data.get('std_complexity'),
          summary_data.get('avg_efficiency_error'),
          summary_data.get('std_efficiency_error'),
          summary_data.get('avg_computation_time'),
          datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


# ============================
# DATA LOADING
# ============================

def load_dataset(primary_use, option_number):
    """Load dataset"""
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path="energy_data.db",
        primary_use=primary_use,
        option_number=option_number,
        scaled=True,
        scale_type="both"
    )


def get_datasets():
    """Get available datasets"""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('''
        SELECT DISTINCT primary_use, option_number
        FROM prediction_performance
        ORDER BY primary_use, option_number
    ''', conn)
    conn.close()
    
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])}
            for _, r in df.iterrows()]


def get_models(primary_use, option_number):
    """Get available models"""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('''
        SELECT DISTINCT model_name
        FROM prediction_performance
        WHERE primary_use = ? AND option_number = ?
        ORDER BY model_name
    ''', conn, params=(primary_use, option_number))
    conn.close()
    return df['model_name'].tolist()


# ============================
# HYPERPARAMETER OPTIMIZATION
# ============================

def create_objective(X_train, X_val, model_predict_func, feature_names, window_size, n_epochs):
    """Create Optuna objective"""
    
    def objective(trial):
        params = {
            'l1_lambda': trial.suggest_float('l1_lambda', 0.001, 0.1, log=True),
            'smoothness_lambda': trial.suggest_float('smoothness_lambda', 0.01, 0.3),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.3),
            'variance_lambda': trial.suggest_float('variance_lambda', 0.001, 0.05, log=True),
            
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4]),
            'num_conv_layers': trial.suggest_int('num_conv_layers', 1, 3),
            'num_filters': trial.suggest_categorical('num_filters', [32, 64, 128]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            
            'masking_mode': trial.suggest_categorical('masking_mode', ['window', 'feature']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 1, 2),
        }
        
        try:
            tde = TemporalDeepExplainer(
                n_epochs=n_epochs,
                patience=8,
                verbose=False,
                window_size=window_size,
                paired_sampling=True,
                **params
            )
            
            val_loss = tde.train(X_train, X_val, model_predict_func, feature_names)
            
            del tde
            torch.cuda.empty_cache()
            
            return val_loss
            
        except Exception as e:
            print(f"    Trial failed: {e}")
            return float('inf')
    
    return objective


def run_optimization(X_train, X_val, model_predict_func, feature_names,
                     window_size, n_trials, n_epochs,
                     primary_use, option_number, model_name):
    """Run hyperparameter optimization"""
    print(f"\n  [OPT] {n_trials} trials, {n_epochs} epochs/trial")
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    objective = create_objective(X_train, X_val, model_predict_func, feature_names,
                                 window_size, n_epochs)
    
    def callback(study, trial):
        save_trial(primary_use, option_number, model_name, trial.number,
                  trial.params, trial.value, len(X_train))
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)
    opt_time = time.time() - start
    
    print(f"\n  [OK] Best: {study.best_value:.6f}")
    
    return study, opt_time


# ============================
# USER INPUT
# ============================

def get_user_inputs():
    """Get configuration from user"""
    print("\n" + "="*80)
    print("TDE TRAINING WITH XAI COMPARISON")
    print("="*80)
    print(f"  Mode: {'DEBUG' if DEBUG_MODE else 'PRODUCTION'}")
    print(f"  Methods: TDE, Gradient SHAP, Deep SHAP, Sampling SHAP")
    print("="*80)
    
    datasets = get_datasets()
    if not datasets:
        print("\n[ERROR] No trained models found!")
        return None
    
    uses = sorted(set(d['primary_use'] for d in datasets))
    
    print(f"\n[LIST] Primary Uses:")
    for i, u in enumerate(uses):
        print(f"  {i}: {u}")
    
    while True:
        inp = input(f"\n--> Select primary use [0-{len(uses)-1}]: ").strip()
        try:
            selected_use = uses[int(inp)]
            break
        except:
            print("  Invalid selection")
    
    use_ds = [d for d in datasets if d['primary_use'] == selected_use]
    
    if len(use_ds) == 1:
        selected_ds = use_ds[0]
    else:
        print(f"\n[LIST] Options:")
        for i, d in enumerate(use_ds):
            print(f"  {i}: Option {d['option_number']}")
        
        while True:
            inp = input(f"\n--> Select option: ").strip()
            try:
                selected_ds = use_ds[int(inp)]
                break
            except:
                print("  Invalid selection")
    
    models = get_models(selected_ds['primary_use'], selected_ds['option_number'])
    
    print(f"\n[LIST] Models:")
    for i, m in enumerate(models):
        print(f"  {i}: {m}")
    
    while True:
        inp = input(f"\n--> Select model or 'all': ").strip().lower()
        if inp == 'all':
            selected_models = models
            break
        try:
            selected_models = [models[int(inp)]]
            break
        except:
            print("  Invalid selection")
    
    inp = input(f"\n--> Window size [{DEFAULT_WINDOW_SIZE}]: ").strip()
    window_size = int(inp) if inp else DEFAULT_WINDOW_SIZE
    
    default_trials = DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS
    inp = input(f"\n--> Optimization trials [{default_trials}]: ").strip()
    n_trials = int(inp) if inp else default_trials
    
    inp = input(f"\n--> Test samples for comparison [5]: ").strip()
    n_test_samples = int(inp) if inp else 5
    
    print("\n" + "="*80)
    print(f"  Dataset: {selected_ds['primary_use']} - Option {selected_ds['option_number']}")
    print(f"  Models: {', '.join(selected_models)}")
    print(f"  Window: {window_size}, Trials: {n_trials}, Test Samples: {n_test_samples}")
    print("="*80)
    
    return {
        'primary_use': selected_ds['primary_use'],
        'option_number': selected_ds['option_number'],
        'models': selected_models,
        'window_size': window_size,
        'n_trials': n_trials,
        'n_test_samples': n_test_samples
    }


# ============================
# MAIN TRAINING AND COMPARISON
# ============================

def train_and_compare(primary_use, option_number, model_name, container,
                      window_size, n_trials, n_test_samples, logger):
    """
    Train TDE and compare with traditional SHAP methods
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[MODEL] {model_name}")
    logger.info(f"{'='*80}")
    
    # ===== STEP 1: LOAD MODEL =====
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
    if not model_path.exists():
        logger.error(f"  Model not found: {model_path}")
        return None
    
    model = load_complete_model(str(model_path), device=device)
    
    # Get prediction info
    test_input = torch.FloatTensor(container.X_test[:1]).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    logger.info(f"  [INFO] Model output shape: {test_output.shape}")
    
    def predict(X):
        if X.ndim == 2:
            X = X.reshape(-1, container.X_train.shape[1], container.X_train.shape[2])
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        if pred.ndim > 1 and pred.shape[1] > PREDICTION_HORIZON:
            return pred[:, PREDICTION_HORIZON]
        return pred.flatten()
    
    # ===== STEP 2: PREPARE DATA =====
    X_all = np.concatenate([container.X_train, container.X_val], axis=0)
    
    frac = DEBUG_TRAINING_FRACTION if DEBUG_MODE else PROD_TRAINING_FRACTION
    n_samples = int(len(X_all) * frac)
    np.random.seed(42)
    X_all = X_all[np.random.choice(len(X_all), n_samples, replace=False)]
    
    n_val = int(len(X_all) * VALIDATION_SPLIT)
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    
    time_steps = container.X_train.shape[1]
    n_features = container.X_train.shape[2]
    
    logger.info(f"  [DATA] Train: {len(X_train)}, Val: {len(X_val)}, Shape: ({time_steps}, {n_features})")
    
    # ===== STEP 3: TRAIN TDE =====
    logger.info(f"\n  [TRAIN] Training TDE...")
    
    trial_epochs = DEBUG_TRIAL_EPOCHS if DEBUG_MODE else PROD_TRIAL_EPOCHS
    
    study, opt_time = run_optimization(
        X_train, X_val, predict, container.feature_names,
        window_size, n_trials, trial_epochs,
        primary_use, option_number, model_name
    )
    
    final_epochs = DEBUG_FINAL_EPOCHS if DEBUG_MODE else PROD_FINAL_EPOCHS
    
    logger.info(f"\n  [FINAL] Training with best params for {final_epochs} epochs...")
    logger.info(f"  Best params: {study.best_params}")
    
    tde = TemporalDeepExplainer(
        n_epochs=final_epochs,
        patience=15,
        verbose=True,
        window_size=window_size,
        paired_sampling=True,
        **study.best_params
    )
    
    start = time.time()
    final_loss = tde.train(X_train, X_val, predict, container.feature_names)
    train_time = time.time() - start
    
    logger.info(f"\n  [OK] TDE trained. Loss: {final_loss:.6f}, Time: {train_time:.1f}s")
    
    # Save TDE
    tde_dir = model_dir / "tde"
    tde_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = tde_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    tde.save(str(tde_dir), "tde_explainer")
    plot_convergence(tde.history, plots_dir / "convergence.png")
    
    # ===== STEP 4: SETUP COMPARISON =====
    logger.info(f"\n  [COMPARE] Setting up XAI comparison...")
    
    # Background for traditional SHAP
    bg_indices = np.random.choice(len(X_train), min(50, len(X_train)), replace=False)
    background = X_train[bg_indices]
    
    # Traditional SHAP methods
    trad_shap = TraditionalSHAPMethods(
        model, background, time_steps, n_features, 
        device, prediction_horizon=PREDICTION_HORIZON
    )
    
    # Metrics calculator
    baseline_np = tde.baseline.cpu().numpy()
    base_pred_val = tde.base_pred.item()
    
    metrics_calc = ExplainabilityMetrics(
        model, baseline_np, base_pred_val, 
        time_steps, n_features, device
    )
    
    # Test samples
    X_test = container.X_test[:n_test_samples]
    
    # Define methods
    methods = {
        'TDE': lambda x: tde.explain(x, enforce_efficiency=True),
        'Gradient_SHAP': trad_shap.gradient_shap,
        'Deep_SHAP': trad_shap.deep_shap,
        'Sampling_SHAP': trad_shap.sampling_shap
    }
    
    # ===== STEP 5: RUN COMPARISON =====
    logger.info(f"\n  [COMPARE] Running comparison on {len(X_test)} samples...")
    logger.info(f"  {'Method':<15} {'Fidelity':<10} {'Rel.Corr':<10} {'TopK%':<10} {'Sparsity':<10} {'Time':<10}")
    logger.info(f"  {'-'*65}")
    
    all_results = {m: [] for m in methods.keys()}
    
    for sample_idx in range(len(X_test)):
        sample = X_test[sample_idx]
        
        logger.info(f"\n  Sample {sample_idx}:")
        
        for method_name, method_func in methods.items():
            start_time = time.time()
            
            try:
                shap_vals = method_func(sample)
            except Exception as e:
                logger.info(f"    {method_name:<15} FAILED: {e}")
                continue
            
            comp_time = time.time() - start_time
            
            if shap_vals is None:
                logger.info(f"    {method_name:<15} FAILED (returned None)")
                continue
            
            # Check shape
            expected_shape = (time_steps, n_features)
            if shap_vals.shape != expected_shape:
                logger.info(f"    {method_name:<15} Shape mismatch: {shap_vals.shape} vs {expected_shape}")
                continue
            
            # Compute metrics
            try:
                fidelity = metrics_calc.fidelity(sample, shap_vals)
                reliability = metrics_calc.reliability(sample, shap_vals, method_func)
                sparsity = metrics_calc.sparsity(shap_vals)
                complexity = metrics_calc.complexity(shap_vals)
                eff_error = metrics_calc.efficiency_error(sample, shap_vals)
                nonzero_pct = metrics_calc.nonzero_percentage(shap_vals)
            except Exception as e:
                logger.error(f"    {method_name:<15} Metrics failed: {e}")
                continue
            
            # Format output
            fid_str = f"{fidelity:.4f}" if fidelity is not None else "N/A"
            rel_str = f"{reliability.get('pearson', 0):.4f}" if reliability.get('pearson') is not None else "N/A"
            topk_str = f"{reliability.get('topk_overlap', 0):.1f}%" if reliability.get('topk_overlap') is not None else "N/A"
            spa_str = f"{sparsity:.1f}%" if sparsity is not None else "N/A"
            time_str = f"{comp_time:.2f}s"
            
            logger.info(f"    {method_name:<15} {fid_str:<10} {rel_str:<10} {topk_str:<10} {spa_str:<10} {time_str:<10}")
            
            # Store results
            all_results[method_name].append({
                'fidelity': fidelity,
                'reliability': reliability,
                'sparsity': sparsity,
                'complexity': complexity,
                'efficiency_error': eff_error,
                'nonzero_pct': nonzero_pct,
                'computation_time': comp_time
            })
            
            # Save to database
            save_comparison_result(
                primary_use, option_number, model_name, sample_idx, method_name,
                fidelity, reliability, sparsity, complexity, eff_error,
                nonzero_pct, comp_time, shap_vals
            )
            
            # Generate heatmap
            metrics_text = f"Fidelity: {fid_str}\nReliability: {rel_str}\nSparsity: {spa_str}"
            heatmap_path = plots_dir / f"heatmap_sample{sample_idx}_{method_name}.pdf"
            generate_shap_heatmap(shap_vals, sample, container.feature_names,
                                 str(heatmap_path), method_name, metrics_text)
    
    # ===== STEP 6: COMPUTE AND SAVE SUMMARY =====
    logger.info(f"\n  {'='*65}")
    logger.info(f"  [SUMMARY]")
    logger.info(f"  {'='*65}")
    
    summary_data = {}
    
    for method_name, results in all_results.items():
        if len(results) == 0:
            continue
        
        # Aggregate metrics
        fid_vals = [r['fidelity'] for r in results if r['fidelity'] is not None]
        rel_vals = [r['reliability'].get('pearson') for r in results if r['reliability'] and r['reliability'].get('pearson') is not None]
        topk_vals = [r['reliability'].get('topk_overlap') for r in results if r['reliability'] and r['reliability'].get('topk_overlap') is not None]
        spa_vals = [r['sparsity'] for r in results if r['sparsity'] is not None]
        com_vals = [r['complexity'] for r in results if r['complexity'] is not None]
        eff_vals = [r['efficiency_error'] for r in results if r['efficiency_error'] is not None]
        time_vals = [r['computation_time'] for r in results]
        
        summary = {
            'n_samples': len(results),
            'avg_fidelity': np.mean(fid_vals) if fid_vals else None,
            'std_fidelity': np.std(fid_vals) if len(fid_vals) > 1 else None,
            'avg_reliability_pearson': np.mean(rel_vals) if rel_vals else None,
            'std_reliability_pearson': np.std(rel_vals) if len(rel_vals) > 1 else None,
            'avg_reliability_topk': np.mean(topk_vals) if topk_vals else None,
            'std_reliability_topk': np.std(topk_vals) if len(topk_vals) > 1 else None,
            'avg_sparsity': np.mean(spa_vals) if spa_vals else None,
            'std_sparsity': np.std(spa_vals) if len(spa_vals) > 1 else None,
            'avg_complexity': np.mean(com_vals) if com_vals else None,
            'std_complexity': np.std(com_vals) if len(com_vals) > 1 else None,
            'avg_efficiency_error': np.mean(eff_vals) if eff_vals else None,
            'std_efficiency_error': np.std(eff_vals) if len(eff_vals) > 1 else None,
            'avg_computation_time': np.mean(time_vals) if time_vals else None
        }
        
        summary_data[method_name] = summary
        
        # Save to database
        save_comparison_summary(primary_use, option_number, model_name, method_name, summary)
        
        # Log summary
        avg_fid = summary['avg_fidelity']
        avg_rel = summary['avg_reliability_pearson']
        avg_topk = summary['avg_reliability_topk']
        avg_spa = summary['avg_sparsity']
        avg_time = summary['avg_computation_time']
        
        logger.info(f"  {method_name:<15} Fidelity: {avg_fid:.4f if avg_fid else 0:.4f}, "
                   f"Reliability: {avg_rel:.4f if avg_rel else 0:.4f}, "
                   f"TopK: {avg_topk:.1f if avg_topk else 0:.1f}%, "
                   f"Sparsity: {avg_spa:.1f if avg_spa else 0:.1f}%, "
                   f"Time: {avg_time:.2f if avg_time else 0:.2f}s")
    
    # Generate summary plot
    if summary_data:
        plot_comparison_summary(summary_data, plots_dir / "comparison_summary.png")
    
    # Feature importance plot (TDE)
    all_shap = tde.explain_batch(X_test, enforce_efficiency=True)
    plot_feature_importance(all_shap, container.feature_names, 
                           plots_dir / "feature_importance_tde.png", "TDE")
    
    logger.info(f"\n  [SAVE] Results saved to: {tde_dir}")
    logger.info(f"  [SAVE] Database: {TDE_DB}")
    
    # Cleanup
    del model, tde
    torch.cuda.empty_cache()
    
    return str(tde_dir)


# ============================
# MAIN FUNCTION
# ============================

def main():
    """Main entry point"""
    
    # Initialize database
    init_tde_database()
    
    # Get user configuration
    config = get_user_inputs()
    if config is None:
        return
    
    if input("\n--> Proceed? (y/n): ").strip().lower() not in ['y', 'yes']:
        print("Cancelled")
        return
    
    # Setup logger
    log_dir = Path(RESULTS_BASE_DIR) / config['primary_use'] / f"option_{config['option_number']}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"tde_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(str(log_path))
    
    logger.info("="*80)
    logger.info("TDE TRAINING WITH XAI COMPARISON REPORT")
    logger.info("="*80)
    logger.info(f"Date: {datetime.now()}")
    logger.info(f"Dataset: {config['primary_use']} - Option {config['option_number']}")
    logger.info(f"Methods: TDE, Gradient SHAP, Deep SHAP, Sampling SHAP")
    logger.info(f"Metrics: Fidelity, Reliability (Pearson, TopK), Sparsity, Complexity")
    logger.info("="*80)
    
    # Load dataset
    container = load_dataset(config['primary_use'], config['option_number'])
    logger.info(f"\n[DATA] Shape: {container.X_train.shape}")
    
    # Process each model
    results = []
    for i, model_name in enumerate(config['models']):
        logger.info(f"\n[{i+1}/{len(config['models'])}] Processing {model_name}")
        
        try:
            path = train_and_compare(
                config['primary_use'], config['option_number'],
                model_name, container,
                config['window_size'], config['n_trials'],
                config['n_test_samples'], logger
            )
            results.append({'model': model_name, 'status': 'ok', 'path': path})
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'model': model_name, 'status': 'failed'})
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("[COMPLETED]")
    for r in results:
        status = "OK" if r['status'] == 'ok' else "FAILED"
        logger.info(f"  [{status}] {r['model']}")
    logger.info(f"Log: {log_path}")
    logger.info(f"Database: {TDE_DB}")
    logger.info("="*80)
    
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()