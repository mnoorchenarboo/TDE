# ============================
# LIBRARY IMPORTS
# ============================

import os
import numpy as np
import warnings
from scipy.special import comb

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

import shap

# ============================
# DEVICE CONFIGURATION
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================
# TDE NETWORK ARCHITECTURE
# ============================
class TemporalExplainerNetwork(nn.Module):
    """
    Temporal Deep Explainer (TDE) Network — paper-aligned implementation.

    Architecture (§3):
      1. Temporal Convolution Block (§3.2):
           Conv1D → GELU → LayerNorm → Dropout  →  H_conv ∈ R^{B×T×hidden_dim}
      2. Multi-Head Attention (§3.3):
           MultiHead(H_conv, H_conv, H_conv)     →  H_attn ∈ R^{B×T×hidden_dim}
      3. Residual Connection via Concat (§3.4):
           H_combined = Concat(H_conv, H_attn)   →  R^{B×T×2·hidden_dim}
      4. Feature Attribution Mapping (§3.5):
           φ' = Conv1D(H_combined) · W_F         →  R^{B×T×D}
      5. Softshrink sparsity activation (§3.5):
           φ  = softshrink(φ', τ)
           Uses F.softshrink to avoid -0.0 artefacts from manual sign()*relu().
    """

    def __init__(self, time_steps, n_features, hidden_dim=128, kernel_size=3,
                 dropout_rate=0.2, sparsity_threshold=0.01, n_attention_heads=4):
        super().__init__()

        self.time_steps = time_steps
        self.n_features = n_features
        self.sparsity_threshold = sparsity_threshold

        # ── §3.2  Temporal Convolution Block ──────────────────────────────────
        padding = (kernel_size - 1) // 2
        self.conv       = nn.Conv1d(n_features, hidden_dim, kernel_size, padding=padding)
        self.gelu       = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout_rate)

        # ── §3.3  Multi-Head Attention ────────────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # ── §3.5  Feature Attribution Mapping ────────────────────────────────
        self.W_F = nn.Conv1d(2 * hidden_dim, n_features, kernel_size=1)

        nn.init.xavier_uniform_(self.W_F.weight, gain=0.1)
        if self.W_F.bias is not None:
            nn.init.zeros_(self.W_F.bias)

    def forward(self, x, baseline=None):
        """
        Args:
            x:        Input tensor — (B, T, D)
            baseline: Unused; kept for API compatibility with the trainer

        Returns:
            φ: Feature attribution tensor — (B, T, D)
        """
        # ── §3.2  Temporal Convolution Block ──────────────────────────────────
        h = x.permute(0, 2, 1)
        h = self.conv(h)
        h = self.gelu(h)
        h = h.permute(0, 2, 1)
        h = self.layer_norm(h)
        h = self.dropout(h)
        H_conv = h

        # ── §3.3  Multi-Head Attention ────────────────────────────────────────
        H_attn, _ = self.attention(H_conv, H_conv, H_conv)

        # ── §3.4  Residual Connection ─────────────────────────────────────────
        H_combined   = torch.cat([H_conv, H_attn], dim=-1)
        H_combined_t = H_combined.permute(0, 2, 1)
        phi_prime    = self.W_F(H_combined_t).permute(0, 2, 1)

        # ── Softshrink sparsity activation (Eq. 9) ───────────────────────────
        # FIX: use F.softshrink instead of sign()*relu(|·|-τ).
        # The manual version produces -0.0 for negative sub-threshold values
        # (sign returns -1, relu returns 0, product is -0.0), which looks like
        # a non-zero negative attribution but is actually zero.
        # F.softshrink is numerically identical but returns clean 0.0 in all cases.
        phi = F.softshrink(phi_prime, lambd=self.sparsity_threshold)
        return phi


# ============================
# TDE TRAINER CLASS
# ============================
class TemporalDeepExplainer:
    """
    Trainer for Temporal Deep Explainer (TDE) — paper-aligned implementation.

    Training procedure (§3):
    ─ Masking: feature-level coalition masks drawn from the Shapley kernel
      distribution p(k) (Eq. 1 of the paper). Paired sampling (M, 1−M) is
      used to reduce estimation variance.
    ─ Loss (Eq. 10):
        L_total = coalition + λ₁‖φ‖₁ + λ_e‖φ‖₂² + λ_s Σ‖φ_t − φ_{t-1}‖₂²
      where coalition = (1/B) Σ (Σ_{t,d} M_{t,d}·φ_{t,d}^(i) − (f(x̃_i) − f(b)))².
      φ is computed on the ORIGINAL input x, not the masked input x̃.
    """

    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True,
                 min_lr=1e-6, l1_lambda=0.01, l2_lambda=0.01, smoothness_lambda=0.1,
                 weight_decay=1e-4, hidden_dim=128, kernel_size=3, dropout_rate=0.2,
                 sparsity_threshold=0.01, n_attention_heads=4, optimizer_type='adam',
                 learning_rate=1e-3, paired_sampling=True, samples_per_feature=2, **kwargs):

        self.device = device

        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.patience    = patience
        self.verbose     = verbose
        self.min_lr      = min_lr

        self.l1_lambda         = l1_lambda
        self.l2_lambda         = l2_lambda
        self.smoothness_lambda = smoothness_lambda

        self.weight_decay   = weight_decay
        self.optimizer_type = optimizer_type
        self.learning_rate  = learning_rate

        self.hidden_dim         = hidden_dim
        self.kernel_size        = kernel_size
        self.dropout_rate       = dropout_rate
        self.sparsity_threshold = sparsity_threshold
        self.n_attention_heads  = n_attention_heads

        self.paired_sampling     = paired_sampling
        self.samples_per_feature = samples_per_feature

        self.explainer          = None
        self.baseline           = None
        self.base_pred          = None
        self.feature_names      = None
        self.time_steps         = None
        self.n_features         = None
        self.model_predict_func = None

        self.best_loss = float('inf')
        self.history   = {'train_loss': [], 'val_loss': [], 'lr': []}

        self._gpu_model              = None
        self._model_on_gpu           = False
        self._baseline_cache         = None
        self._shapley_probs_features = None

        self.scaler = GradScaler() if torch.cuda.is_available() else None

        self._init_params = {k: v for k, v in locals().items()
                             if k not in ('self', 'kwargs')}

    # ──────────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────────
    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        self.time_steps         = X_train.shape[1]
        self.n_features         = X_train.shape[2]
        self.feature_names      = feature_names
        self.model_predict_func = model_predict_func

        if len(feature_names) != self.n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"n_features ({self.n_features})"
            )

        if gpu_model is not None:
            self._gpu_model    = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        else:
            self._gpu_model    = None
            self._model_on_gpu = False

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.nan_to_num(
            torch.median(X_tensor, dim=0)[0],
            nan=0.0, posinf=1.0, neginf=-1.0,
        )

        if self._model_on_gpu:
            with torch.no_grad():
                base_raw  = self._gpu_model(self.baseline.unsqueeze(0))
                bp_tensor = (base_raw[:, 0] if base_raw.ndim > 1 else base_raw).flatten()[0]
                bp = float(bp_tensor.cpu().item()) if isinstance(bp_tensor, torch.Tensor) else float(bp_tensor)
        else:
            base_np  = self.baseline.unsqueeze(0).cpu().numpy()
            base_raw = model_predict_func(base_np)
            if isinstance(base_raw, torch.Tensor):
                base_raw = base_raw.cpu().numpy()
            bp = float(np.atleast_1d(base_raw).flatten()[0])

        if not np.isfinite(bp):
            if self.verbose:
                print(f"    [WARN] base_pred={bp} for baseline input — resetting to 0.0")
            bp = 0.0
        self.base_pred = torch.tensor(bp, dtype=torch.float32, device=self.device)

        self.explainer = TemporalExplainerNetwork(
            self.time_steps, self.n_features,
            self.hidden_dim, self.kernel_size,
            self.dropout_rate, self.sparsity_threshold,
            self.n_attention_heads,
        ).to(self.device)

        _, self._shapley_probs_features = self._compute_shapley_kernel(self.n_features)
        self._baseline_cache = None

    # ──────────────────────────────────────────────────────────────────────────
    # Shapley kernel
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_shapley_kernel(self, d):
        """p(k) ∝ (D−1) / [k · (D−k) · C(D,k)]   k ∈ {1, …, D−1}"""
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)

        k = torch.arange(1, d, device=self.device, dtype=torch.float64)
        log_binom = (
            torch.lgamma(torch.tensor(d + 1.0, device=self.device, dtype=torch.float64))
            - torch.lgamma(k + 1)
            - torch.lgamma(d - k + 1)
        )
        binom   = torch.exp(log_binom)
        weights = ((d - 1) / (k * (d - k) * binom + 1e-10)).float()
        probs   = weights / weights.sum()
        return weights, probs

    # ──────────────────────────────────────────────────────────────────────────
    # Masking
    # ──────────────────────────────────────────────────────────────────────────
    def _generate_feature_masks(self, batch_size):
        probs     = self._shapley_probs_features
        total     = batch_size * self.samples_per_feature
        k_idx     = torch.multinomial(probs, total, replacement=True)
        k_samples = torch.arange(1, self.n_features, device=self.device)[k_idx]
        rand      = torch.rand(total, self.n_features, device=self.device)
        masks_feat = (torch.argsort(rand, dim=1) < k_samples.unsqueeze(1)).float()
        masks = masks_feat.unsqueeze(1).expand(-1, self.time_steps, -1).contiguous()
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks

    # ──────────────────────────────────────────────────────────────────────────
    # Black-box predictions
    # ──────────────────────────────────────────────────────────────────────────
    def _get_predictions(self, inputs):
        raw_fallback = self.base_pred.item() if isinstance(self.base_pred, torch.Tensor) else (float(self.base_pred) if self.base_pred is not None else 0.0)
        fallback = raw_fallback if np.isfinite(raw_fallback) else 0.0
        with torch.no_grad():
            if self._model_on_gpu and self._gpu_model is not None:
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                elif inputs.device != self.device:
                    inputs = inputs.to(self.device)
                inputs = torch.nan_to_num(inputs.float(), nan=0.0, posinf=1.0, neginf=-1.0)
                pred   = self._gpu_model(inputs)
                result = (pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()).float()
                return torch.nan_to_num(result, nan=fallback, posinf=fallback, neginf=fallback)
            else:
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.cpu().numpy()
                inputs = np.nan_to_num(inputs.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
                preds  = self.model_predict_func(inputs)
                result = np.nan_to_num(np.atleast_1d(preds).flatten().astype(np.float32), nan=fallback, posinf=fallback, neginf=fallback)
                return torch.tensor(result, dtype=torch.float32, device=self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────
    def _process_batch(self, X_batch, optimizer):
        batch_size = X_batch.size(0)
        X_batch    = X_batch.to(self.device, non_blocking=True)

        X_expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks      = self._generate_feature_masks(batch_size)
        total      = masks.size(0)

        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired      = X_expanded.repeat(repeat_factor, 1, 1)[:total]

        if self._baseline_cache is None or self._baseline_cache.size(0) < total:
            max_cache = max(total, self.batch_size * self.samples_per_feature * 4)
            self._baseline_cache = (
                self.baseline.unsqueeze(0).expand(max_cache, -1, -1).contiguous().clone()
            )
        baseline_exp = self._baseline_cache[:total]

        X_masked = torch.addcmul(baseline_exp, X_paired - baseline_exp, masks)
        f_masked  = self._get_predictions(X_masked)
        if not isinstance(f_masked, torch.Tensor):
            f_masked = torch.tensor(f_masked, dtype=torch.float32, device=self.device)

        use_amp = self.scaler is not None and self.device.type == 'cuda'
        with autocast(enabled=use_amp):
            # FIX: explain the ORIGINAL input X_paired, not the masked input.
            # The coalition loss requires mask-weighted phi(x) to match
            # f(x̃) − f(baseline) across many masks simultaneously — this
            # forces the network to learn which features genuinely matter.
            # Passing X_masked instead collapses phi to a uniform scalar.
            phi = self.explainer(X_paired, self.baseline)

            # FIX: mask-weighted sum, not full sum.
            phi_sum   = (masks * phi).sum(dim=(1, 2))
            pred_diff = f_masked - self.base_pred
            fidelity  = ((phi_sum - pred_diff) ** 2).mean()

            l1_loss   = self.l1_lambda * torch.abs(phi).mean()
            l2_loss   = self.l2_lambda * (phi ** 2).mean()
            smooth_loss = (
                self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
                if phi.size(1) > 1 else torch.tensor(0.0, device=self.device)
            )
            loss = fidelity + l1_loss + l2_loss + smooth_loss

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

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────
    def _validate(self, X_val):
        self.explainer.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=self.batch_size, shuffle=False,
        )
        total_loss, n_batches = 0.0, 0

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi     = self.explainer(X_batch, self.baseline)
                preds   = self._get_predictions(X_batch)
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
                eff_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                if torch.isfinite(eff_err):
                    total_loss += eff_err.item()
                    n_batches  += 1

        self.explainer.train()
        return total_loss / max(n_batches, 1) if n_batches > 0 else float('inf')

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — train / explain / save / load
    # ──────────────────────────────────────────────────────────────────────────
    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model=gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')

        use_cuda    = self.device.type == 'cuda'
        num_workers = 4 if use_cuda else 0

        effective_bs = min(self.batch_size, len(X_train) - 1)
        if effective_bs < 1:
            if self.verbose:
                print("    [ERROR] Not enough training samples")
            return float('inf')

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=effective_bs, shuffle=True,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )

        opt_cls   = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(
            self.optimizer_type, torch.optim.Adam
        )
        optimizer = opt_cls(
            self.explainer.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5, min_lr=self.min_lr,
        )

        best_val, best_weights, no_improve = float('inf'), None, 0

        for epoch in range(self.n_epochs):
            self.explainer.train()
            epoch_loss, n_batches = 0.0, 0

            for (X_batch,) in loader:
                bl = self._process_batch(X_batch, optimizer)
                if bl != float('inf'):
                    epoch_loss += bl
                    n_batches  += 1

            if n_batches == 0:
                if self.verbose:
                    print(f"    [ERROR] All batches failed at epoch {epoch + 1}")
                return float('inf')

            epoch_loss /= n_batches
            val_loss    = self._validate(X_val)

            if val_loss == float('inf'):
                if self.verbose:
                    print(f"    [ERROR] Validation failed at epoch {epoch + 1}")
                continue

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)

            if val_loss < best_val - 1e-6:
                best_val     = val_loss
                src          = self.explainer._orig_mod if hasattr(self.explainer, '_orig_mod') else self.explainer
                best_weights = {k: v.clone() for k, v in src.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch + 1:3d} | L:{epoch_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")

            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [STOP] epoch {epoch + 1}")
                break

        if best_weights:
            tgt = self.explainer._orig_mod if hasattr(self.explainer, '_orig_mod') else self.explainer
            tgt.load_state_dict(best_weights)

        self.best_loss       = best_val
        self._baseline_cache = None
        return best_val

    def explain(self, instance):
        if self.explainer is None:
            raise ValueError("Explainer not trained. Call train() first.")
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
        os.makedirs(path, exist_ok=True)
        src        = self.explainer._orig_mod if hasattr(self.explainer, '_orig_mod') else self.explainer
        state_dict = src.state_dict()
        bp         = self.base_pred
        if isinstance(bp, torch.Tensor):
            bp = bp.cpu()
        else:
            bp = torch.tensor(float(bp))
        state = {
            'explainer':     state_dict,
            'baseline':      self.baseline.cpu(),
            'base_pred':     bp,
            'time_steps':    self.time_steps,
            'n_features':    self.n_features,
            'feature_names': self.feature_names,
            'best_loss':     self.best_loss,
            'history':       self.history,
            'init_params':   self._init_params,
        }
        save_path = os.path.join(path, f"{filename}.pt")
        torch.save(state, save_path)
        return save_path

    @classmethod
    def load(cls, path, filename="tde_explainer", device_override=None):
        dev       = device_override or device
        load_path = os.path.join(path, f"{filename}.pt")
        state     = torch.load(load_path, map_location=dev, weights_only=False)
        params    = state.get('init_params', {})
        exp       = cls(**params)
        exp.device        = dev
        exp.time_steps    = state['time_steps']
        exp.n_features    = state['n_features']
        exp.feature_names = state['feature_names']
        exp.baseline      = state['baseline'].to(dev)
        bp = state['base_pred']
        exp.base_pred        = bp.to(dev) if isinstance(bp, torch.Tensor) else torch.tensor(float(bp), device=dev)
        exp.best_loss        = state.get('best_loss', float('inf'))
        exp.history          = state.get('history', {})
        exp._gpu_model       = None
        exp._model_on_gpu    = False
        exp._baseline_cache  = None
        exp.explainer = TemporalExplainerNetwork(
            exp.time_steps, exp.n_features,
            params.get('hidden_dim', 128), params.get('kernel_size', 3),
            params.get('dropout_rate', 0.2), params.get('sparsity_threshold', 0.01),
            params.get('n_attention_heads', 4),
        ).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        _, exp._shapley_probs_features = exp._compute_shapley_kernel(exp.n_features)
        return exp


# ============================
# FASTSHAP NETWORK ARCHITECTURE
# ============================
class FastSHAPNetwork(nn.Module):
    """
    FastSHAP Neural Network - Pure MLP Architecture.
    """

    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# ============================
# FASTSHAP TRAINER CLASS
# ============================
class FastSHAPExplainer:
    """FastSHAP Explainer — GPU-optimised baseline implementation."""

    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True,
                 min_lr=1e-6, l1_lambda=0.01, efficiency_lambda=0.1,
                 weight_decay=1e-4, hidden_dim=256, n_layers=2, dropout_rate=0.2,
                 optimizer_type='adam', learning_rate=1e-3, paired_sampling=True,
                 samples_per_feature=2, **kwargs):

        self.device = device

        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.verbose    = verbose
        self.min_lr     = min_lr

        self.l1_lambda         = l1_lambda
        self.efficiency_lambda = efficiency_lambda
        self.weight_decay      = weight_decay

        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.dropout_rate = dropout_rate

        self.optimizer_type = optimizer_type
        self.learning_rate  = learning_rate

        self.paired_sampling     = paired_sampling
        self.samples_per_feature = samples_per_feature

        self.explainer          = None
        self.baseline           = None
        self.base_pred          = None
        self.feature_names      = None
        self.input_dim          = None
        self.time_steps         = None
        self.n_features         = None
        self.model_predict_func = None

        self.best_loss = float('inf')
        self.history   = {'train_loss': [], 'val_loss': [], 'lr': []}

        self._gpu_model              = None
        self._model_on_gpu           = False
        self._shapley_probs_elements = None

        self.scaler       = GradScaler() if torch.cuda.is_available() else None
        self._init_params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}

    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        self.time_steps         = X_train.shape[1]
        self.n_features         = X_train.shape[2]
        self.input_dim          = self.time_steps * self.n_features
        self.feature_names      = feature_names
        self.model_predict_func = model_predict_func

        if gpu_model is not None:
            self._gpu_model    = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        else:
            self._gpu_model    = None
            self._model_on_gpu = False

        X_flat   = X_train.reshape(len(X_train), -1)
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        self.baseline = torch.nan_to_num(
            torch.median(X_tensor, dim=0)[0],
            nan=0.0, posinf=1.0, neginf=-1.0,
        )

        if self._model_on_gpu:
            with torch.no_grad():
                b3d       = self.baseline.view(1, self.time_steps, self.n_features)
                base_raw  = self._gpu_model(b3d)
                bp_tensor = (base_raw[:, 0] if base_raw.ndim > 1 else base_raw).flatten()[0]
                bp = float(bp_tensor.cpu().item()) if isinstance(bp_tensor, torch.Tensor) else float(bp_tensor)
        else:
            b_np     = self.baseline.unsqueeze(0).cpu().numpy().reshape(1, self.time_steps, self.n_features)
            base_raw = model_predict_func(b_np)
            if isinstance(base_raw, torch.Tensor):
                base_raw = base_raw.cpu().numpy()
            bp = float(np.atleast_1d(base_raw).flatten()[0])

        if not np.isfinite(bp):
            if self.verbose:
                print(f"    [WARN] base_pred={bp} for baseline input — resetting to 0.0")
            bp = 0.0
        self.base_pred = torch.tensor(bp, dtype=torch.float32, device=self.device)

        self.explainer = FastSHAPNetwork(self.input_dim, self.hidden_dim, self.n_layers, self.dropout_rate).to(self.device)
        _, self._shapley_probs_elements = self._compute_shapley_kernel(self.input_dim)

    def _compute_shapley_kernel(self, d):
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        k         = torch.arange(1, d, device=self.device, dtype=torch.float64)
        log_binom = (
            torch.lgamma(torch.tensor(d + 1.0, device=self.device, dtype=torch.float64))
            - torch.lgamma(k + 1) - torch.lgamma(d - k + 1)
        )
        weights = ((d - 1) / (k * (d - k) * torch.exp(log_binom) + 1e-10)).float()
        probs   = weights / weights.sum()
        return weights, probs

    def _generate_element_masks(self, batch_size):
        probs     = self._shapley_probs_elements
        total     = batch_size * self.samples_per_feature
        d         = self.input_dim
        k_idx     = torch.multinomial(probs, total, replacement=True)
        k_samples = torch.arange(1, d, device=self.device)[k_idx]
        rand      = torch.rand(total, d, device=self.device)
        masks     = (torch.argsort(rand, dim=1) < k_samples.unsqueeze(1)).float()
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        return masks

    def _get_predictions(self, inputs_flat):
        raw_fallback = self.base_pred.item() if isinstance(self.base_pred, torch.Tensor) else (float(self.base_pred) if self.base_pred is not None else 0.0)
        fallback = raw_fallback if np.isfinite(raw_fallback) else 0.0
        with torch.no_grad():
            if self._model_on_gpu and self._gpu_model is not None:
                if not isinstance(inputs_flat, torch.Tensor):
                    inputs_flat = torch.tensor(inputs_flat, dtype=torch.float32, device=self.device)
                elif inputs_flat.device != self.device:
                    inputs_flat = inputs_flat.to(self.device)
                inputs_flat = torch.nan_to_num(inputs_flat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
                inp3d  = inputs_flat.view(-1, self.time_steps, self.n_features)
                pred   = self._gpu_model(inp3d)
                result = (pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()).float()
                return torch.nan_to_num(result, nan=fallback, posinf=fallback, neginf=fallback)
            else:
                if isinstance(inputs_flat, torch.Tensor):
                    inputs_flat = inputs_flat.cpu().numpy()
                inputs_flat = np.nan_to_num(inputs_flat.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
                inp3d  = inputs_flat.reshape(-1, self.time_steps, self.n_features)
                preds  = self.model_predict_func(inp3d)
                result = np.nan_to_num(np.atleast_1d(preds).flatten().astype(np.float32), nan=fallback, posinf=fallback, neginf=fallback)
                return torch.tensor(result, dtype=torch.float32, device=self.device)

    def _process_batch(self, X_batch_flat, optimizer):
        batch_size    = X_batch_flat.size(0)
        X_batch_flat  = X_batch_flat.to(self.device, non_blocking=True)
        expanded      = X_batch_flat.repeat(self.samples_per_feature, 1)
        masks         = self._generate_element_masks(batch_size)
        total         = masks.size(0)
        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired      = expanded.repeat(repeat_factor, 1)[:total]
        baseline_exp  = self.baseline.unsqueeze(0).expand(total, -1)

        masked       = X_paired * masks + baseline_exp * (1.0 - masks)
        preds_masked = self._get_predictions(masked)

        if self.paired_sampling:
            n_unique   = total // 2
            preds_orig = self._get_predictions(X_paired[:n_unique]).repeat(2)
        else:
            preds_orig = self._get_predictions(X_paired)

        if not isinstance(preds_masked, torch.Tensor):
            preds_masked = torch.tensor(preds_masked, dtype=torch.float32, device=self.device)
        if not isinstance(preds_orig, torch.Tensor):
            preds_orig   = torch.tensor(preds_orig, dtype=torch.float32, device=self.device)

        use_amp = self.scaler is not None and self.device.type == 'cuda'
        with autocast(enabled=use_amp):
            phi            = self.explainer(X_paired)
            masked_sum     = (masks * phi).sum(dim=1)
            coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
            eff_loss       = self.efficiency_lambda * ((phi.sum(dim=1) - (preds_orig - self.base_pred)) ** 2).mean()
            l1_loss        = self.l1_lambda * torch.abs(phi).mean()
            loss           = coalition_loss + eff_loss + l1_loss

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
        self.explainer.eval()
        loader      = DataLoader(TensorDataset(torch.FloatTensor(X_val_flat)), batch_size=self.batch_size, shuffle=False)
        total_loss, n_batches = 0.0, 0

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi     = self.explainer(X_batch)
                preds   = self._get_predictions(X_batch)
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
                eff_err = ((phi.sum(dim=1) - (preds - self.base_pred)) ** 2).mean()
                if torch.isfinite(eff_err):
                    total_loss += eff_err.item()
                    n_batches  += 1

        self.explainer.train()
        return total_loss / max(n_batches, 1) if n_batches > 0 else float('inf')

    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model=gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')

        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat   = X_val.reshape(len(X_val), -1)

        use_cuda     = self.device.type == 'cuda'
        num_workers  = 4 if use_cuda else 0
        effective_bs = min(self.batch_size, len(X_train_flat) - 1)
        if effective_bs < 1:
            return float('inf')

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_flat)),
            batch_size=effective_bs, shuffle=True,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )

        opt_cls   = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(self.optimizer_type, torch.optim.Adam)
        optimizer = opt_cls(self.explainer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, min_lr=self.min_lr)

        best_val, best_weights, no_improve = float('inf'), None, 0

        for epoch in range(self.n_epochs):
            self.explainer.train()
            epoch_loss, n_batches = 0.0, 0

            for (X_batch,) in loader:
                bl = self._process_batch(X_batch, optimizer)
                if bl != float('inf'):
                    epoch_loss += bl
                    n_batches  += 1

            if n_batches == 0:
                return float('inf')

            epoch_loss /= n_batches
            val_loss    = self._validate(X_val_flat)
            if val_loss == float('inf'):
                continue

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)

            if val_loss < best_val - 1e-6:
                best_val     = val_loss
                src          = self.explainer._orig_mod if hasattr(self.explainer, '_orig_mod') else self.explainer
                best_weights = {k: v.clone() for k, v in src.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch + 1:3d} | L:{epoch_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")

            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [STOP] epoch {epoch + 1}")
                break

        if best_weights:
            tgt = self.explainer._orig_mod if hasattr(self.explainer, '_orig_mod') else self.explainer
            tgt.load_state_dict(best_weights)

        self.best_loss = best_val
        return best_val

    def explain(self, instance):
        if self.explainer is None:
            raise ValueError("Explainer not trained. Call train() first.")
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        if instance.ndim == 3:
            instance = instance.reshape(instance.size(0), -1)
        elif instance.ndim == 2:
            if instance.size(0) == self.time_steps and instance.size(1) == self.n_features:
                instance = instance.reshape(1, -1)
            elif instance.size(1) != self.input_dim:
                instance = instance.reshape(1, -1)
        elif instance.ndim == 1:
            instance = instance.unsqueeze(0)
        instance = instance.to(self.device)
        self.explainer.eval()
        with torch.no_grad():
            phi_flat = self.explainer(instance).cpu().numpy()[0]
        return phi_flat.reshape(self.time_steps, self.n_features)

    def save(self, path, filename="fastshap_explainer"):
        os.makedirs(path, exist_ok=True)
        src = self.explainer._orig_mod if hasattr(self.explainer, '_orig_mod') else self.explainer
        bp  = self.base_pred
        if isinstance(bp, torch.Tensor):
            bp = bp.cpu()
        else:
            bp = torch.tensor(float(bp))
        state = {
            'explainer':     src.state_dict(),
            'baseline':      self.baseline.cpu(),
            'base_pred':     bp,
            'input_dim':     self.input_dim,
            'time_steps':    self.time_steps,
            'n_features':    self.n_features,
            'feature_names': self.feature_names,
            'best_loss':     self.best_loss,
            'history':       self.history,
            'init_params':   self._init_params,
        }
        save_path = os.path.join(path, f"{filename}.pt")
        torch.save(state, save_path)
        return save_path

    @classmethod
    def load(cls, path, filename="fastshap_explainer", device_override=None):
        dev       = device_override or device
        load_path = os.path.join(path, f"{filename}.pt")
        state     = torch.load(load_path, map_location=dev, weights_only=False)
        params    = state.get('init_params', {})
        exp       = cls(**params)
        exp.device        = dev
        exp.input_dim     = state['input_dim']
        exp.time_steps    = state['time_steps']
        exp.n_features    = state['n_features']
        exp.feature_names = state['feature_names']
        exp.baseline      = state['baseline'].to(dev)
        bp = state['base_pred']
        exp.base_pred     = bp.to(dev) if isinstance(bp, torch.Tensor) else torch.tensor(float(bp), device=dev)
        exp.best_loss     = state.get('best_loss', float('inf'))
        exp.history       = state.get('history', {})
        exp._gpu_model    = None
        exp._model_on_gpu = False
        exp.explainer = FastSHAPNetwork(
            exp.input_dim,
            params.get('hidden_dim', 256),
            params.get('n_layers', 2),
            params.get('dropout_rate', 0.2),
        ).to(dev)
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        _, exp._shapley_probs_elements = exp._compute_shapley_kernel(exp.input_dim)
        return exp