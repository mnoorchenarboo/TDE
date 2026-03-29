import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


class TimeSeriesSHAPExplainer:
    def __init__(self,
                 n_epochs=200,
                 batch_size=512,
                 patience=10,
                 delta=1e-4,
                 verbose=True,
                 min_lr=1e-6,
                 # Regularization
                 l1_lambda=0.1,
                 weight_decay=1e-4,
                 activation_shrink=0.1,
                 smoothness_lambda=0.1,
                 efficiency_lambda=0.05,
                 # ViT Parameters
                 vit_dim=128,
                 vit_depth=4,
                 vit_heads=4,
                 vit_mlp_dim=256,
                 # Sampling
                 paired_sampling=False,
                 samples_per_feature=1):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.min_lr = min_lr
        self.l1_lambda = l1_lambda
        self.weight_decay = weight_decay
        self.activation_shrink = activation_shrink
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda
        self.vit_dim = vit_dim
        self.vit_depth = vit_depth
        self.vit_heads = vit_heads
        self.vit_mlp_dim = vit_mlp_dim
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature

        # Track initialization parameters
        self._init_params = locals()
        del self._init_params['self']

        # Core components
        self.explainer = None
        self.baseline = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.model_predict_func = None
        self.base_pred = None
        self.best_loss = float('inf')

    class ViTExplainer(nn.Module):
        """Vision Transformer for SHAP value estimation"""

        def __init__(self, time_steps, n_features, activation_shrink,
                     dim, depth, heads, mlp_dim):
            super().__init__()
            self.time_steps = time_steps
            self.n_features = n_features

            # Input embedding
            self.patch_embed = nn.Linear(time_steps, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.randn(1, n_features + 1, dim))

            # Transformer encoder
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    batch_first=True,
                    dropout=0.1
                ),
                num_layers=depth
            )

            # Output decoder
            self.decoder = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, time_steps),
                nn.Softshrink(activation_shrink)
            )

        def forward(self, x):
            # x shape: (batch, time, features)
            batch_size = x.size(0)

            # Embed patches (features)
            x = x.permute(0, 2, 1)  # (batch, features, time)
            patch_embeds = self.patch_embed(x)  # (batch, features, dim)

            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, patch_embeds), dim=1)

            # Add positional embeddings
            embeddings += self.pos_embed[:, :(self.n_features + 1)]

            # Transformer processing
            encoded = self.transformer(embeddings)

            # Decode features (excluding CLS token)
            features = encoded[:, 1:]  # (batch, features, dim)
            decoded = self.decoder(features)  # (batch, features, time)

            return decoded.permute(0, 2, 1)  # (batch, time, features)

    # Core Methods =============================================================

    def initialize(self, X_train, model_predict_func, feature_names):
        """Initialize explainer with training data and model"""
        self._validate_input(X_train)
        self._setup_core_components(X_train, model_predict_func, feature_names)
        self._train_fastshap(X_train)
        return self

    def explain(self, instance):
        """Generate SHAP values for an input instance"""
        instance = self._preprocess_input(instance)
        with torch.no_grad():
            shap_values = self.explainer(instance).cpu().numpy()
        return self._create_shap_dataframe(shap_values[0])

    # Training Methods =========================================================

    def _train_fastshap(self, X_train):
        """Core training procedure with ViT integration"""
        d = self.time_steps * self.n_features
        weights, probs = self._compute_shapley_kernel(d)

        loader = self._create_dataloader(X_train)
        optimizer = torch.optim.AdamW(
            self.explainer.parameters(),
            lr=1e-4,
            weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.patience // 2,
            min_lr=self.min_lr,
            verbose=self.verbose
        )

        best_weights = None
        no_improve = 0
        best_loss = float('inf')
        smoothed_loss = None  # For exponential moving average

        for epoch in range(self.n_epochs):
            self.explainer.train()
            epoch_loss = 0.0

            for X_batch in loader:
                X_batch = X_batch[0].to(self.device)  # Move to device here
                loss = self._process_batch(X_batch, d, weights, probs, optimizer)
                epoch_loss += loss.item()

            # Update learning rate
            current_loss = epoch_loss / len(loader)
            smoothed_loss = self._update_smoothed_loss(smoothed_loss, current_loss)
            scheduler.step(smoothed_loss)

            # Early stopping check
            if smoothed_loss < (best_loss - self.delta):
                best_loss = smoothed_loss
                best_weights = self.explainer.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            # Print progress
            if self.verbose:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1:03d} | Loss: {current_loss:.4f} "
                      f"(Smoothed: {smoothed_loss:.4f}) | LR: {lr:.2e}")

            # Early stopping
            if no_improve >= self.patience and epoch >= self.n_epochs // 2:
                if best_weights:
                    self.explainer.load_state_dict(best_weights)
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        return self

    def _process_batch(self, X_batch, d, weights, probs, optimizer):
        """Process a training batch"""
        batch_size = X_batch.size(0)

        # Generate masks
        masks = self._generate_shapley_masks(batch_size, d, probs)
        total_samples = masks.size(0)

        # Prepare inputs
        X_expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        X_paired = X_expanded.repeat(total_samples // (batch_size * self.samples_per_feature), 1, 1)
        baseline_paired = self.baseline.repeat(total_samples, 1, 1)

        masked_inputs = X_paired * masks + baseline_paired * (1 - masks)

        # Get predictions
        with torch.no_grad():
            preds = self._get_model_predictions(masked_inputs)

        # Compute SHAP values
        phi = self.explainer(X_paired)

        # Calculate loss components
        mse_loss = ((preds - self.base_pred - (masks * phi).sum((1, 2))) ** 2).mean()
        l1_reg = self.l1_lambda * torch.abs(phi).mean()
        smooth_loss = self.smoothness_lambda * (phi.diff(dim=1) ** 2).mean()
        eff_loss = self.efficiency_lambda * (phi ** 2).mean()

        total_loss = mse_loss + l1_reg + smooth_loss + eff_loss

        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), 1.0)
        optimizer.step()

        return total_loss

    # Helper Methods ===========================================================

    def _validate_input(self, X_train):
        """Validate input dimensions"""
        if X_train.ndim != 3:
            raise ValueError("Input must be 3D: (samples, time_steps, features)")
        if X_train.shape[0] < 10:
            raise ValueError("Need at least 10 samples for stable training")

    def _setup_core_components(self, X_train, model_predict_func, feature_names):
        """Initialize core components"""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func

        # Initialize baseline
        self.baseline = torch.median(torch.FloatTensor(X_train), dim=0)[0].to(self.device)
        baseline_np = self.baseline.unsqueeze(0).cpu().numpy()
        self.base_pred = torch.FloatTensor(model_predict_func(baseline_np)).to(self.device)

        # Initialize ViT explainer
        self.explainer = self.ViTExplainer(
            time_steps=self.time_steps,
            n_features=self.n_features,
            activation_shrink=self.activation_shrink,
            dim=self.vit_dim,
            depth=self.vit_depth,
            heads=self.vit_heads,
            mlp_dim=self.vit_mlp_dim
        ).to(self.device)

    def _generate_shapley_masks(self, batch_size, d, probs):
        """Generate coalition masks using Shapley kernel"""
        # Sample coalition sizes
        k_indices = torch.multinomial(probs, batch_size * self.samples_per_feature, replacement=True)
        k_samples = torch.arange(1, d, device=self.device)[k_indices]

        # Generate random masks
        rand = torch.rand(batch_size * self.samples_per_feature, d, device=self.device)
        sorted_indices = torch.argsort(rand, dim=1)
        masks = (sorted_indices < k_samples.unsqueeze(1)).float()

        # Reshape to time series format
        masks = masks.view(-1, self.time_steps, self.n_features)

        # Add paired masks if enabled
        if self.paired_sampling:
            masks = torch.cat([masks, 1 - masks], dim=0)

        return masks

    def _compute_shapley_kernel(self, d):
        """Calculate Shapley kernel weights with backward compatibility"""
        k_values = torch.arange(1, d, device=self.device)

        # Manually calculate binomial coefficients
        def binomial_coeff(n, k):
            return torch.exp(torch.lgamma(torch.tensor(n + 1)) -
                             torch.lgamma(torch.tensor(k + 1)) -
                             torch.lgamma(torch.tensor(n - k + 1)))

        # Calculate binomial coefficients for all k in k_values
        binom_coeffs = binomial_coeff(d, k_values).to(self.device)

        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs)
        probs = weights / weights.sum()
        return weights, probs

    def _update_smoothed_loss(self, current_smooth, new_loss, alpha=0.1):
        """Update exponential moving average of loss"""
        return new_loss if current_smooth is None else alpha * new_loss + (1 - alpha) * current_smooth

    def _create_dataloader(self, X_train):
        """Create DataLoader without premature device assignment"""
        # Keep data on CPU and let DataLoader handle device transfer
        dataset = TensorDataset(torch.FloatTensor(X_train))  # Remove .to(device)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()  # Only pin if GPU available
        )

    def _preprocess_input(self, instance):
        """Prepare input instance for explanation"""
        if instance.ndim == 2:
            instance = instance[np.newaxis, :, :]
        return torch.FloatTensor(instance).to(self.device)

    def _create_shap_dataframe(self, shap_values):
        """Create formatted output dataframe"""
        return pd.DataFrame(
            shap_values,
            columns=self.feature_names,
            index=[f"t-{i}" for i in range(self.time_steps)]
        )

    def _get_model_predictions(self, inputs):
        """Get predictions from black-box model"""
        with torch.no_grad():
            masked_np = inputs.cpu().numpy()
            preds = torch.FloatTensor(self.model_predict_func(masked_np))
            return preds.flatten().to(self.device)

    # Serialization Methods ====================================================

    def save(self, path, filename="explainer"):
        """Save complete explainer state with all parameters and components."""
        import os
        os.makedirs(path, exist_ok=True)

        try:
            # Capture all initialization parameters
            self._init_params = {
                k: v for k, v in self.__dict__.items()
                if k in self.__init__.__code__.co_varnames
            }

            # Create comprehensive state dictionary
            state = {
                "explainer": self.explainer.state_dict(),
                "baseline": self.baseline.cpu(),
                "base_pred": self.base_pred.cpu(),
                "time_steps": self.time_steps,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "best_loss": self.best_loss,
                "init_params": self._init_params,
                "vit_config": {
                    "time_steps": self.time_steps,
                    "n_features": self.n_features,
                    "activation_shrink": self.activation_shrink,
                    "dim": self.vit_dim,
                    "depth": self.vit_depth,
                    "heads": self.vit_heads,
                    "mlp_dim": self.vit_mlp_dim
                }
            }

            # Save with metadata
            torch.save(state, os.path.join(path, f"{filename}.pt"))

            if self.verbose:
                print(f"Model saved successfully at: {os.path.join(path, f'{filename}.pt')}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    @classmethod
    def load(cls, path, filename="explainer", device=None):
        """Load complete explainer state with robust error handling."""
        import os
        try:
            # Device configuration
            device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load state dictionary
            full_path = os.path.join(path, f"{filename}.pt")
            state = torch.load(full_path, map_location=device)

            # Restore initialization parameters
            init_params = state.get("init_params", {})
            explainer = cls(**init_params)

            # Restore core attributes
            explainer.time_steps = state["time_steps"]
            explainer.n_features = state["n_features"]
            explainer.feature_names = state["feature_names"]
            explainer.best_loss = state.get("best_loss", float("inf"))

            # Restore baseline components
            explainer.baseline = state["baseline"].to(device)
            explainer.base_pred = state["base_pred"].to(device)

            # Rebuild ViT explainer architecture
            vit_config = state.get("vit_config", {
                "time_steps": explainer.time_steps,
                "n_features": explainer.n_features,
                "activation_shrink": explainer.activation_shrink,
                "dim": explainer.vit_dim,
                "depth": explainer.vit_depth,
                "heads": explainer.vit_heads,
                "mlp_dim": explainer.vit_mlp_dim
            })

            explainer.explainer = cls.ViTExplainer(
                time_steps=vit_config["time_steps"],
                n_features=vit_config["n_features"],
                activation_shrink=vit_config["activation_shrink"],
                dim=vit_config["dim"],
                depth=vit_config["depth"],
                heads=vit_config["heads"],
                mlp_dim=vit_config["mlp_dim"]
            ).to(device)

            # Load trained weights
            explainer.explainer.load_state_dict(state["explainer"])

            if explainer.verbose:
                print(f"Successfully loaded explainer from {full_path}")
                print(f"Device: {device}")
                print(f"Time steps: {explainer.time_steps}")
                print(f"Features: {explainer.n_features} ({', '.join(explainer.feature_names[:3])}...)")
                print(f"Baseline shape: {explainer.baseline.shape}")

            return explainer

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None