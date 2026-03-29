from Functions import main
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# Load and preprocess data
dataset_types = ["Residential", "Manufacturing facility", "Office building", "Retail store", "Medical clinic"]
mydata = main.load_and_preprocess_data(
    dataset_type=dataset_types[0],
    option_number=0,
    scaled=True,
    scale_type='both',
    val_ratio=0.1,
    test_ratio=0.1,
    input_seq_length=48,
    output_seq_length=24
)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_and_preprocess_features(data, feature_names=None):
    """
    Automatically detect feature types from data and preprocess them

    Args:
        data: Input data array of shape (samples, seq_len, features)
        feature_names: Optional list of feature names (for better logging)

    Returns:
        Tuple of (feature_types, processed_data)
    """
    n_features = data.shape[2]
    feature_types = []
    processed_data = data.copy()

    for f in range(n_features):
        # Extract all values for this feature across all samples and timesteps
        feature_values = data[:, :, f].flatten()

        # Count unique values
        unique_values = np.unique(feature_values)
        n_unique = len(unique_values)

        feature_name = feature_names[f] if feature_names is not None and f < len(feature_names) else f"Feature {f + 1}"

        # Detect feature type
        if n_unique <= 2:
            # Binary feature (0/1 or True/False)
            feature_types.append('binary')
            print(f"Detected {feature_name} as binary feature")

            # Ensure binary features are exactly 0 or 1
            processed_data[:, :, f] = np.round(processed_data[:, :, f])

        elif n_unique <= 10 and np.all(np.mod(unique_values, 1) == 0):
            # Categorical/integer feature with few unique values
            feature_types.append('categorical')
            print(f"Detected {feature_name} as categorical feature")

            # For categorical features, ensure they're integers
            processed_data[:, :, f] = np.round(processed_data[:, :, f])

        else:
            # Continuous feature
            feature_types.append('continuous')
            print(f"Detected {feature_name} as continuous feature")

            # Scale continuous features to [0,1] range for better stability
            f_min = np.min(data[:, :, f])
            f_max = np.max(data[:, :, f])
            if f_max > f_min:
                processed_data[:, :, f] = (data[:, :, f] - f_min) / (f_max - f_min)

    return feature_types, processed_data


class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len, feature_dim, latent_dim, hidden_dim, feature_types=None):
        """
        Improved VAE for time series with mixed feature types

        Args:
            seq_len: Length of input sequence
            feature_dim: Number of features
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            feature_types: List of feature types ('continuous', 'binary', or 'categorical')
                          If None, all features are treated as continuous
        """
        super(TimeSeriesVAE, self).__init__()

        # Store dimensions and feature types
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Default feature types to all continuous if not provided
        if feature_types is None:
            self.feature_types = ['continuous'] * feature_dim
        else:
            self.feature_types = feature_types

        # Encoder - using LSTM to preserve temporal structure
        self.encoder_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2  # Increased dropout
        )

        # Add layer normalization for stability
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Mean and variance projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Latent to decoder initial state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # Decoder - using LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2  # Increased dropout
        )

        # Add layer normalization for decoder
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Feature-specific output layers
        self.feature_output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(feature_dim)
        ])

    def encode(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        outputs, (hidden, _) = self.encoder_lstm(x)
        # Use the last layer's hidden state
        hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Apply normalization for stability
        hidden = self.encoder_norm(hidden)

        # Project to latent space
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x=None, teacher_forcing_ratio=0.5):
        # z shape: [batch_size, latent_dim]
        batch_size = z.size(0)

        # Check if teacher forcing should be used (only during training)
        use_teacher_forcing = x is not None and random.random() < teacher_forcing_ratio

        # Convert z to initial hidden state
        h0 = self.latent_to_hidden(z)
        h0 = h0.repeat(2, 1, 1)  # For 2-layer LSTM
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        # Initialize decoder input with zeros for first timestep
        decoder_input = torch.zeros(batch_size, 1, self.feature_dim, device=z.device)

        # Output tensor to store all timesteps
        outputs = torch.zeros(batch_size, self.seq_len, self.feature_dim, device=z.device)

        # Generate sequence auto-regressively
        for t in range(self.seq_len):
            # Pass current input through LSTM
            lstm_out, hidden = self.decoder_lstm(decoder_input, hidden)

            # Apply normalization
            lstm_out = self.decoder_norm(lstm_out.squeeze(1)).unsqueeze(1)

            # Process each feature with its appropriate output layer
            feature_outputs = []
            for f in range(self.feature_dim):
                # Apply feature-specific layer
                feature_out = self.feature_output_layers[f](lstm_out.squeeze(1)).unsqueeze(1)

                # Apply appropriate activation based on feature type
                if self.feature_types[f] == 'binary':
                    feature_out = torch.sigmoid(feature_out)
                elif self.feature_types[f] == 'categorical':
                    # No specific activation, will be handled in loss function
                    pass
                else:
                    # For continuous features, use sigmoid to bound to [0,1]
                    feature_out = torch.sigmoid(feature_out)

                feature_outputs.append(feature_out)

            # Combine all feature outputs
            curr_output = torch.cat(feature_outputs, dim=2)
            outputs[:, t:t + 1, :] = curr_output

            # Set up next input
            if t < self.seq_len - 1:  # No need for input on last timestep
                if use_teacher_forcing and x is not None:
                    # Use ground truth as next input
                    decoder_input = x[:, t:t + 1, :]
                else:
                    # Use current output as next input
                    decoder_input = curr_output

        return outputs

    def forward(self, x, teacher_forcing_ratio=0.5):
        # Encode
        mu, logvar = self.encode(x)

        # Sample from latent space
        z = self.reparameterize(mu, logvar)

        # Decode (with teacher forcing during training)
        if self.training:
            recon_x = self.decode(z, x, teacher_forcing_ratio)
        else:
            recon_x = self.decode(z, None, 0.0)  # No teacher forcing during inference

        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar, feature_types, sparsity_weight=0.0, kl_weight=1.0, beta=1.0):
    """
    Beta-VAE loss function that handles different feature types with better weighting

    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        feature_types: List of feature types ('continuous', 'binary', 'categorical')
        sparsity_weight: Weight for latent space sparsity regularization
        kl_weight: Weight for KL divergence term
        beta: Coefficient for KL term in beta-VAE formulation
    """
    batch_size = x.size(0)

    # Initialize feature weights (can be adjusted based on domain knowledge)
    feature_weights = []
    for f_type in feature_types:
        if f_type == 'continuous':
            feature_weights.append(0.8)  # Slightly lower weight for continuous
        else:
            feature_weights.append(1.0)

    # Separate loss for each feature type
    recon_loss = 0
    feature_losses = []  # For logging individual feature losses

    # Process each feature with appropriate loss
    for f, (f_type, weight) in enumerate(zip(feature_types, feature_weights)):
        if f_type == 'binary':
            # Binary cross-entropy loss for binary features
            f_loss = F.binary_cross_entropy(
                recon_x[:, :, f],
                x[:, :, f],
                reduction='sum'
            )
        elif f_type == 'categorical':
            # Use MSE for now (could be replaced with cross-entropy if one-hot encoded)
            f_loss = F.mse_loss(
                recon_x[:, :, f],
                x[:, :, f],
                reduction='sum'
            )
        else:  # Default to MSE for continuous features
            f_loss = F.mse_loss(
                recon_x[:, :, f],
                x[:, :, f],
                reduction='sum'
            )

        # Apply feature weight
        weighted_f_loss = weight * f_loss
        recon_loss += weighted_f_loss
        feature_losses.append(f_loss.item() / batch_size)

    # Normalize by batch size
    recon_loss = recon_loss / batch_size

    # KL divergence with beta coefficient
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # Sparsity regularization (optional)
    sparsity_loss = sparsity_weight * torch.sum(torch.abs(mu)) / batch_size if sparsity_weight > 0 else 0

    # Combine losses with KL weight and beta
    total_loss = recon_loss + kl_weight * beta * kld_loss + sparsity_loss

    return total_loss, recon_loss, kld_loss, sparsity_loss, feature_losses


def train_vae_model(
        X_train,
        X_val,
        feature_names=None,
        latent_dim=64,  # Increased from 32
        hidden_dim=512,  # Increased from 256
        batch_size=32,  # Decreased from 64
        learning_rate=1e-4,  # Decreased from 5e-4
        sparsity_weight=0,  # Disabled initially
        num_epochs=200,  # Increased from 100
        kl_annealing=True,
        beta=1.0,  # Beta-VAE coefficient
        optimizer_type='adam',
        gradient_accumulation_steps=4,  # New parameter for gradient accumulation
        warmup_epochs=5  # New parameter for LR warmup
):
    """
    Trains an improved Variational Autoencoder (VAE) for time series data with mixed feature types.
    """
    # Detect feature types and preprocess data
    print("Detecting feature types and preprocessing data...")
    feature_types, processed_X_train = detect_and_preprocess_features(X_train, feature_names)
    _, processed_X_val = detect_and_preprocess_features(X_val, feature_names)

    # Convert processed data to tensors
    X_train_tensor = torch.tensor(processed_X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(processed_X_val, dtype=torch.float32)

    # Get data dimensions
    n_samples, seq_len, feature_dim = X_train.shape

    # Create dataset and loader
    dataset = TensorDataset(X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataset
    val_dataset = TensorDataset(X_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("Initializing VAE model...")
    model = TimeSeriesVAE(
        seq_len=seq_len,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        feature_types=feature_types
    ).to(device)

    # Use Adam optimizer with lower learning rate
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        if optimizer_type.lower() == 'adam'
        else torch.optim.SGD(model.parameters(), lr=learning_rate)
    )

    # Learning rate scheduler - more patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Logging
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'sparsity_loss': [],
        'val_loss': [],
        'feature_losses': []
    }

    # Variables for early stopping - increased patience
    early_stop_counter = 0
    best_loss = float('inf')
    patience = 15

    # Training loop with improved logging
    for epoch in range(num_epochs):
        # Apply learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor
                print(f"Warmup LR: {param_group['lr']:.6f}")

        # KL annealing - gradually increase from 0 to 1
        kl_weight = 1.0
        if kl_annealing:
            # More gradual annealing over half the epochs
            kl_weight = min(1.0, epoch / (num_epochs / 2))

        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_feature_losses = [0.0] * feature_dim

        # Gradient accumulation setup
        optimizer.zero_grad()
        accumulated_batches = 0

        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(device)

            # Forward pass with teacher forcing during training
            recon, mu, logvar = model(x, teacher_forcing_ratio=0.5)

            # Calculate loss
            loss, recon_loss, kl_loss, sparsity_loss, feature_losses = vae_loss(
                recon, x, mu, logvar, feature_types, sparsity_weight, kl_weight, beta
            )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Track losses
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item() if sparsity_weight > 0 else 0

            # Track individual feature losses
            for i, f_loss in enumerate(feature_losses):
                epoch_feature_losses[i] += f_loss

            # Accumulate gradients
            accumulated_batches += 1

            # Update weights after accumulating enough gradients
            if accumulated_batches == gradient_accumulation_steps or batch_idx == len(loader) - 1:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()
                optimizer.zero_grad()
                accumulated_batches = 0

        # Calculate average training losses
        avg_loss = epoch_loss / len(loader)
        avg_recon_loss = epoch_recon_loss / len(loader)
        avg_kl_loss = epoch_kl_loss / len(loader)
        avg_sparsity_loss = epoch_sparsity_loss / len(loader)
        avg_feature_losses = [f_loss / len(loader) for f_loss in epoch_feature_losses]

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, mu, logvar = model(x, teacher_forcing_ratio=0.0)  # No teacher forcing in validation
                loss, recon_l, kl_l, _, _ = vae_loss(
                    recon, x, mu, logvar, feature_types, sparsity_weight, kl_weight=1.0, beta=beta
                )
                val_loss += loss.item()
                val_recon_loss += recon_l.item()
                val_kl_loss += kl_l.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Log losses
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['sparsity_loss'].append(avg_sparsity_loss)
        history['val_loss'].append(avg_val_loss)
        history['feature_losses'].append(avg_feature_losses)

        # Print detailed progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Recon: {avg_val_recon_loss:.4f}, "
              f"Val KL: {avg_val_kl_loss:.4f}, KL Weight: {kl_weight:.2f}")

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save best model
            torch.save(model.state_dict(), 'best_vae_model.pth')
            early_stop_counter = 0
            print(f"New best model saved with validation loss: {best_loss:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break

    # Load best model
    model.load_state_dict(torch.load('best_vae_model.pth'))

    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(history['total_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.plot(history['kl_loss'], label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Loss')
    plt.tight_layout()
    plt.savefig('vae_loss_curves.png')
    plt.close('all')

    # Plot feature-specific losses
    if len(history['feature_losses']) > 0:
        plt.figure(figsize=(14, 8))
        for i in range(feature_dim):
            feature_loss_values = [epoch_losses[i] for epoch_losses in history['feature_losses']]
            feature_name = feature_names[i] if feature_names is not None and i < len(
                feature_names) else f"Feature {i + 1}"
            plt.plot(feature_loss_values, label=f"{feature_name} ({feature_types[i]})")
        plt.xlabel('Epoch')
        plt.ylabel('Feature Loss')
        plt.legend()
        plt.title('Feature-specific Reconstruction Losses')
        plt.tight_layout()
        plt.savefig('feature_loss_curves.png')
        plt.close('all')

    return model, history, feature_types


# Function to visualize reconstructions
def visualize_reconstructions(model, data, feature_names, n_samples=5):
    """
    Visualize original vs reconstructed time series with improved layout

    Args:
        model: The trained VAE model
        data: Input data to reconstruct (numpy array)
        feature_names: List of feature names
        n_samples: Number of sample windows to visualize
    """
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(data), n_samples, replace=False)
        samples = torch.tensor(data[indices], dtype=torch.float32).to(device)

        # Get reconstructions
        recon, _, _ = model(samples)

        # Convert to numpy for plotting
        samples_np = samples.cpu().numpy()
        recon_np = recon.cpu().numpy()

        # Number of features
        n_features = samples.shape[2]

        # Create PDF with matplotlib's PdfPages
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        with PdfPages('vae_reconstructions.pdf') as pdf:
            # Create one page per sample
            for i in range(n_samples):
                # Calculate grid dimensions - aim for roughly square layout
                n_cols = min(4, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

                # Create figure
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
                fig.suptitle(f"Sample {i + 1} - Original vs Reconstruction", fontsize=16)

                # Flatten axes if needed
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()

                # Plot each feature
                for f in range(n_features):
                    row, col = f // n_cols, f % n_cols
                    ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[f]

                    # Get data for this feature
                    orig_data = samples_np[i, :, f]
                    recon_data = recon_np[i, :, f]

                    # Calculate reconstruction error
                    mse = np.mean((orig_data - recon_data) ** 2)

                    # Check if feature appears to be binary/categorical (mostly 0s and 1s)
                    unique_vals = np.unique(orig_data)
                    is_binary = len(unique_vals) <= 5 and np.all(
                        np.logical_or(np.isclose(unique_vals, 0), np.isclose(unique_vals, 1)))

                    if is_binary:
                        # For binary data - use step plot
                        ax.step(range(len(orig_data)), orig_data, 'b-', where='post', label='Original')
                        ax.step(range(len(recon_data)), recon_data, 'r-', where='post', label='Reconstructed')
                        # Add threshold line for binary classification
                        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
                    else:
                        # For continuous data - use line plot
                        ax.plot(orig_data, 'b-', label='Original')
                        ax.plot(recon_data, 'r-', label='Reconstructed')

                    # Feature name as title with MSE
                    feature_name = feature_names[f] if f < len(feature_names) else f"Feature {f + 1}"
                    ax.set_title(f"{feature_name} (MSE: {mse:.4f})")

                    # Only add legend to first plot
                    if f == 0:
                        ax.legend()

                    # Add grid for better visibility
                    ax.grid(True, alpha=0.3)

                # Remove empty subplots
                for f in range(n_features, n_rows * n_cols):
                    row, col = f // n_cols, f % n_cols
                    ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[f]
                    ax.axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Generated visualization PDF with {n_samples} samples")


# Function to analyze the latent space
def analyze_latent_space(model, data):
    """Analyze latent space by encoding data"""
    model.eval()
    with torch.no_grad():
        # Convert data to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

        # Encode data
        mu, logvar = model.encode(data_tensor)

        # Plot latent space distribution
        plt.figure(figsize=(10, 8))

        # If latent dim > 2, use PCA to visualize
        if mu.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            mu_2d = pca.fit_transform(mu.cpu().numpy())
            plt.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.5)
            plt.title('Latent Space Distribution (PCA)')
        else:
            plt.scatter(mu[:, 0].cpu().numpy(), mu[:, 1].cpu().numpy(), alpha=0.5)
            plt.title('Latent Space Distribution')

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.savefig('latent_space.png')
        plt.close('all')  # Close all figures

        # Check latent space sparsity
        avg_active = (torch.abs(mu) > 0.01).float().mean(dim=0)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(avg_active)), avg_active.cpu().numpy())
        plt.title('Average Activation of Latent Dimensions')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Activation Frequency')
        plt.tight_layout()
        plt.savefig('latent_sparsity.png')
        plt.close('all')  # Close all figures


print("Training VAE model with improved architecture...")
vae_model, training_history, feature_types = train_vae_model(
    X_train=mydata.X_train,
    X_val=mydata.X_val,
    latent_dim=32,
    hidden_dim=256,
    batch_size=64,        # Reduced batch size
    learning_rate=5e-4,     # Reduced learning rate
    sparsity_weight=0.0001, # Significantly reduced sparsity weight
    num_epochs=100,
    kl_annealing=True,     # Enable KL annealing
    optimizer_type='adam'
)

print("Visualizing reconstructions...")
visualize_reconstructions(vae_model, mydata.X_val, feature_names=mydata.feature_names)

print("Analyzing latent space...")
analyze_latent_space(vae_model, mydata.X_val)
print("VAE training complete!")

