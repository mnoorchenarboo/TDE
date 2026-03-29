import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import time

# 1. Defining the TCN with Multi-Head Attention Model

class ResidualBlock(layers.Layer):
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, padding=padding)
        self.batch_norm1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, padding=padding)
        self.batch_norm2 = layers.BatchNormalization()
        self.activation2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(dropout_rate)

        self.downsample = layers.Conv1D(filters=nb_filters, kernel_size=1, padding='same')
        self.activation3 = layers.Activation('relu')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)

        # If input shape doesn't match the output shape, use a 1x1 conv for linear projection
        if inputs.shape[-1] != x.shape[-1]:
            inputs = self.downsample(inputs)

        return self.activation3(x + inputs)


class TCN(layers.Layer):
    def __init__(self, nb_filters, kernel_size, nb_stacks, dilations, padding='causal', dropout_rate=0.1):
        super(TCN, self).__init__()
        self.blocks = []
        self.nb_stacks = nb_stacks

        for s in range(nb_stacks):
            for d in dilations:
                self.blocks.append(ResidualBlock(d, nb_filters, kernel_size, padding, dropout_rate))

    def call(self, inputs, training=None):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x


class EnhancedMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, temperature=1.0):
        super(EnhancedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.temperature = temperature  # Temperature parameter for softmax

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

        # Add attention weights as a layer variable
        self.attention_weights = None

        # Temporal positional encoding
        self.supports_positional_encoding = True

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, temporal_bias=None):
        batch_size = tf.shape(query)[0]

        q = self.wq(query)  # (batch_size, seq_len, d_model)
        k = self.wk(key)  # (batch_size, seq_len, d_model)
        v = self.wv(value)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / (tf.math.sqrt(dk) * self.temperature)

        # Apply temporal bias if provided
        if temporal_bias is not None:
            # Reshape temporal_bias for broadcasting: [batch_size, 1, 1, seq_len]
            bias_expanded = tf.reshape(temporal_bias, [batch_size, 1, 1, -1])
            # Apply the bias to the attention logits
            scaled_attention_logits = scaled_attention_logits + bias_expanded

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        self.attention_weights = attention_weights

        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

    def get_attention_weights(self):
        return self.attention_weights


class TCNWithAttention(Model):
    def __init__(self, nb_filters, kernel_size, nb_stacks, dilations,
                 input_shape, output_shape, num_heads=4, d_model=64,
                 dropout_rate=0.1, attention_temperature=1.0):
        super(TCNWithAttention, self).__init__()
        self.input_shape_val = input_shape
        self.output_shape_val = output_shape

        self.tcn = TCN(nb_filters, kernel_size, nb_stacks, dilations, dropout_rate=dropout_rate)

        # Embedding layer to project input to d_model dimensions
        self.embedding = layers.Dense(d_model)

        # Enhanced Multi-head attention with temperature parameter
        self.attention = EnhancedMultiHeadAttention(d_model, num_heads, temperature=attention_temperature)

        # Temporal bias parameters (learnable)
        self.seq_len = input_shape[0]
        self.temporal_bias = self.add_weight(
            shape=(1, self.seq_len),
            initializer=tf.keras.initializers.Constant(
                value=np.linspace(0, 1, self.seq_len).reshape(1, -1)
            ),
            trainable=True,
            name="temporal_bias"
        )

        # Final output layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(np.prod(output_shape))
        self.reshape = layers.Reshape(output_shape)

    def call(self, inputs, training=None):
        x = self.tcn(inputs, training=training)

        # Project to embedding dimension
        x_emb = self.embedding(x)

        # Create batch-sized temporal bias
        batch_size = tf.shape(inputs)[0]
        batch_temporal_bias = tf.repeat(self.temporal_bias, batch_size, axis=0)

        # Apply multi-head attention with temporal bias
        attn_output = self.attention(x_emb, x_emb, x_emb, temporal_bias=batch_temporal_bias)

        # Combine with TCN output (residual connection)
        x = x + attn_output

        # Output layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.reshape(x)

    def build_graph(self):
        input_shape = (None,) + self.input_shape_val
        self.build(input_shape)


# 2. Implementing Integrated Gradients with caching

class IntegratedGradients:
    def __init__(self, model, baseline=None, cache_dir=None):
        self.model = model
        self.baseline = baseline
        self.cache_dir = cache_dir

        # Create cache directory if specified
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def interpolate(self, inputs, baseline, alphas):
        """Interpolate between inputs and baseline using alphas."""
        alphas_x = tf.reshape(alphas, [-1, 1, 1])
        delta = inputs - baseline
        return baseline + alphas_x * delta

    def _compute_gradients(self, inputs):
        """Compute gradients of model output with respect to inputs."""
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.model(inputs)
            # Using mean of outputs for simplicity - can be modified for different targets
            target_outputs = tf.reduce_mean(outputs, axis=[1, 2])

        return tape.gradient(target_outputs, inputs)

    def _get_cache_filename(self, inputs_hash, m_steps):
        return f"ig_cache_{inputs_hash}_{m_steps}.pkl"

    def _compute_inputs_hash(self, inputs):
        """Compute a hash for inputs to use as cache key."""
        # Simple hash based on shape and first/last few values
        input_shape = inputs.shape
        sample_values = np.concatenate([
            inputs[:min(2, len(inputs))].flatten()[:10],  # First 2 samples, first 10 values
            inputs[-min(2, len(inputs)):].flatten()[-10:]  # Last 2 samples, last 10 values
        ])

        # Create string representation and hash
        shape_str = "_".join([str(dim) for dim in input_shape])
        values_str = "_".join([f"{val:.6f}" for val in sample_values])
        combined = f"{shape_str}_{values_str}"

        import hashlib
        return hashlib.md5(combined.encode()).hexdigest()

    def calculate_integrated_gradients(self, inputs, baseline=None, m_steps=50, batch_size=8, use_cache=True):
        """
        Calculate integrated gradients for given inputs, with caching support.
        """
        # Check cache if enabled
        if use_cache and self.cache_dir is not None:
            inputs_hash = self._compute_inputs_hash(inputs)
            cache_file = os.path.join(self.cache_dir, self._get_cache_filename(inputs_hash, m_steps))

            if os.path.exists(cache_file):
                print(f"Loading integrated gradients from cache: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading from cache: {e}. Recalculating...")

        if baseline is None:
            if self.baseline is None:
                self.baseline = np.zeros_like(inputs)
            baseline = self.baseline

        # Convert inputs and baseline to tensors if they're not already
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        # Generate alphas for integral approximation
        alphas = tf.linspace(0.0, 1.0, m_steps)

        # Initialize attributions
        total_attributions = np.zeros_like(inputs)

        # Process in batches to avoid memory issues
        num_samples = inputs.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for batch_idx in tqdm(range(num_batches), desc="Calculating Integrated Gradients"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_inputs = inputs[start_idx:end_idx]
            batch_baseline = baseline[start_idx:end_idx]

            batch_attributions = np.zeros_like(batch_inputs)

            # For each alpha, calculate interpolated inputs and gradients
            for alpha_idx, alpha in enumerate(alphas):
                # Create alpha tensor with proper shape for each batch
                alpha_batch = tf.ones((end_idx - start_idx,), dtype=tf.float32) * alpha

                # Create interpolated inputs properly
                interpolated_inputs = self.interpolate(batch_inputs, batch_baseline, alpha_batch)

                # Compute gradients
                gradients = self._compute_gradients(interpolated_inputs)
                if gradients is not None:  # Handle cases where gradients might be None
                    batch_attributions += gradients.numpy()

            # Average gradients across steps and multiply by input-baseline difference
            batch_attributions = batch_attributions / m_steps
            batch_attributions = batch_attributions * (batch_inputs.numpy() - batch_baseline.numpy())

            total_attributions[start_idx:end_idx] = batch_attributions

        # Cache the result if enabled
        if use_cache and self.cache_dir is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(total_attributions, f)
                print(f"Saved integrated gradients to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving to cache: {e}")

        return total_attributions


# 3. Function to Update Attention Weights Based on Integrated Gradients

def update_attention_with_ig(model, X_sample, ig_attributions, alpha=0.5):
    """
    Updates attention weights in the model based on integrated gradients attributions,
    with emphasis on temporal relationships.

    Args:
        model: The TCNWithAttention model
        X_sample: Sample input data used to trigger attention mechanisms
        ig_attributions: Integrated gradients attributions
        alpha: Blending factor between original and IG-guided attention weights
    """
    # Normalize attributions across features and time steps
    importance_across_time = np.abs(ig_attributions).sum(axis=2)

    # Get sequence length
    seq_len = X_sample.shape[1]

    # Create temporal bias that emphasizes recent timesteps
    # This will enhance the importance of recent timesteps
    temporal_bias = np.linspace(0.2, 1.0, seq_len)

    # Apply temporal bias to feature importance
    for i in range(len(importance_across_time)):
        importance_across_time[i] = importance_across_time[i] * temporal_bias

    # Ensure minimum importance for all timesteps to prevent attention collapse
    min_importance = 0.1
    importance_across_time = np.maximum(importance_across_time, min_importance)

    # Normalize to [0, 1] scale
    max_importance = np.max(importance_across_time, axis=1, keepdims=True)
    importance_across_time = importance_across_time / (max_importance + 1e-10)

    # Update the model's temporal bias weights directly
    # This is more efficient than modifying attention weights during each forward pass
    # Convert to appropriate tensor and update
    new_temporal_bias = tf.convert_to_tensor(
        np.mean(importance_across_time, axis=0, keepdims=True),
        dtype=tf.float32
    )

    # Use a weighted average to update the model's temporal bias
    current_bias = model.temporal_bias.numpy()
    updated_bias = (1 - alpha) * current_bias + alpha * new_temporal_bias

    # Assign the updated bias to the model
    model.temporal_bias.assign(updated_bias)

    print("Attention temporal bias updated based on integrated gradients attributions.")


# 4. Baseline calculation with caching

def calculate_or_load_baseline(baseline_type, X_data, save_dir="./baselines/"):
    """
    Calculate baseline or load from cache if available.

    Args:
        baseline_type: Type of baseline ('zero', 'mean', etc.)
        X_data: Input data to calculate baseline for
        save_dir: Directory to save/load baseline calculations

    Returns:
        Baseline data and convergence information
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create a simple hash based on data shape and type
    data_shape = "_".join([str(dim) for dim in X_data.shape])
    filename = f"{baseline_type}_baseline_{data_shape}.pkl"
    filepath = os.path.join(save_dir, filename)

    # Check if baseline file exists
    if os.path.exists(filepath):
        print(f"Loading cached {baseline_type} baseline...")
        with open(filepath, 'rb') as f:
            baseline_data = pickle.load(f)

        steps = baseline_data.get('steps', 50)
        print(f"Baseline: {baseline_type} - Final Steps: {steps}, Time: Loaded from cache")
        return baseline_data

    # Calculate baseline
    print(f"Processing {baseline_type} baseline...")
    start_time = time.time()

    # Initialize baseline data
    if baseline_type == 'zero':
        baseline_result = np.zeros_like(X_data)
        steps_list = list(range(10, 51))
        # Simulate convergence data (replace with real calculations for actual use)
        differences = [0.01 / (i + 1) for i in range(len(steps_list))]

    elif baseline_type == 'mean':
        baseline_result = np.mean(X_data, axis=0, keepdims=True)
        baseline_result = np.repeat(baseline_result, X_data.shape[0], axis=0)
        steps_list = list(range(10, 51))
        # Simulate convergence data (replace with real calculations for actual use)
        differences = [0.0001 / (i + 1) for i in range(len(steps_list))]

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Create baseline data with convergence information
    baseline_data = {
        'baseline': baseline_result,
        'steps': 50,
        'time': elapsed_time,
        'convergence': {
            'steps_list': steps_list,
            'differences': differences
        }
    }

    # Save for future use
    with open(filepath, 'wb') as f:
        pickle.dump(baseline_data, f)

    print(f"Baseline: {baseline_type} - Final Steps: {baseline_data['steps']}, Time: {baseline_data['time']:.2f} sec")
    return baseline_data


# 5. Main Function to Train and Improve Model with Integrated Gradients

def train_and_improve_with_ig(X_train, y_train, X_val, y_val, X_test, y_test,
                              epochs=50, batch_size=32, visualize=False,
                              use_cache=True, cache_dir="./cache"):
    """
    Complete pipeline to train a TCN with attention and then improve it using integrated gradients.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size for training
        visualize: Whether to generate visualizations
        use_cache: Whether to use caching for IG calculations
        cache_dir: Directory for caching
    """
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]

    # Create cache directories
    os.makedirs(cache_dir, exist_ok=True)
    baseline_dir = os.path.join(cache_dir, "baselines")
    ig_cache_dir = os.path.join(cache_dir, "integrated_gradients")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(ig_cache_dir, exist_ok=True)

    # Calculate or load baselines
    zero_baseline_data = calculate_or_load_baseline('zero', X_train, save_dir=baseline_dir)
    mean_baseline_data = calculate_or_load_baseline('mean', X_train, save_dir=baseline_dir)

    # Extract the actual baseline arrays
    zero_baseline = zero_baseline_data['baseline']
    mean_baseline = mean_baseline_data['baseline']

    # 1. Create and train the initial model
    print("Creating and training initial TCN with attention model...")
    model = TCNWithAttention(
        nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dilations=[1, 2, 4, 8],
        input_shape=input_shape,
        output_shape=output_shape,
        num_heads=4,
        d_model=64,
        attention_temperature=0.7  # Lower temperature for sharper attention
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Build the model graph
    model.build_graph()

    # Train the initial model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    initial_history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # 2. Evaluate the initial model
    initial_loss, initial_mae = model.evaluate(X_test, y_test)
    print(f"Initial model - Test Loss: {initial_loss:.4f}, Test MAE: {initial_mae:.4f}")

    # 3. Calculate integrated gradients
    print("Calculating integrated gradients on a subset of training data...")
    # Use a subset of training data to calculate IG for efficiency
    n_samples_for_ig = min(500, len(X_train))
    X_sample = X_train[:n_samples_for_ig]

    # Initialize IG calculator with mean baseline and caching
    ig_calculator = IntegratedGradients(
        model,
        baseline=mean_baseline[:n_samples_for_ig],
        cache_dir=ig_cache_dir if use_cache else None
    )

    # Calculate attributions with robust error handling
    try:
        attributions = ig_calculator.calculate_integrated_gradients(
            X_sample,
            baseline=mean_baseline[:n_samples_for_ig],
            m_steps=20,  # Lower for efficiency, increase for accuracy
            batch_size=16,
            use_cache=use_cache
        )

        # Check if attributions contain NaN values
        if np.isnan(attributions).any():
            print("Warning: NaN values detected in attributions. Using absolute values of inputs as attributions.")
            attributions = np.abs(X_sample)
    except Exception as e:
        print(f"Error calculating integrated gradients: {e}")
        print("Using absolute values of inputs as attributions instead.")
        attributions = np.abs(X_sample)

    # 4. Update attention weights based on IG attributions
    print("Updating attention weights based on integrated gradients...")
    update_attention_with_ig(model, X_sample, attributions, alpha=0.3)

    # 5. Fine-tune the model with updated attention weights
    print("Fine-tuning model with IG-guided attention...")
    # We're not reinitializing the model - we're continuing training with the updated attention
    fine_tune_history = model.fit(
        X_train, y_train,
        epochs=epochs // 2,  # Fewer epochs for fine-tuning
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # 6. Evaluate the improved model
    improved_loss, improved_mae = model.evaluate(X_test, y_test)
    print(f"Improved model - Test Loss: {improved_loss:.4f}, Test MAE: {improved_mae:.4f}")
    print(
        f"Improvement: Loss reduced by {initial_loss - improved_loss:.4f}, MAE reduced by {initial_mae - improved_mae:.4f}")

    # 7. Visualize the attributions and attention weights - only if requested
    if visualize:
        try:
            visualize_attributions_and_attention(X_sample[:3], attributions[:3], model)
        except Exception as e:
            print(f"Error visualizing attributions and attention: {e}")

    return model, initial_history, fine_tune_history


# 6. Improved Visualization Function

def visualize_attributions_and_attention(X_sample, attributions, model, save_path=None):
    """
    Visualize feature attributions from integrated gradients and attention weights
    with option to save instead of displaying
    """
    n_samples = len(X_sample)
    time_steps = X_sample.shape[1]
    n_features = X_sample.shape[2]

    # Get attention weights
    _ = model(X_sample)
    attention_weights = model.attention.get_attention_weights().numpy()

    # Get temporal bias
    temporal_bias = model.temporal_bias.numpy()[0]

    # Set up the plots
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 5 * n_samples))

    # Handle case of a single sample
    if n_samples == 1:
        axes = np.array([axes])

    for i in range(n_samples):
        # 1. Plot feature attributions
        # Sum absolute attributions across features
        feature_importance = np.abs(attributions[i]).sum(axis=1)
        ax1 = axes[i, 0]
        ax1.bar(range(time_steps), feature_importance)
        ax1.set_title(f'Sample {i + 1}: Feature Importance from IG')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Importance')

        # 2. Plot attention heatmap (using first head for simplicity)
        ax2 = axes[i, 1]
        head_idx = 0  # First attention head
        im = ax2.imshow(attention_weights[i, head_idx], cmap='viridis')
        ax2.set_title(f'Sample {i + 1}: Attention Weights (Head {head_idx + 1})')
        ax2.set_xlabel('Time Step (Key)')
        ax2.set_ylabel('Time Step (Query)')
        plt.colorbar(im, ax=ax2)

        # 3. Plot temporal bias
        ax3 = axes[i, 2]
        ax3.plot(range(time_steps), temporal_bias, 'r-', linewidth=2)
        ax3.set_title('Temporal Bias Weights')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Bias Weight')
        ax3.grid(True)

    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


# 7. Comprehensive Example Usage Function

def run_experiment(dataset_info, exp_name=None, visualize=False,
                   use_cache=True, cache_dir="./cache",
                   save_model=True, save_results=True):
    """
    Comprehensive function to run experiments with all options

    Args:
        dataset_info: Dictionary with dataset information or data tuple (X_train, y_train, X_val, y_val, X_test, y_test)
        exp_name: Name for the experiment (used for saving results)
        visualize: Whether to generate visualization plots
        use_cache: Whether to cache baseline and IG calculations
        cache_dir: Directory for caching
        save_model: Whether to save the trained model
        save_results: Whether to save results and plots
    """
    if exp_name is None:
        exp_name = "tcn_attention_experiment"

    print(f"Running experiment: {exp_name}")

    # 1. Setup directories
    results_dir = f"./results/{exp_name}"
    if save_results or save_model:
        os.makedirs(results_dir, exist_ok=True)

    # 2. Load data
    if isinstance(dataset_info, dict):
        # Assume dataset_info has a 'type' key to specify which dataset to load
        if 'type' in dataset_info:
            from Functions import main
            print(f"Loading {dataset_info['type']} dataset...")
            mydata = main.load_and_preprocess_data(dataset_type=dataset_info['type'])
            X_train, y_train = mydata.X_train, mydata.y_train
            X_val, y_val = mydata.X_val, mydata.y_val
            X_test, y_test = mydata.X_test, mydata.y_test
        else:
            # Assume dataset_info contains the data directly
            X_train, y_train = dataset_info.get('X_train'), dataset_info.get('y_train')
            X_val, y_val = dataset_info.get('X_val'), dataset_info.get('y_val')
            X_test, y_test = dataset_info.get('X_test'), dataset_info.get('y_test')
    else:
        # Assume dataset_info is a tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset_info

    print("Shapes:")
    print(f"X: {np.shape(np.concatenate([X_train, X_val, X_test], axis=0))}, "
          f"y: {np.shape(np.concatenate([y_train, y_val, y_test], axis=0))}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 3. Run the training and improvement process
    model, initial_history, fine_tune_history = train_and_improve_with_ig(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=50,
        batch_size=32,
        visualize=visualize,
        use_cache=use_cache,
        cache_dir=cache_dir
    )

    # 4. Save model if requested
    if save_model:
        model_path = os.path.join(results_dir, f"{exp_name}_model")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # 5. Save results if requested
    if save_results:
        # Save history
        history_data = {
            'initial': {
                'loss': initial_history.history['loss'],
                'val_loss': initial_history.history['val_loss'],
                'mae': initial_history.history['mae'],
                'val_mae': initial_history.history['val_mae']
            },
            'fine_tune': {
                'loss': fine_tune_history.history['loss'],
                'val_loss': fine_tune_history.history['val_loss'],
                'mae': fine_tune_history.history['mae'],
                'val_mae': fine_tune_history.history['val_mae']
            }
        }

        with open(os.path.join(results_dir, f"{exp_name}_history.pkl"), 'wb') as f:
            pickle.dump(history_data, f)

        # Plot and save training history
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(initial_history.history['loss'], label='Initial Training')
        plt.plot(initial_history.history['val_loss'], label='Initial Validation')
        plt.plot(range(len(initial_history.history['loss']),
                       len(initial_history.history['loss']) + len(fine_tune_history.history['loss'])),
                 fine_tune_history.history['loss'], label='Fine-Tune Training')
        plt.plot(range(len(initial_history.history['val_loss']),
                       len(initial_history.history['val_loss']) + len(fine_tune_history.history['val_loss'])),
                 fine_tune_history.history['val_loss'], label='Fine-Tune Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot MAE
        plt.subplot(2, 2, 2)
        plt.plot(initial_history.history['mae'], label='Initial Training')
        plt.plot(initial_history.history['val_mae'], label='Initial Validation')
        plt.plot(range(len(initial_history.history['mae']),
                       len(initial_history.history['mae']) + len(fine_tune_history.history['mae'])),
                 fine_tune_history.history['mae'], label='Fine-Tune Training')
        plt.plot(range(len(initial_history.history['val_mae']),
                       len(initial_history.history['val_mae']) + len(fine_tune_history.history['val_mae'])),
                 fine_tune_history.history['val_mae'], label='Fine-Tune Validation')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{exp_name}_training_history.png"))
        plt.close()

        print(f"Results saved to {results_dir}")

    return model, initial_history, fine_tune_history


# 8. Function to make predictions with visualization of attention

def predict_with_attention_visualization(model, X_input, true_y=None, n_samples=3, save_path=None):
    """
    Make predictions with the model and visualize attention weights to explain the predictions.

    Args:
        model: Trained TCNWithAttention model
        X_input: Input data for prediction
        true_y: Optional true values for comparison
        n_samples: Number of samples to visualize
        save_path: Path to save visualization (None to display)

    Returns:
        Predictions and visualization
    """
    # Make predictions
    predictions = model(X_input).numpy()

    # Get attention weights
    _ = model(X_input)
    attention_weights = model.attention.get_attention_weights().numpy()

    # Get temporal bias
    temporal_bias = model.temporal_bias.numpy()[0]

    # Limit the number of samples to visualize
    n_samples = min(n_samples, len(X_input))
    time_steps = X_input.shape[1]

    # Set up the plots - include prediction and ground truth if available
    n_cols = 3 if true_y is None else 4
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))

    # Handle case of a single sample
    if n_samples == 1:
        axes = np.array([axes])

    for i in range(n_samples):
        # 1. Plot input features
        ax1 = axes[i, 0]
        for j in range(X_input.shape[2]):  # Plot each feature
            ax1.plot(range(time_steps), X_input[i, :, j], label=f'Feature {j + 1}')
        ax1.set_title(f'Sample {i + 1}: Input Features')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        if X_input.shape[2] <= 10:  # Only show legend if not too many features
            ax1.legend()

        # 2. Plot attention weights (using first head)
        ax2 = axes[i, 1]
        head_idx = 0  # First attention head
        im = ax2.imshow(attention_weights[i, head_idx], cmap='viridis')
        ax2.set_title(f'Attention Weights (Head {head_idx + 1})')
        ax2.set_xlabel('Time Step (Key)')
        ax2.set_ylabel('Time Step (Query)')
        plt.colorbar(im, ax=ax2)

        # 3. Plot temporal bias
        ax3 = axes[i, 2]
        ax3.plot(range(time_steps), temporal_bias, 'r-', linewidth=2)
        ax3.set_title('Temporal Bias')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Bias Weight')
        ax3.grid(True)

        # 4. Plot prediction vs ground truth if available
        if true_y is not None:
            ax4 = axes[i, 3]
            # Handle different output shapes
            if len(predictions.shape) == 3:  # Multiple timestep outputs
                for j in range(predictions.shape[1]):  # For each output timestep
                    ax4.plot(range(predictions.shape[2]), predictions[i, j], 'b-', label=f'Pred t+{j + 1}')
                    ax4.plot(range(true_y.shape[2]), true_y[i, j], 'g--', label=f'True t+{j + 1}')
            else:  # Single value output
                ax4.plot(predictions[i], 'b-', label='Prediction')
                ax4.plot(true_y[i], 'g--', label='Ground Truth')

            ax4.set_title('Prediction vs Ground Truth')
            ax4.legend()
            ax4.grid(True)

    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()

    return predictions


# 9. Function to evaluate model performance with metrics

def evaluate_model_performance(model, X_test, y_test, save_path=None):
    """
    Evaluate model performance with various metrics and plots.

    Args:
        model: Trained model
        X_test: Test input data
        y_test: Test target data
        save_path: Path to save evaluation results

    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)

    # Calculate R^2 score
    y_mean = np.mean(y_test)
    ss_total = np.sum((y_test - y_mean) ** 2)
    ss_residual = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Print metrics
    print(f"Model Evaluation Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R^2 Score: {r2:.6f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))

    # Prepare data for plotting
    # This flattens multi-dimensional outputs if necessary
    y_test_flat = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # Limit points to plot if too many
    max_points = 1000
    if len(y_test_flat) > max_points:
        indices = np.random.choice(len(y_test_flat), max_points, replace=False)
        y_test_flat = y_test_flat[indices]
        y_pred_flat = y_pred_flat[indices]

    # Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5)

    # Add perfect prediction line
    min_val = min(np.min(y_test_flat), np.min(y_pred_flat))
    max_val = max(np.max(y_test_flat), np.max(y_pred_flat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)

    # Plot error distribution
    plt.subplot(1, 2, 2)
    errors = y_test_flat - y_pred_flat
    plt.hist(errors, bins=50, alpha=0.75)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Evaluation plots saved to {save_path}")
    else:
        plt.show()

    # Return metrics
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    return metrics


# 10. Main function to run the entire pipeline

def main(dataset_type='synthetic', run_name=None, epochs=50, use_cache=True, save_results=True):
    """
    Main function to run the entire pipeline.

    Args:
        dataset_type: Type of dataset to use ('synthetic', 'real', etc.)
        run_name: Name for the run (for saving results)
        epochs: Number of training epochs
        use_cache: Whether to use caching
        save_results: Whether to save results
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create run name if not provided
    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{dataset_type}_{timestamp}"

    # Create directories
    results_dir = f"./results/{run_name}"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    # Log run information
    print(f"Starting run: {run_name}")
    print(f"Dataset: {dataset_type}")
    print(f"Epochs: {epochs}")

    # Load data
    from Functions import main
    print(f"Loading {dataset_type} dataset...")
    mydata = main.load_and_preprocess_data(dataset_type=dataset_type)
    X_train, y_train = mydata.X_train, mydata.y_train
    X_val, y_val = mydata.X_val, mydata.y_val
    X_test, y_test = mydata.X_test, mydata.y_test

    # Run experiment
    model, initial_history, fine_tune_history = run_experiment(
        (X_train, y_train, X_val, y_val, X_test, y_test),
        exp_name=run_name,
        visualize=save_results,
        use_cache=use_cache,
        cache_dir=f"./cache/{run_name}",
        save_model=save_results,
        save_results=save_results
    )

    # Additional evaluation
    if save_results:
        eval_path = os.path.join(results_dir, f"{run_name}_evaluation.png")
    else:
        eval_path = None

    metrics = evaluate_model_performance(model, X_test, y_test, save_path=eval_path)

    # Make predictions with visualization
    n_samples_to_viz = min(5, len(X_test))
    X_viz = X_test[:n_samples_to_viz]
    y_viz = y_test[:n_samples_to_viz]

    if save_results:
        pred_viz_path = os.path.join(results_dir, f"{run_name}_predictions.png")
    else:
        pred_viz_path = None

    predict_with_attention_visualization(model, X_viz, true_y=y_viz, save_path=pred_viz_path)

    # Save experiment summary
    if save_results:
        # Create summary dictionary
        summary = {
            'run_name': run_name,
            'dataset': dataset_type,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'data_shapes': {
                'X_train': X_train.shape,
                'y_train': y_train.shape,
                'X_val': X_val.shape,
                'y_val': y_val.shape,
                'X_test': X_test.shape,
                'y_test': y_test.shape
            },
            'metrics': metrics,
            'model_config': {
                'nb_filters': 64,
                'kernel_size': 3,
                'nb_stacks': 1,
                'dilations': [1, 2, 4, 8],
                'num_heads': 4,
                'd_model': 64,
                'attention_temperature': 0.7
            }
        }

        # Save as JSON
        import json
        with open(os.path.join(results_dir, f"{run_name}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4, default=str)

        print(f"Experiment summary saved to {results_dir}")

    print(f"Run {run_name} completed successfully!")
    return model, metrics


# Run the code if module is executed as main
import argparse

parser = argparse.ArgumentParser(description='Run TCN with Attention and Integrated Gradients')
parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset type (synthetic, real, etc.)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--run_name', type=str, default=None, help='Name for this run')
parser.add_argument('--no_cache', action='store_true', help='Disable caching')
parser.add_argument('--no_save', action='store_true', help='Disable saving results')

args = parser.parse_args()

main(
    dataset_type=args.dataset,
    run_name=args.run_name,
    epochs=args.epochs,
    use_cache=not args.no_cache,
    save_results=not args.no_save
)