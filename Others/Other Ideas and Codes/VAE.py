import sys
import os
os.getcwd()
os.chdir('./phd/Paper 2/')

# Get the absolute path of the current working directory
current_dir = os.getcwd()
# Append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(current_dir, '.')))

import pandas as pd
from Functions import BDG2, preprocess, models
import numpy as np


df = pd.read_csv('./Data/building-data-genome-project-2/electricity_cleaned.txt')

target_column = 'energy_consumption'

df_name = "industrial"; option_number=1 #PrimaryUses Manufacturing corresponds to 'industrial'
# df_name = "health"; option_number=1 #PrimaryUses Medical Clinic corresponds to 'health'
# df_name = "retail"; option_number=3 #PrimaryUses Retail Store corresponds to 'retail'
# df_name = "office"; option_number=1 #PrimaryUses Office corresponds to 'office'
cleaned_df, more_info = BDG2.get_column_by_criteria(df, primary_use=df_name, option_number=option_number)

#scale type = both, features, outcome
X_train, y_train, X_val, y_val, X_test, y_test, X, y, df_scaled = preprocess.load_and_preprocess_data_with_sequences(cleaned_df, target=target_column, scaled=True, scale_type='features', val_ratio=0.1, test_ratio=0.1, input_seq_length=48, output_seq_length=1)

data = (X_train, y_train, X_val, y_val, X_test, y_test)

# Print shapes to verify
print("\nShapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

df_scaled.head()


model, metrics, history = models.implement_tcn(data)

# Full VAE Implementation and Explainability Workflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

# -----------------------------------
# 1. Define VAE Architecture
# -----------------------------------

# Input shape for the time-series data
input_shape = (48, 10)  # (time_steps, features)
latent_dim = 16         # Latent space dimension

# Encoder
encoder_inputs = layers.Input(shape=input_shape, name='encoder_input')
x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(encoder_inputs)
x = layers.MaxPooling1D(pool_size=2, padding='same')(x)  # (24, 32)
x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(pool_size=2, padding='same')(x)  # (12, 64)
x = layers.Flatten()(x)                                  # (768,)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer using the reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Encoder model
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(12 * 64, activation='relu')(latent_inputs)  # (768,)
x = layers.Reshape((12, 64))(x)                              # (12, 64)
x = layers.Conv1DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)  # (12, 64)
x = layers.UpSampling1D(size=2)(x)                                             # (24, 64)
x = layers.Conv1DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)  # (24, 32)
x = layers.UpSampling1D(size=2)(x)                                             # (48, 32)
decoder_outputs = layers.Conv1DTranspose(10, kernel_size=3, activation='linear', padding='same')(x)  # (48, 10)

# Decoder model
decoder = Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()

# VAE model
outputs = decoder(z)
vae = Model(encoder_inputs, outputs, name='vae')
vae.summary()

# -----------------------------------
# 2. Loss Function Integration
# -----------------------------------

# Reconstruction loss: MSE between input and output
reconstruction_loss = K.sum(K.square(encoder_inputs - outputs), axis=[1, 2])  # Sum over time and features
reconstruction_loss = K.mean(reconstruction_loss)  # Mean over the batch

# KL divergence loss
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)  # Sum over latent dimensions
kl_loss = -0.5 * K.mean(kl_loss)   # Mean over the batch

# Total VAE loss
vae_loss = reconstruction_loss + kl_loss

# Add the loss to the model
vae.add_loss(vae_loss)

# Compile the VAE
vae.compile(optimizer='adam')

# -----------------------------------
# 3. Data Preparation
# -----------------------------------

# Create dummy data for demonstration
X_train = np.random.rand(13962, 48, 10).astype(np.float32)  # Training data
X_val = np.random.rand(1745, 48, 10).astype(np.float32)     # Validation data

# -----------------------------------
# 4. Train the VAE
# -----------------------------------

# Train the VAE on the training data
history = vae.fit(
    X_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, None)  # No target labels needed
)

# -----------------------------------
# 5. Summary of Fix
# -----------------------------------

# Key Fix: The reconstruction loss is now computed directly as part of the model's `add_loss` method.
# This avoids passing inputs and outputs directly to the loss function outside the model, preventing errors related to `NoneType`.
