import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from Functions import main
from tensorflow.keras.models import Sequential, load_model, Model
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*The default value of non_blocking.*")

dataset_types = ["Residential", "Manufacturing facility", "Office building", "Retail store", "Medical clinic"]
model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']

dataset_type = dataset_types[4]

os.makedirs(f'./Results/FastSHAP/{dataset_type}', exist_ok=True)
mydata = main.load_and_preprocess_data(dataset_type=dataset_type)
model_dir = f"./Results/Models/{mydata.data_type}"

for model_type in model_types:
    # model_type = model_types[9]
    best_model = load_model(f"{model_dir}/{model_type}.keras")

    # Define prediction function
    def model_predict(X):
        # Reshape if needed (handle both 2D and 3D inputs)
        if X.ndim == 2:
            X = X.reshape(-1, mydata.X.shape[1], mydata.X.shape[2])
        return best_model.predict(X, verbose=0)[:, 0]  # Return first horizon predictions

    if os.path.exists(f"./Results/FastSHAP/{mydata.data_type}/{model_type}.pt"):
        print(f"Explainer for {model_type} exist. Skipped training ...")
    else:
        explainer = main.TimeSeriesSHAPExplainerUpdated(
            # Training settings (optimized for speed)
            n_epochs=300,  # Reduced from 200 (early stopping often kicks in sooner)
            batch_size=512,  # Kept large for GPU efficiency
            patience=5,  # Slightly more aggressive early stopping
            delta=1e-2,
            verbose=True,
            min_lr=1e-3,

            # Regularization (adjusted for feature-wise regularization)
            l1_lambda=0.15,  # Slightly reduced for faster convergence
            smoothness_lambda=0.1,  # Balanced temporal smoothing
            efficiency_lambda=0.05,  # Reduced magnitude control
            weight_decay=1e-3,  # Increased for better regularization with AdamW
            activation_shrink=0.05,
            regularization_mode='feature',  # 'feature': Apply regularization per-feature across time
            #'element'  Original per-time-step mode

        # SHAP sampling (optimized)
            paired_sampling=True,  # Keep for variance reduction
            samples_per_feature=10,  # Minimum effective samples

            # Architecture (lightweight but effective)
            # completely related to samples_per_feature if use higher num_attention_heads need to
            # use more samples_per_feature
            num_attention_heads=4,
            num_conv_layers=4,  # Maintain temporal processing
            num_filters=32,  # Reduced from 64 (faster forward passes)
            kernel_size=3,  # Kept for temporal context
            dropout_rate=0.1,  # Slightly reduced for faster learning

            # Optimizer (updated)
            optimizer_type="adam",  # 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'nadam'
            learning_rate=1e-3  # Slightly increased for faster convergence
        ).initialize(
            X_train=mydata.X[:mydata.X_train.shape[0] + mydata.X_val.shape[0], :, :],
            model_predict_func=model_predict,
            feature_names=mydata.feature_names
        )

        # Save to disk
        explainer.save(f"./Results/FastSHAP/{dataset_type}/", filename=f"{model_type}")


# fastshap_dir = f"./Results/FastSHAP/{mydata.data_type}/"
# loaded_explainer = main.TimeSeriesSHAPExplainerUpdated.load(fastshap_dir, filename="GRU")
# loaded_explainer.l1_lambda
# loaded_explainer.smoothness_lambda
# loaded_explainer.efficiency_lambda
# loaded_explainer.activation_shrink
