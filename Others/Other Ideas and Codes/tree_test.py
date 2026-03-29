import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap

# Step 1: Generate a regression dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train a RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Step 4: Use Kernel SHAP to calculate SHAP values for a small subset
def generate_kernel_shap_values(X_train, regressor, subset_size=100):
    """
    Compute Kernel SHAP values for a small subset of the training data.
    """
    kernel_explainer = shap.KernelExplainer(lambda x: regressor.predict(x), X_train[:subset_size])
    shap_values = []

    for i in range(subset_size):
        shap_values.append(kernel_explainer.shap_values(X_train[i]))

    shap_values = np.array(shap_values)
    return X_train[:subset_size], shap_values

# Generate SHAP values for a small subset
subset_size = 100
X_train_subset, shap_values_subset = generate_kernel_shap_values(X_train, regressor, subset_size=subset_size)

# Step 5: Define the FastSHAP Model
class FastSHAPModel(nn.Module):
    def __init__(self, n_features, hidden_size=32, n_hidden=2):
        super(FastSHAPModel, self).__init__()
        layers = []
        input_size = n_features
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, n_features))  # Output SHAP values for all features
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Step 6: Train the FastSHAP Model
def train_fastshap_model(X_train, shap_values, n_features, batch_size=32, epochs=10, lr=0.001):
    """
    Train the FastSHAP model using SHAP values from a small subset of training data.

    Args:
        X_train (np.array): Features of the subset.
        shap_values (np.array): SHAP values of the subset (ground truth).
        n_features (int): Number of features in the dataset.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        model: Trained FastSHAP model.
    """
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(shap_values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FastSHAPModel(n_features=n_features, hidden_size=32, n_hidden=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model

# Train the FastSHAP model
fastshap_model = train_fastshap_model(X_train_subset, shap_values_subset, n_features=X_train.shape[1])

# Step 7: Use FastSHAP for Inference
def fastshap_explain(model, instance):
    """
    Use the trained FastSHAP model to compute SHAP values for a single instance.

    Args:
        model: Trained FastSHAP model.
        instance (np.array): Instance to explain.

    Returns:
        shap_values (np.array): SHAP values for the given instance.
    """
    instance_tensor = torch.tensor(instance, dtype=torch.float32).unsqueeze(0)
    shap_values = model(instance_tensor).detach().numpy()
    return shap_values.flatten()

# Step 8: Compute SHAP values for all methods
test_instance = X_test[0]

# KernelExplainer SHAP
start_time = time.time()
kernel_explainer = shap.KernelExplainer(lambda x: regressor.predict(x), X_train[:subset_size])
kernel_shap_values = kernel_explainer.shap_values(test_instance)
kernel_baseline = kernel_explainer.expected_value
kernel_time = time.time() - start_time

# FastSHAP
start_time = time.time()
fastshap_shap_values = fastshap_explain(fastshap_model, test_instance)
fastshap_time = time.time() - start_time

# Step 9: Results Table
results = pd.DataFrame({
    "Feature": [f"Feature {i+1}" for i in range(X.shape[1])],
    "KernelExplainer SHAP": kernel_shap_values,
    "FastSHAP": fastshap_shap_values,
})

results.loc["Baseline"] = {
    "Feature": "Baseline",
    "KernelExplainer SHAP": kernel_baseline,
    "FastSHAP": kernel_baseline,  # FastSHAP uses KernelExplainer's baseline
}

results.loc["Computation Time (s)"] = {
    "Feature": "Computation Time (s)",
    "KernelExplainer SHAP": kernel_time,
    "FastSHAP": fastshap_time,
}

results