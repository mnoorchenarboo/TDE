from scipy.special import comb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def shapley_sampler(num_features, min_inclusion_count, balanced=True):
    """
    Generalized Shapley sampler that ensures balanced inclusion of features.

    Args:
        num_features (int): Total number of features (d).
        min_inclusion_count (int): Minimum number of times each feature must be included.
        balanced (bool): Whether to balance the inclusion counts for all features.

    Returns:
        np.ndarray: Binary masks of shape (total_samples, num_features).
    """
    def compute_weights(num_features):
        """Compute weights for subset sizes based on Shapley distribution."""
        d = num_features
        weights = []
        for k in range(1, d):  # Subset sizes from 1 to d-1
            weight = (d - 1) / (comb(d, k) * k * (d - k))
            weights.append(weight)
        weights = np.array(weights)
        return weights / weights.sum()  # Normalize to form a probability distribution

    def generate_mask(num_features, size, required_feature=None):
        """Generate a binary mask of the given size, optionally including a required feature."""
        mask = np.zeros(num_features)
        if required_feature is not None:
            # Ensure the required feature is included
            included_indices = np.random.choice(num_features - 1, size=size - 1, replace=False)
            included_indices = np.insert(included_indices, 0, required_feature)
        else:
            included_indices = np.random.choice(num_features, size=size, replace=False)
        mask[included_indices] = 1
        return mask

    weights = compute_weights(num_features)  # Compute weights for subset sizes
    subset_sizes = np.arange(1, num_features)  # Possible subset sizes

    masks = []

    if balanced:
        # Ensure each feature is included at least `min_inclusion_count` times
        for feature in range(num_features):
            for _ in range(min_inclusion_count):
                size = np.random.choice(subset_sizes, p=weights)  # Sample subset size
                size = max(1, size)  # Ensure at least one feature is included
                mask = generate_mask(num_features, size, required_feature=feature)
                masks.append(mask)

        # Shuffle to randomize the order of the balanced masks
        np.random.shuffle(masks)

    else:
        # Generate masks without balancing
        for _ in range(min_inclusion_count * num_features):  # Generate approximately equal total samples
            size = np.random.choice(subset_sizes, p=weights)  # Sample subset size
            mask = generate_mask(num_features, size)
            masks.append(mask)

    return np.array(masks)

def cluster_data(dataset, n_clusters):
    """
    Perform clustering on a dataset and calculate cluster centers.

    Args:
        dataset (pd.DataFrame): The dataset with both categorical and continuous features.
        n_clusters (int): Number of clusters.

    Returns:
        clusters (np.ndarray): Cluster assignments for each row in the dataset.
        cluster_centers (dict): Cluster centers with averages for continuous variables
                                and modes for categorical variables.
    """
    # Separate categorical and continuous variables
    continuous_cols = dataset.select_dtypes(include=[np.number]).columns
    categorical_cols = dataset.select_dtypes(exclude=[np.number]).columns

    # Encode categorical variables into integers for clustering
    dataset_encoded = dataset.copy()
    if not categorical_cols.empty:
        for col in categorical_cols:
            dataset_encoded[col] = dataset_encoded[col].astype('category').cat.codes

    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(dataset_encoded)

    # Calculate cluster centers
    cluster_centers = {}
    for cluster_id in range(n_clusters):
        cluster_data = dataset[clusters == cluster_id]  # Get data for the current cluster

        # Calculate averages for continuous columns
        center_continuous = cluster_data[continuous_cols].mean().astype(float).to_dict() if not continuous_cols.empty\
            else {}

        # Calculate modes for categorical columns
        center_categorical = cluster_data[categorical_cols].apply(
            lambda x: x.mode()[0]) if not categorical_cols.empty else {}

        # Combine continuous and categorical centers
        cluster_centers[cluster_id] = {**center_continuous, **center_categorical}

    return clusters, cluster_centers


def impute_from_other_clusters(x, mask, cluster_centers, current_cluster):
    x_imputed = x.copy()
    excluded_features = np.where(mask == 0)[0]

    for feature_idx in excluded_features:
        feature_name = x.index[feature_idx]
        other_clusters = [c for c in cluster_centers.keys() if c != current_cluster]
        random_cluster = np.random.choice(other_clusters)

        # Cast value to original column type
        imputed_value = cluster_centers[random_cluster][feature_name]
        x_imputed[feature_name] = x[feature_name].dtype.type(imputed_value)

    return x_imputed


def calculate_exact_shapley_values(dataset, model, sampler, cluster_centers, clusters):
    num_features = dataset.shape[1]
    shapley_values = np.zeros((dataset.shape[0], num_features))  # Initialize Shapley values

    for i, x in dataset.iterrows():
        current_cluster = clusters[i]  # Get cluster assignment for the current data point
        sampled_masks = sampler(num_features, min_inclusion_count=100, balanced=True)
        shapley_contributions = np.zeros(num_features)

        for mask in sampled_masks:
            # Impute missing values using other clusters' centers
            x_imputed = impute_from_other_clusters(x, mask, cluster_centers, current_cluster)

            # Compute model predictions
            prediction_with_mask = model.predict(np.array(x_imputed).reshape(1, -1))
            prediction_without_mask = model.predict(np.array(x).reshape(1, -1))

            # Marginal contribution
            marginal_contributions = prediction_with_mask - prediction_without_mask

            # Accumulate contributions
            shapley_contributions += marginal_contributions.squeeze() * mask

        # Average over all sampled subsets
        shapley_values[i] = shapley_contributions / len(sampled_masks)

    # Ensure numerical consistency
    shapley_values = np.array(shapley_values, dtype=np.float64)
    return shapley_values

def train_shapley_neural_network(X, shapley_values):
    """
    Train a neural network to predict Shapley values.

    Args:
        X (np.ndarray): Input data.
        shapley_values (np.ndarray): Exact Shapley values as the target.

    Returns:
        keras.Model: Trained neural network model.
    """
    input_dim = X.shape[1]

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(shapley_values.shape[1])  # Output layer has one neuron per feature
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, shapley_values, epochs=100, batch_size=32, verbose=1)

    return model


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

# Step 1: Create Example Dataset
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(20, 60, 50),
    'Salary': np.random.randint(30000, 100000, 50),
    'Experience': np.random.randint(1, 40, 50),
    'City': np.random.choice(['A', 'B', 'C'], 50),
    'Target': np.random.randint(20000, 150000, 50)
})

# Encode categorical variables
data['City'] = data['City'].astype('category').cat.codes

# Split into features and target
X = data.drop(columns=['Target'])
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train Random Forest Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 3: Exact Shapley Values using SHAP's ExactExplainer
exact_explainer = shap.ExactExplainer(model.predict, X_train)  # Train on X_train

# Compute exact Shapley values for a single row (e.g., first row of X_test)
row_index = 0
row_values = X_test.iloc[row_index].values  # Convert to array for ExactExplainer

model.predict(X_test.iloc[[row_index]])

exact_values = exact_explainer.explain_row(
    row_values,
    max_evals=2**X_train.shape[1],
    main_effects=False,
    error_bounds=False,
    batch_size=10,
    outputs=1,
    interactions=False,
    silent=True
)

# Extract feature Shapley values
print(f"Exact Shapley Values for Row {row_index}:\n{exact_values}")

# Step 4: TreeExplainer Shapley Values
tree_explainer = shap.TreeExplainer(model)
tree_shap_values = tree_explainer.shap_values(X_test.iloc[[row_index]])[0]
print(f"TreeExplainer Shapley Values for Row {row_index}:\n{tree_shap_values}")

# Step 5: Custom Shapley Value Calculation
# Assuming you have implemented calculate_custom_shapley_values
from sklearn.cluster import KMeans

# Cluster data for custom method
n_clusters = 3
clusters, cluster_centers = cluster_data(X_train, n_clusters)

custom_shapley_values = calculate_exact_shapley_values(X_test, model, shapley_sampler, cluster_centers, clusters)
print(f"Custom Shapley Values for Row {row_index}:\n{custom_shapley_values[row_index]}")

# Step 6: Compare All Results
differences_exact_vs_tree = np.abs(exact_shapley_values - tree_shap_values)
differences_exact_vs_custom = np.abs(exact_shapley_values - custom_shapley_values[row_index])

print(f"Differences (Exact vs. TreeExplainer) for Row {row_index}:\n{differences_exact_vs_tree}")
print(f"Differences (Exact vs. Custom) for Row {row_index}:\n{differences_exact_vs_custom}")
