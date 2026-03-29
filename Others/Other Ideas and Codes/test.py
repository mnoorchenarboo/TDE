def generate_multivariate_cluster_distributions(dataset, clusters, categorical_columns):
    """
    Generate multivariate distributions for each cluster.

    Args:
        dataset (pd.DataFrame): Original dataset.
        clusters (np.ndarray): Cluster assignments for each row.
        categorical_columns (list): List of categorical column names.

    Returns:
        dict: A dictionary where keys are cluster IDs and values are dictionaries with:
              - 'continuous': Mean vector and covariance matrix for continuous variables.
              - 'categorical': Joint frequency distributions for categorical variables.
    """
    cluster_distributions = {}
    continuous_columns = dataset.columns.difference(categorical_columns)

    for cluster_id in np.unique(clusters):
        cluster_data = dataset[clusters == cluster_id]
        distributions = {}

        # Continuous variables: Compute mean vector and covariance matrix
        if not continuous_columns.empty:
            mean_vector = cluster_data[continuous_columns].mean().values
            covariance_matrix = cluster_data[continuous_columns].cov().values
            distributions['continuous'] = {
                'mean': mean_vector,
                'cov': covariance_matrix
            }

        # Categorical variables: Compute joint probability distribution
        if not categorical_columns.empty:
            joint_probs = cluster_data[categorical_columns].value_counts(normalize=True)
            distributions['categorical'] = joint_probs

        cluster_distributions[cluster_id] = distributions

    return cluster_distributions
