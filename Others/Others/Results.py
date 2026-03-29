import os
from scipy.stats import ttest_rel
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.backends.backend_pdf import PdfPages
os.getcwd()

import sqlite3
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import ttest_rel


def compare_shap_values_all_rows(dataset_name, xai_methods, model_type, database_path='./Results/result.db'):
    """
    Compares absolute differences of SHAP values between 'original' and 'random' data types
    for all row_ids and performs a single paired t-test per XAI method.

    Parameters:
    - dataset_name: The name of the dataset to evaluate.
    - xai_methods: List of XAI methods to evaluate.
    - model_type: The model type to evaluate.
    - database_path: Path to the SQLite database (default: './Results/result.db').

    Returns:
    - Dictionary with XAI methods as keys and p-values as values.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    results = {}

    for xai_method in xai_methods:
        cursor.execute('''
            SELECT row_id, data_type, shap_df FROM XAI
            WHERE dataset_name = ? AND xai_method = ? AND model_type = ? 
            AND data_type IN ('original', 'random')
        ''', (dataset_name, xai_method, model_type))

        data = cursor.fetchall()

        if not data:
            print(f"No data found for {xai_method}. Skipping.")
            continue

        # Organizing SHAP values for each row_id
        shap_values = {}
        for row_id, data_type, shap_json in data:
            if row_id not in shap_values:
                shap_values[row_id] = {}
            shap_values[row_id][data_type] = pd.read_json(StringIO(shap_json)).values.flatten()

        # Collecting absolute differences across all row_ids
        all_absolute_differences = []
        for row_id, values in shap_values.items():
            if 'original' in values and 'random' in values:
                # abs_diff = np.abs(values['original'] - values['random'])
                abs_diff = values['original'] - values['random']
                all_absolute_differences.extend(abs_diff)

        if len(all_absolute_differences) == 0:
            print(f"No valid comparisons found for {xai_method}.")
            continue

        # Perform paired t-test on absolute differences
        t_stat, p_value = ttest_rel(all_absolute_differences, np.zeros(len(all_absolute_differences)))
        results[xai_method] = p_value

    conn.close()
    return results


# Example usage
xai_methods = ['TDE', 'Kernel', 'Permutation', 'Sampling', 'Partition', 'Lime', 'Tree']
dataset_types = ["Residential", "Manufacturing facility", "Office building", "Retail store", "Medical clinic"]
model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']

p_values = compare_shap_values_all_rows(dataset_name=dataset_types[1], xai_methods=xai_methods,
                                        model_type=model_types[7])

# Print results
for method, p_val in p_values.items():
    print(f"{method}: p_value = {p_val:.10f}")


dataset_types = ["Residential", "Manufacturing facility"]#, "Office building", "Retail store", "Medical clinic"]
model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']
xai_methods = ['Kernel', 'Permutation', 'Sampling', 'Partition', 'Lime', 'Tree']

results = compare_shap_values_by_row_id(dataset_name=dataset_types[1], xai_method=xai_methods[0],
                                        model_type=model_types[7])
for res in results:
    print(f"Row ID {res['row_id']}: t_statistic = {res['t_statistic']:.4f}, p_value = {res['p_value']:.4f}")

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.backends.backend_pdf import PdfPages


def plot_abs_residuals_lineplot(dataset_name, xai_methods, model_type,
                                output_file='shap_abs_residuals_lineplot.pdf',
                                database_path='./Results/result.db'):
    """
    Computes absolute residuals (|original - random|) for each row_id and each XAI method,
    aggregates them per feature, and then plots a single line plot where each line (with a unique color)
    corresponds to one XAI method.

    Parameters:
    - dataset_name: Name of the dataset.
    - xai_methods: List of XAI methods to process.
    - model_type: Model type.
    - output_file: Name of the output PDF file.
    - database_path: Path to the SQLite database.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Dictionary to store aggregated absolute residuals for each XAI method.
    residuals_dict = {}

    for xai_method in xai_methods:
        # Retrieve all distinct row_ids for this combination
        cursor.execute('''
            SELECT DISTINCT row_id FROM XAI 
            WHERE dataset_name = ? AND xai_method = ? AND model_type = ?
        ''', (dataset_name, xai_method, model_type))
        row_ids = [row[0] for row in cursor.fetchall()]
        if not row_ids:
            print(f"No row_ids found for {xai_method}.")
            continue

        aggregate = None  # Will accumulate absolute residuals per feature
        valid_count = 0  # Count of row_ids with valid data

        for row_id in row_ids:
            # Retrieve SHAP values for 'original'
            cursor.execute('''
                SELECT shap_df FROM XAI 
                WHERE dataset_name = ? AND xai_method = ? AND model_type = ? 
                  AND data_type = 'original' AND row_id = ?
            ''', (dataset_name, xai_method, model_type, row_id))
            orig_row = cursor.fetchone()

            # Retrieve SHAP values for 'random'
            cursor.execute('''
                SELECT shap_df FROM XAI 
                WHERE dataset_name = ? AND xai_method = ? AND model_type = ? 
                  AND data_type = 'random' AND row_id = ?
            ''', (dataset_name, xai_method, model_type, row_id))
            rand_row = cursor.fetchone()

            if not orig_row or not rand_row:
                continue

            # Convert JSON strings to DataFrames
            try:
                orig_df = pd.read_json(StringIO(orig_row[0]))
                rand_df = pd.read_json(StringIO(rand_row[0]))
            except Exception as e:
                print(f"Error parsing row_id {row_id} for {xai_method}: {e}")
                continue

            if orig_df.shape != rand_df.shape:
                print(f"Shape mismatch for row_id {row_id} in {xai_method}.")
                continue

            # Compute absolute residuals (|original - random|) for this row.
            abs_residual_row = np.abs(orig_df - rand_df).mean(axis=0)  # Average over time steps
            if aggregate is None:
                aggregate = abs_residual_row
            else:
                aggregate += abs_residual_row
            valid_count += 1

        if valid_count > 0:
            # Average the aggregated absolute residuals over the number of valid rows.
            aggregate = aggregate / valid_count
            residuals_dict[xai_method] = aggregate
        else:
            print(f"No valid data for {xai_method}.")

    conn.close()

    if not residuals_dict:
        print("No residual data computed for any XAI method.")
        return

    # Create a single line plot with one line per XAI method.
    plt.figure(figsize=(12, 6))
    # Get feature names from one of the aggregated Series (they should be consistent across methods)
    example_method = next(iter(residuals_dict))
    features = residuals_dict[example_method].index.tolist()
    x = np.arange(len(features))

    # Get a colormap using the correct Matplotlib API
    cmap = plt.cm.get_cmap("tab10")

    for i, (xai_method, residuals_series) in enumerate(residuals_dict.items()):
        plt.plot(x, residuals_series.values, marker='o', linestyle='-',
                 label=xai_method, color=cmap(i))

    plt.xlabel('Features')
    plt.ylabel('Average Absolute Residual |Original - Random|')
    plt.title(f'Absolute Residuals per Feature for {dataset_name} - {model_type}')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a single-page PDF.
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Line plot of absolute residuals saved to {output_file}")


plot_abs_residuals_lineplot(
    dataset_name=dataset_types[0],
    xai_methods=xai_methods,
    model_type=model_types[0],
    output_file="shap_residuals_lineplot_residential.pdf"
)
