# region 1: Loading Libraries
import numpy as np
import pandas as pd

from Functions import main
import sqlite3
# endregion
import ast

# region2: Predictions
# mydata = main.load_and_preprocess_data(dataset_type=dataset_types[0])
# # mydata.full_data['energy_consumption'].describe()
# # mydata.feature_names
# # mydata.full_data.head()
#
# data = (mydata.X_train, mydata.y_train, mydata.X_val, mydata.y_val, mydata.X_test, mydata.y_test)
#
# for model_type in model_types:
#     main.optimize_and_save_model(data=data, df_name=mydata.data_type, n_trials=5, epochs=100, verbosity=0, model_type=model_type, more_info=mydata.more_info)
#     main.load_and_analyze_model(data=data, df_name=mydata.data_type, model_type=model_type, more_info=mydata.more_info)
#     main.optimize_and_save_rf_model(data=data, df_name=mydata.data_type, n_trials=10, verbosity=0, model_type=model_type, more_info=mydata.more_info)
#     main.load_and_plot_models(mydata=mydata, model_type=model_type)
#endregion

# region 3: find_minimal_error_points_with_dynamic_threshold
# # Prepare a list to collect results
# results = []
# # Iterate over dataset types and model types
# for dataset_type in dataset_types:
#     for model_type in model_types:
#         mydata = main.load_and_preprocess_data(dataset_type=dataset_type)
#         best_threshold, sorted_best_indices = main.find_minimal_error_points_with_local_threshold(mydata, model_type,
#                                                                                                   min_data_points=5, neighbors=2)
#         results.append({
#             "dataset_name": dataset_type,
#             "model_type": model_type,
#             "error_threshold": np.round(best_threshold,3),
#             "data_index": str(sorted_best_indices)  # Save list of indices as a string
#         })
#
# # Convert results to a pandas DataFrame
# results_df = pd.DataFrame(results)
#
# # Connect to the SQLite database (create it if it doesn't exist)
# db_path = "./Results/result.db"
# conn = sqlite3.connect(db_path)
#
# # Save the DataFrame to a new table named "threshold_selection_results"
# table_name = "index_selected"
# results_df.to_sql(table_name, conn, if_exists="replace", index=False)
#
# # Close the connection
# conn.close()
# # endregion

#region 4: Load indexs
# Connect to your SQLite database
conn = sqlite3.connect('./Results/result.db')

# Read the whole table into a pandas DataFrame
index_selected = pd.read_sql_query("SELECT * FROM index_selected;", conn)

# Close the connection
conn.close()
#endregion

#region4: XAI
xai_methods = ['Gradient']#, 'Tree', 'Kernel', 'Permutation', 'Sampling', 'Partition', 'Lime']
# xai_methods = ['TDE', 'Kernel', 'Tree', 'Partition', 'Lime', 'Permutation', 'Sampling']
# xai_methods = ['Sampling']
# model_type = model_types[0]
dataset_types = ["Residential", "Manufacturing facility", "Office building", "Retail store", "Medical clinic"]
model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']

dataset_type = dataset_types[0]

# for dataset_type in dataset_types:
mydata = main.load_and_preprocess_data(dataset_type=dataset_type)
model_type = model_types[9]
# for model_type in model_types:
minimal_error_points = index_selected.loc[
(index_selected['dataset_name'] == dataset_type) &
(index_selected['model_type'] == model_type),'data_index'
].apply(ast.literal_eval).tolist()[0]

for minimal_error_point in minimal_error_points:
    for xai_method in xai_methods:
        for data_type in ['original', 'random']:
            main.myshap(
                mydata,
                xai_method=xai_method,
                model_type=model_type,
                row_id=minimal_error_points[0],
                data_type=data_type,
                noise_factor=0.01,
                background_training=None,
                background_size=10,
                background_type='random',
                horizon=0,
                replace_sql=True)

#endregion

# for dataset_type in dataset_types:
mydata = main.load_and_preprocess_data(dataset_type=dataset_type)
for xai_method in xai_methods:
    # for model_type in model_types:
    for data_type in ['original', 'random']:
        main.generate_shap_heatmap_all_rows(mydata=mydata, xai_method=xai_method, model_type=model_type, data_type=data_type)

