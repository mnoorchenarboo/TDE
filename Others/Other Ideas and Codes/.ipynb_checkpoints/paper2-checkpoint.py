import pandas as pd
from Functions import BDG2, preprocess, my


# https://github.com/buds-lab/building-data-genome-project-2/tree/master/data/meters/cleaned
# Unique Primary Uses: ['services', 'health', 'education', 'lodging', 'industrial', 'unknown', 'office', 'retail', 'food', 'other', 'religion', 'utility', 'public', 'science', 'warehouse', 'parking', 'assembly']

df = pd.read_csv('./Data/building-data-genome-project-2/electricity_cleaned.txt')

target_column = 'energy_consumption'

# df_name = "industrial" #PrimaryUses Manufacturing corresponds to 'industrial'. option_number=1
# df_name = "health" #PrimaryUses Medical Clinic corresponds to 'health'. option_number=1
df_name = "retail" #PrimaryUses Retail Store corresponds to 'retail'. option_number=3
# df_name = "office" #PrimaryUses Office corresponds to 'office'. option_number=1
cleaned_df, more_info = BDG2.get_column_by_criteria(df, primary_use=df_name, option_number=2)

#scale type = both, features, outcome
X_train, y_train, X_val, y_val, X_test, y_test, X, y, df_scaled = preprocess.load_and_preprocess_data_with_sequences(cleaned_df, target=target_column, scaled=True, scale_type='features', val_ratio=0.1, test_ratio=0.1, input_seq_length=48, output_seq_length=24)

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


tcn_model, tcn_metrics, history_tcn = my.implement_tcn(X_train, y_train, X_val, y_val, X_test, y_test)

tcn_with_attention_model, tcn_with_attention_metrics, attention_scores, history_tcn_with_attention = my.implement_tcn_with_attention(
    X_train, y_train, X_val, y_val, X_test, y_test,
    num_filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8],
    dropout_rate=0.2, num_heads=4, epochs=20, batch_size=32)

print("TCN Metrics:", tcn_metrics)
print("TCN Metrics:", tcn_with_attention_metrics)



my.plot_loss_curves(history_tcn, history_tcn_with_attention)
