from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import optuna
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import optuna
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


import optuna
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def create_lstm_model(trial, input_shape, output_shape):
    model = Sequential()
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    
    model.add(Input(shape=input_shape))
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_sequences, kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg)))

    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_model_with_params(params, input_shape, output_shape, create_model_func):
    trial = optuna.trial.FixedTrial(params)
    model = create_model_func(trial, input_shape, output_shape)
    return model

def objective(trial, create_model_func, X_train, y_train, X_val, y_val, epochs, verbosity):
    model = create_model_func(trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=verbosity, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=verbosity)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=verbosity, callbacks=[early_stopping, reduce_lr])
    val_loss = model.evaluate(X_val, y_val, verbose=verbosity)
    
    return val_loss

def run_optimization(data, df_name, file_name, n_trials, create_model_func, epochs=100, verbosity=1):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, create_model_func, X_train, y_train, X_val, y_val, epochs, verbosity), n_trials=n_trials)
    
    best_trial = study.best_trial
    
    best_model = create_model_func(best_trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=verbosity, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=verbosity)
    
    history = best_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=verbosity, callbacks=[early_stopping, reduce_lr])
    
    y_pred = best_model.predict(X_test)

    # Select first horizon
    y_test = y_test[:, 0].flatten()
    y_pred = y_pred[:, 0].flatten()
    
    # Calculate and round metrics
    mse_score = round(mean_squared_error(y_test, y_pred), 2)
    rmse_score = round(np.sqrt(mse_score), 2)
    mae_score = round(mean_absolute_error(y_test, y_pred), 2)
    smape_score = round(100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), 2)
    mape_score = round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)

    print(f'MSE: {mse_score}')
    print(f'RMSE: {rmse_score}')
    print(f'MAE: {mae_score}')
    print(f'MAPE: {mape_score}%')
    print(f'SMAPE: {smape_score}%')

    # Plot the first 10% of the data
    plot_predictions(y_test, y_pred, percentage=10)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Convert train and validation loss histories to strings
    train_loss_history = ','.join(map(str, history.history['loss']))
    val_loss_history = ','.join(map(str, history.history['val_loss']))

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        "dataset_name": [df_name],
        "best_params": [str(best_trial.params)],
        "duration": [best_trial.duration.total_seconds()],
        "mse_score": [mse_score],
        "rmse_score": [rmse_score],
        "mae_score": [mae_score],
        "smape_score": [smape_score],
        "mape_score": [mape_score],
        "train_loss_history": [train_loss_history],
        "val_loss_history": [val_loss_history]
    })
    
    # Append the results to the CSV file
    try:
        existing_df = pd.read_csv(file_name)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        combined_df = results_df
    
    combined_df.to_csv(file_name, index=False)
    
    return best_trial





























def replace_rare_categories(df, columns, threshold=0.01):
    if columns is None:
        return df
    for col in columns:
        value_counts = df[col].value_counts(normalize=True)
        categories_to_replace = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: f'{col}_other' if x in categories_to_replace else x)
    return df

def one_hot_encode(df, categorical_columns): 
    if categorical_columns is None:
        return df
    encoder = OneHotEncoder(drop=None, sparse_output=False)
    categorical_features = df[categorical_columns]
    encoded_features = encoder.fit_transform(categorical_features)
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
    for col in categorical_columns:
        col_prefix = f"{col}_"
        other_columns = [c for c in encoded_features_df.columns if c.startswith(col_prefix) and 'other' in c.lower()]
        if other_columns:
            encoded_features_df = encoded_features_df.drop(columns=other_columns[0])

    df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_features_df], axis=1)
    return df_encoded

def fill_missing_values(df, numerical_columns, categorical_columns=None, binary_columns=None):
    df_filled = df.copy()
    
    if numerical_columns is not None:
        for col in numerical_columns:
            for i in df.index:
                if pd.isna(df_filled.at[i, col]):
                    neighbors = []
                    if i > 1:
                        neighbors.append(df_filled.at[i-2, col])
                    if i > 0:
                        neighbors.append(df_filled.at[i-1, col])
                    if i < len(df)-1:
                        neighbors.append(df_filled.at[i+1, col])
                    if i < len(df)-2:
                        neighbors.append(df_filled.at[i+2, col])
                    neighbors = [n for n in neighbors if not pd.isna(n)]
                    if neighbors:
                        df_filled.at[i, col] = np.mean(neighbors)
    
    for col in (categorical_columns or []) + (binary_columns or []):
        for i in df.index:
            if pd.isna(df_filled.at[i, col]):
                neighbors = []
                if i > 1 and not pd.isna(df_filled.at[i-2, col]):
                    neighbors.append(df_filled.at[i-2, col])
                if i > 0 and not pd.isna(df_filled.at[i-1, col]):
                    neighbors.append(df_filled.at[i-1, col])
                if i < len(df)-1 and not pd.isna(df_filled.at[i+1, col]):
                    neighbors.append(df_filled.at[i+1, col])
                if i < len(df)-2 and not pd.isna(df_filled.at[i+2, col]):
                    neighbors.append(df_filled.at[i+2, col])
                if neighbors:
                    df_filled.at[i, col] = pd.Series(neighbors).mode()[0]
    
    return df_filled

def remove_leading_trailing_zeros(df, column_name):
    while len(df) > 0 and df[column_name].iloc[0] == 0:
        df = df.iloc[1:]
    while len(df) > 0 and df[column_name].iloc[-1] == 0:
        df = df.iloc[:-1]
    return df

def replace_middle_zeros(df, column_name):
    values = df[column_name].values
    for i in range(1, len(values) - 1):
        if values[i] == 0:
            prev_non_zero = next((values[j] for j in range(i-1, -1, -1) if values[j] != 0), None)
            next_non_zero = next((values[j] for j in range(i+1, len(values)) if values[j] != 0), None)
            if prev_non_zero is not None and next_non_zero is not None:
                values[i] = (prev_non_zero + next_non_zero) / 2
    df[column_name] = values
    return df

def process_dataframe(df, numerical_columns, categorical_columns=None, binary_columns=None, datetime_column=None, target_column=None):
    if datetime_column:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.sort_values(by=datetime_column)

    df = replace_rare_categories(df, categorical_columns)
    df = one_hot_encode(df, categorical_columns)
    df = fill_missing_values(df, numerical_columns, categorical_columns, binary_columns)
    
    if target_column:
        df = remove_leading_trailing_zeros(df, target_column)
        df = replace_middle_zeros(df, target_column)
    
    df = df.sort_values(by=datetime_column)
    
    return df

# Example usage:
# df = pd.read_csv('your_data.csv')
# numerical_columns = ['num_col1', 'num_col2']
# categorical_columns = ['cat_col1', 'cat_col2']
# binary_columns = ['bin_col1']
# datetime_column = 'date_col'
# target_column = 'target_col'
# df_processed = process_dataframe(df, numerical_columns, categorical_columns, binary_columns, datetime_column, target_column)
# print(df_processed)


def add_cos_transformations(df, col, period):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    df.drop(columns=[col], inplace=True)
    return df

def preprocess_energy_data(df, datetime_column):
    # Ensure Datetime is in datetime format
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Extracting date-time features
    df['Hour'] = df[datetime_column].dt.hour
    df['DayOfWeek'] = df[datetime_column].dt.dayofweek
    df['DayOfMonth'] = df[datetime_column].dt.day
    df['Month'] = df[datetime_column].dt.month
    df['DayOfYear'] = df[datetime_column].dt.dayofyear

    # Binary representation for weekends
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Apply cosine transformations for periodic features
    df = add_cos_transformations(df, 'Hour', 24)
    df = add_cos_transformations(df, 'DayOfWeek', 7)
    df = add_cos_transformations(df, 'DayOfMonth', 31)  # Approximate number of days in a month
    df = add_cos_transformations(df, 'Month', 12)
    df = add_cos_transformations(df, 'DayOfYear', 365)

    # Dropping Datetime column if not needed anymore
    df.set_index(datetime_column, inplace=True)

    return df

# Example usage:
# Assuming 'energy' is your DataFrame and target column index is the position of the target in the DataFrame
# energy = pd.read_csv('your_data.csv')
# datetime_column = 'Datetime'  # Replace with the actual name of your datetime column
# target_column = energy.columns.get_loc('TargetColumnName')  # Replace 'TargetColumnName' with your actual target column name
# model, scaler = train_lstm_model(energy, datetime_column, target_column)

# Function to seq the data
def seq_data(df, input_seq_length=48, output_seq_length=24, target='Energy'):
    X, y = [], []
    max_idx = len(df) - input_seq_length - output_seq_length + 1
    for i in range(max_idx):
        X.append(df.iloc[i:i+input_seq_length].values)
        y.append(df.iloc[i+input_seq_length:i+input_seq_length+output_seq_length][target].values)
    return np.array(X), np.array(y).reshape(max_idx, output_seq_length, 1)


def load_and_preprocess_data_with_sequences(df, target='Energy', val_ratio=0.1, test_ratio=0.1, input_seq_length=48, output_seq_length=24):
    # Drop NA values or consider imputation
    df.dropna(inplace=True)
    
    # Calculate test size and validation size
    test_size = int(len(df) * test_ratio)
    val_size = int(len(df) * val_ratio)
    
    # Check for edge cases
    if test_size == 0 or val_size == 0:
        raise ValueError("Dataset is too small to be split into train, validation, and test sets with the given ratios.")
    
    # Split the data manually to preserve the time series sequence
    train_df = df.iloc[:-(test_size + val_size), :]
    val_df = df.iloc[-(test_size + val_size):-test_size, :]
    test_df = df.iloc[-test_size:, :]
    
    # Initialize the scaler and fit it on the training data
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    
    # Scale the data
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # Convert scaled data back to DataFrame for easier manipulation
    train_scaled = pd.DataFrame(train_scaled, columns=train_df.columns)
    val_scaled = pd.DataFrame(val_scaled, columns=val_df.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns)
    

    # seq the data
    X_train, y_train = seq_data(train_scaled, input_seq_length, output_seq_length, target)
    X_val, y_val = seq_data(val_scaled, input_seq_length, output_seq_length, target)
    X_test, y_test = seq_data(test_scaled, input_seq_length, output_seq_length, target)
    
    # Combine all data for overall analysis
    all_scaled = scaler.transform(df)
    all_scaled = pd.DataFrame(all_scaled, columns=df.columns)
    X, y = seq_data(all_scaled, input_seq_length, output_seq_length, target)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, all_scaled


# Function to plot the first portion of the data
def plot_predictions(y_test, y_pred, percentage=10):
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # Calculate SMAPE
    smape = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))) * 100
    length = len(y_test)
    end = int(length * percentage / 100)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:end], label='Actual')
    plt.plot(y_pred[:end], label='Predicted')
    plt.title(f'Actual vs Predicted Values (MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%)')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    
    
def history_loss(history):
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()