import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.backends.backend_pdf import PdfPages

def add_feature(df):
    # df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Dayofweek'] = df.index.dayofweek
    df['Hour'] = df.index.hour

    df['Weekend'] = df['Dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Daylight'] = df['Hour'].apply(lambda x: 1 if 7 <= x <= 19 else 0)

    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/7)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/7)
    df.drop('Month', axis=1, inplace=True)  # Drop the original hour column

    df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
    df.drop('Day', axis=1, inplace=True)  # Drop the original hour column

    df['Dayofweek_sin'] = np.sin(2 * np.pi * df['Dayofweek']/7)
    df['Dayofweek_cos'] = np.cos(2 * np.pi * df['Dayofweek']/7)
    df.drop('Dayofweek', axis=1, inplace=True)  # Drop the original hour column

    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/7)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/7)
    df.drop('Hour', axis=1, inplace=True)  # Drop the original hour column

    return df

def Alexandra(id_number=1, folder_path='./Data/Alexandra/'):

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter the files to only include CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Define the data
    df = pd.read_csv(folder_path + csv_files[id_number])
    # Let's convert 'DateTime' column to datetime type if it's not already
    df.rename(columns={'energy': 'VAL'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['DateTime'] += pd.to_timedelta(df['hour'], unit='h')
    df= df[['DateTime','VAL']]

    df.index = df['DateTime']
    df.drop(columns='DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = add_feature(df)
    # print(csv_files[id_number])
    return df, folder_path + csv_files[id_number], len(csv_files)

def Muhammad(id_number=1, folder_path= './Data/Muhammad/'):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter the files to only include CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Define the data
    df = pd.read_csv(folder_path + csv_files[id_number])

    # Let's convert 'DateTime' column to datetime type if it's not already
    df.rename(columns={'PJME_MW': 'VAL', 'Datetime':'DateTime'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df= df[['DateTime','VAL']]

    df.index = df['DateTime']
    df.drop(columns='DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df = add_feature(df)

    return df, folder_path + csv_files[id_number], len(csv_files)

def Kaggle1(id_number=1, folder_path= './Data/Kaggle1/'):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter the files to only include CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Define the data
    df = pd.read_csv(folder_path + csv_files[id_number])

    # Let's convert 'DateTime' column to datetime type if it's not already
    df.rename(columns={'Datetime':'DateTime'}, inplace=True)
    df.rename(columns={df.columns[1]: 'VAL'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df= df[['DateTime','VAL']]

    df.index = df['DateTime']
    df.drop(columns='DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df = add_feature(df)

    return df, folder_path + csv_files[id_number], len(csv_files)

def load_and_preprocess_data_with_sequences(df, val_ratio=0.1, test_ratio=0.10, input_seq_length=48, output_seq_length=24):
    # Drop NA values or consider imputation
    df.dropna(inplace=True)

    scaler = MinMaxScaler()

    target = df['VAL']
    features = df
    save_features = df

    # Initialize the scaler for the features
    feature_scaler = MinMaxScaler()

    features['VAL'] = feature_scaler.fit_transform(features['VAL'].values.reshape(-1, 1))

    # Initialize a separate scaler for the target since it has a different number of features
    target_scaler = MinMaxScaler()

    # Scale the target
    target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # Convert the scaled target back to a 1D array
    target = target.flatten()

    features = pd.DataFrame(features, columns=save_features.columns)
    target = pd.DataFrame(target, columns=['VAL'])

    # Calculate test size and validation size
    test_size = int(len(df) * test_ratio)
    val_size = int(len(df) * val_ratio)

    # Split the data manually to preserve the time series sequence
    X_train = features.iloc[:-(test_size + val_size), :]
    y_train = target.iloc[:-(test_size + val_size), :]

    X_val = features.iloc[-(test_size + val_size):-test_size, :]
    y_val = target.iloc[-(test_size + val_size):-test_size, :]

    X_test = features.iloc[-test_size:, :]
    y_test = target.iloc[-test_size:, :]

    # Function to prepare the data
    def prepare_data(features, targets, input_seq_length, output_seq_length):
        X, y = [], []
        for i in range(len(features) - input_seq_length - output_seq_length):
            X.append(features[i:i+input_seq_length])
            y.append(targets[i+input_seq_length:i+input_seq_length+output_seq_length])
        return np.array(X), np.array(y)

    # Prepare the data
    X_train, y_train = prepare_data(X_train.values, y_train.values, input_seq_length, output_seq_length)
    X_val, y_val = prepare_data(X_val.values, y_val.values, input_seq_length, output_seq_length)
    X_test, y_test = prepare_data(X_test.values, y_test.values, input_seq_length, output_seq_length)

    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_or_save_loss(train_loss,val_loss, save_as_pdf=False, filename="loss_plot.pdf"):
    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    
    if save_as_pdf:
        with PdfPages(filename) as pdf:
            pdf.savefig()
            plt.close()
        print(f"Loss plot saved as {filename}")
    else:
        plt.show()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Create a mask to identify non-zero values in y_true
    if np.any(mask):  # Check if there are any non-zero values
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    else:
        return 0  # Return 0 if all y_true values are zero, which prevents division by zero

def plot_predictions(model, X_test, y_test, i=0, j=100, save_to_pdf=False, pdf_filename="predictions.pdf", real_transparency=0.5, pred_transparency=0.5):
    # Predict the values
    y_pred = model.predict(X_test)

    # Extract the specific feature's predictions and actual values
    y_pred_temp = y_pred[:, i]
    y_test_temp = y_test[:, i, 0]

    # Calculate MAPE for the entire series
    mape = mean_absolute_percentage_error(y_test_temp, y_pred_temp)

    # Define the time index for plotting
    time_index = range(len(y_test_temp))

    # Create a PDF file if required
    if save_to_pdf:
        pdf = PdfPages(pdf_filename)

    # Plot y_test and y_pred for the entire series
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, y_test_temp, label='Actual', color=(0, 0, 1, real_transparency))  # Blue with specified transparency
    plt.plot(time_index, y_pred_temp, label='Predicted', color=(1, 0, 0, pred_transparency))  # Red with specified transparency
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted - MAPE: {mape:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('white')  # Set axes background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white
    plt.gca().patch.set_facecolor('white')  # Remove axes patch background
    if save_to_pdf:
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()
    else:
        plt.show()

    # Slice the first j elements for visualization
    y_pred_100 = y_pred_temp[:j]
    y_test_flat_100 = y_test_temp[:j]

    # Calculate MAPE for the first j steps
    mape_100 = mean_absolute_percentage_error(y_test_flat_100, y_pred_100)

    # Plotting the results for the first j steps
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_flat_100, label='Actual Values', marker='o', color=(0, 0, 1, real_transparency))  # Blue with specified transparency
    plt.plot(y_pred_100, label='Predicted Values', marker='x', color=(1, 0, 0, pred_transparency))  # Red with specified transparency
    plt.title(f'Comparison of Actual and Predicted Values for the First {j} Time Steps')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('white')  # Set axes background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white
    plt.gca().patch.set_facecolor('white')  # Remove axes patch background
    if save_to_pdf:
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()
        pdf.close()  # Close the PDF file
    else:
        plt.show()

# Example of calling the function with different transparency levels
# plot_predictions(model, X_test, y_test, i=0, j=100, save_to_pdf=False, pdf_filename="predictions.pdf", real_transparency=0.7, pred_transparency=0.5)




import ast
import optuna
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def create_lstm_model(trial, input_shape, output_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    lstm_units = trial.suggest_int('lstm_units', 20, 100)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])

    model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_shape))

    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_model_with_params(params, input_shape, output_shape, create_model_func):
    trial = optuna.trial.FixedTrial(params)
    model = create_model_func(trial, input_shape, output_shape)
    return model

def objective(trial, create_model_func, X_train, y_train, X_val, y_val):
    model = create_model_func(trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    
    return val_loss

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def run_optimization(data, df_name, file_name, n_trials, create_model_func):
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, create_model_func, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    best_trial = study.best_trial
    
    best_model = create_model_func(best_trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    history = best_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
    
    y_pred = best_model.predict(X_test)

    # Ensure y_test and y_pred are 2D arrays
    y_test = np.squeeze(y_test)
    y_pred = np.squeeze(y_pred)

    mse_score = mean_squared_error(y_test, y_pred)
    rmse_score = np.sqrt(mse_score)
    mae_score = mean_absolute_error(y_test, y_pred)
    mape_score = mean_absolute_percentage_error(y_test, y_pred)
    smape_score = smape(y_test, y_pred)

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
        "mape_score": [mape_score],
        "smape_score": [smape_score],
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

# Function to create and train the final model using the best hyperparameters
def train_with_best_hyperparameters(best_params_str, data, create_model_func):
    best_params = ast.literal_eval(best_params_str)  # Convert string representation to dictionary
    X_train, X_val, X_test, y_train, y_val, y_test = data
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]

    # Create the model with the best hyperparameters
    model = create_model_with_params(best_params, input_shape, output_shape, create_model_func)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
    
    # Make predictions
    y_pred = model.predict(X_test)

    return model, y_pred

