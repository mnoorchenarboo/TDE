import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import optuna
import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


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

def pre_process(df, numerical_columns, categorical_columns=None, binary_columns=None, datetime_column=None, target_column=None):
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


def add_cos_transformations(df, col, period):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    df.drop(columns=[col], inplace=True)
    return df

def get_datetime_features(df, datetime_column, cos_sin = False):
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
    if cos_sin:
        # Apply cosine transformations for periodic features
        df = add_cos_transformations(df, 'Hour', 24)
        df = add_cos_transformations(df, 'DayOfWeek', 7)
        df = add_cos_transformations(df, 'DayOfMonth', 31)  # Approximate number of days in a month
        df = add_cos_transformations(df, 'Month', 12)
        df = add_cos_transformations(df, 'DayOfYear', 365)

    # Dropping Datetime column if not needed anymore
    df.set_index(datetime_column, inplace=True)

    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate seasonal weather data with realistic seasonal changes
def seasonal_variation(day_of_year, min_value, max_value, noise_std=0):
    # Use a sine wave to simulate seasonality
    base_value = min_value + (max_value - min_value) * (np.sin(2 * np.pi * day_of_year / 365))
    noise = np.random.normal(loc=0, scale=noise_std, size=len(day_of_year))
    return base_value + noise

# Function to determine appropriate min/max values for temperature, humidity, and wind speed
def get_seasonal_ranges(day_of_year):
    temperature_min = -5 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    temperature_max = 5 + 20 * np.sin(2 * np.pi * day_of_year / 365)

    humidity_min = 30 + 20 * np.sin(2 * np.pi * day_of_year / 365)
    humidity_max = 60 + 20 * np.sin(2 * np.pi * day_of_year / 365)

    wind_speed_min = 0 + 5 * np.sin(2 * np.pi * day_of_year / 365)
    wind_speed_max = 5 + 5 * np.sin(2 * np.pi * day_of_year / 365)

    return (temperature_min, temperature_max), (humidity_min, humidity_max), (wind_speed_min, wind_speed_max)

# Function to generate daily energy consumption patterns
def daily_energy_pattern(hour, noise_std=0):
    morning_peak = (7 <= hour) & (hour < 9)
    daytime = (9 <= hour) & (hour < 17)
    evening_peak = (17 <= hour) & (hour < 21)
    night = ((21 <= hour) & (hour < 24)) | ((0 <= hour) & (hour < 7))

    base_energy_pattern = (
        morning_peak * np.random.uniform(1.1, 1.2, size=len(hour)) +
        daytime * np.random.uniform(1.0, 1.1, size=len(hour)) +
        evening_peak * np.random.uniform(1.2, 1.3, size=len(hour)) +
        night * np.random.uniform(0.8, 0.9, size=len(hour))
    )

    noise = np.random.normal(loc=0, scale=noise_std, size=len(hour))
    return base_energy_pattern + noise

# Function to generate synthetic weather and energy data
def generate_synthetic_data(start_date, end_date, freq='h', weather_noise_std=0.5, energy_noise_std=0.5, plot_percentage=0):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    num_samples = len(date_range)

    np.random.seed(0)

    day_of_year = date_range.dayofyear
    hour_of_day = date_range.hour
    weekday = date_range.weekday

    # Get the seasonal min/max ranges
    (temperature_min, temperature_max), (humidity_min, humidity_max), (wind_speed_min, wind_speed_max) = get_seasonal_ranges(day_of_year)

    # Generate weather data with seasonal variation
    temperature = seasonal_variation(day_of_year, temperature_min, temperature_max, noise_std=weather_noise_std)
    humidity = seasonal_variation(day_of_year, humidity_min, humidity_max, noise_std=weather_noise_std)
    wind_speed = seasonal_variation(day_of_year, wind_speed_min, wind_speed_max, noise_std=weather_noise_std)

    # Create base energy consumption model
    base_energy_consumption = 100 + temperature * 3.5 - humidity * 1.2 + wind_speed * 0.1

    # Apply daily pattern with noise
    daily_pattern = daily_energy_pattern(hour_of_day, noise_std=energy_noise_std)
    energy_consumption = base_energy_consumption * daily_pattern

    # Smoothing the energy consumption by making it partially dependent on the previous hour's value
    energy_consumption = energy_consumption.to_numpy()
    for i in range(1, num_samples):
        energy_consumption[i] = 0.7 * energy_consumption[i-1] + 0.3 * energy_consumption[i]

    data = pd.DataFrame({
        'timestamp': date_range,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'energy_consumption': energy_consumption
    })

    # Plotting
    if plot_percentage > 0:
        plot_samples = int(len(data) * (plot_percentage / 100))
        plot_data = data.head(plot_samples)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)

        plot_data.plot(x='timestamp', y='temperature', ax=axes[0], title='Temperature')
        plot_data.plot(x='timestamp', y='humidity', ax=axes[1], title='Humidity')
        plot_data.plot(x='timestamp', y='wind_speed', ax=axes[2], title='Wind Speed')
        plot_data.plot(x='timestamp', y='energy_consumption', ax=axes[3], title='Energy Consumption')

        plt.tight_layout()
        plt.show()

    # Correlation analysis
    correlation = data[['temperature', 'energy_consumption']].corr().iloc[0, 1]
    print(f"Correlation between temperature and energy consumption: {correlation:.2f}")

    return data

# Function to create sequences
def seq_data(df, input_seq_length=48, output_seq_length=24, target='Energy'):
    X, y = [], []
    max_idx = len(df) - input_seq_length - output_seq_length + 1
    for i in range(max_idx):
        X.append(df.iloc[i:i+input_seq_length].values)
        y.append(df.iloc[i+input_seq_length:i+input_seq_length+output_seq_length][target].values)
    return np.array(X), np.array(y).reshape(max_idx, output_seq_length, 1)

from sklearn.preprocessing import MinMaxScaler

# Function to load and preprocess data with sequences
def load_and_preprocess_data_with_sequences(df, target='Energy', scaled=False, scale_type='features', val_ratio=0.1, test_ratio=0.1, input_seq_length=48, output_seq_length=24):
    """
    Parameters:
    - df: DataFrame containing the data
    - target: Target variable for prediction
    - scaled: Boolean flag to apply scaling
    - scale_type: Specify scaling for 'features', 'output', or 'both'
    - val_ratio: Ratio of the dataset to be used for validation
    - test_ratio: Ratio of the dataset to be used for testing
    - input_seq_length: Number of time steps in the input sequences
    - output_seq_length: Number of time steps in the output sequences
    
    Returns:
    - X_train, y_train, X_val, y_val, X_test, y_test, X, y, df_scaled
    """

    # Drop NA values or consider imputation
    df.dropna(inplace=True)

    # Calculate the number of rows to lose at the beginning and end
    rows_to_lose = input_seq_length + output_seq_length - 1
    effective_rows = len(df) - rows_to_lose

    # Calculate sizes for test and validation
    test_size = int(effective_rows * test_ratio)
    val_size = int(effective_rows * val_ratio)
    train_size = effective_rows - test_size - val_size

    # Check for edge cases
    if test_size == 0 or val_size == 0 or train_size <= 0:
        raise ValueError("Dataset is too small to be split into train, validation, and test sets with the given ratios.")

    if scaled:
        # Initialize scalers
        feature_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

        if scale_type == 'features' or scale_type == 'both':
            # Fit and transform the features scaler on the training data
            feature_columns = df.columns[df.columns != target]
            feature_scaler.fit(df[feature_columns].iloc[:train_size + input_seq_length + output_seq_length - 1])
            df[feature_columns] = feature_scaler.transform(df[feature_columns])
        
        if scale_type == 'output' or scale_type == 'both':
            # Fit and transform the output scaler on the training data
            output_scaler.fit(df[[target]].iloc[:train_size + input_seq_length + output_seq_length - 1])
            df[[target]] = output_scaler.transform(df[[target]])

        df_scaled = df
    else:
        df_scaled = df

    # Generate sequences
    X, y = seq_data(df_scaled, input_seq_length, output_seq_length, target)

    # Split sequences into training, validation, and test sets
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, df_scaled























import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Flatten, Concatenate, TimeDistributed, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, SpatialDropout1D, Add, Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam


class MAPECallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_val_pred = self.model.predict(X_val)
        mape_score = mean_absolute_percentage_error(y_val[:, 0].flatten(), y_val_pred[:, 0].flatten()) * 100
        print(f"Epoch {epoch + 1} - MAPE: {mape_score:.2f}%")





#RNN
def create_lstm_model(trial, input_shape, output_shape):
    model = Sequential()

    lstm_units = trial.suggest_int('lstm_units', 64, 128)  # Narrowed and consistent with successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(lstm_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model



def create_gru_model(trial, input_shape, output_shape):
    model = Sequential()

    gru_units = trial.suggest_int('gru_units', 64, 128)  # Consistent with other successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_gru_layers):
        return_sequences = (i < num_gru_layers - 1)
        model.add(GRU(gru_units, return_sequences=return_sequences))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(gru_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def create_blstm_model(trial, input_shape, output_shape):
    model = Sequential()

    lstm_units = trial.suggest_int('lstm_units', 64, 128)  # Narrowed and consistent with successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=return_sequences)))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(lstm_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def create_bgru_model(trial, input_shape, output_shape):
    model = Sequential()

    gru_units = trial.suggest_int('gru_units', 64, 128)  # Consistent with other successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_gru_layers):
        return_sequences = (i < num_gru_layers - 1)
        model.add(Bidirectional(GRU(gru_units, return_sequences=return_sequences)))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(gru_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model














#CNN

def create_cnn_model(trial, input_shape, output_shape):
    model = Sequential()
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)

    model.add(Input(shape=input_shape))
    for i in range(num_conv_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg)))
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def create_tcn_model(trial, input_shape, output_shape):
    # Hyperparameter suggestions
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_tcn_blocks = trial.suggest_int('num_tcn_blocks', 2, 5)
    dilation_base = trial.suggest_int('dilation_base', 2, 4)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Define the TCN block
    def tcn_block(x, filters, kernel_size, dilation_rate, l2_reg):
        x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        return x

    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_tcn_blocks):
        dilation_rate = dilation_base ** i
        prev_x = x  # Save the previous layer's output
        x = tcn_block(x, filters, kernel_size, dilation_rate, l2_reg)

        # Adjust the shape of prev_x to match x using a 1x1 convolution
        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = Conv1D(filters, kernel_size=1, padding='same')(prev_x)

        # Skip connection (Residual)
        x = Add()([x, prev_x])

    x = Flatten()(x)
    x = Dense(output_shape, activation='linear')(x)

    model = Model(inputs, x)

    # Compile the model
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def create_dilated_cnn_model(trial, input_shape, output_shape):
    model = Sequential()
    
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dilation_rate = trial.suggest_int('dilation_rate', 1, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    
    model.add(Input(shape=input_shape))
    
    for i in range(num_conv_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, 
                         dilation_rate=dilation_rate, activation='relu', 
                         kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg)))
    
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model


def create_wavenet_model(trial, input_shape, output_shape):
    # Hyperparameter suggestions
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dilation_rate = trial.suggest_int('dilation_rate', 1, 4)
    num_wavenet_blocks = trial.suggest_int('num_wavenet_blocks', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    
    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Define a single WaveNet block
    def wavenet_block(x, filters, kernel_size, dilation_rate, l2_reg):
        res_x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal',
                       kernel_regularizer=l2(l2_reg))(x)
        res_x = Activation('relu')(res_x)
        res_x = Dropout(dropout_rate)(res_x)
        
        # Adjust the shape of x to match res_x
        if x.shape[-1] != filters:
            x = Conv1D(filters, 1, padding='same', kernel_regularizer=l2(l2_reg))(x)
        
        res_x = Conv1D(filters, 1, padding='same', kernel_regularizer=l2(l2_reg))(res_x)
        return Add()([x, res_x])  # Residual connection
    
    # Add multiple WaveNet blocks
    for i in range(num_wavenet_blocks):
        x = wavenet_block(x, filters, kernel_size, dilation_rate, l2_reg)
    
    # Flatten and add final Dense layer to match output shape
    x = Flatten()(x)
    outputs = Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg))(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model


















def create_tft_model(trial, input_shape, output_shape):

    # Hyperparameter suggestions
    hidden_units = trial.suggest_int('hidden_units', 32, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 4)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_attention_blocks = trial.suggest_int('num_attention_blocks', 1, 4)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Input layer
    inputs = Input(shape=input_shape)

    # LSTM layers with Layer Normalization and Residual Connections
    lstm_output = inputs
    for i in range(num_lstm_layers):
        lstm_output_residual = lstm_output
        lstm_output = LSTM(hidden_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lstm_output)
        lstm_output = LayerNormalization()(lstm_output)  # Layer Normalization after LSTM
        
        # Adjust the dimension of residual to match lstm_output
        if lstm_output_residual.shape[-1] != lstm_output.shape[-1]:
            lstm_output_residual = Dense(hidden_units)(lstm_output_residual)
        
        lstm_output = Add()([lstm_output, lstm_output_residual])  # Residual Connection
        lstm_output = Dropout(dropout_rate)(lstm_output)

    # Attention mechanism (Temporal fusion)
    attention_heads = []
    for _ in range(num_attention_blocks):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(lstm_output, lstm_output)
        attention_output = LayerNormalization()(attention_output)
        attention_heads.append(attention_output)

    if len(attention_heads) > 1:
        lstm_output = Concatenate()(attention_heads)
    else:
        lstm_output = attention_heads[0]

    # Final dense layers to match the output shape
    lstm_output = TimeDistributed(Dense(hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))(lstm_output)
    lstm_output = Dropout(dropout_rate)(lstm_output)

    # Automatically adapt to the output sequence length
    lstm_output = lstm_output[:, -output_shape:, :]
    lstm_output = TimeDistributed(Dense(1, activation='linear'))(lstm_output)

    # Model creation
    model = Model(inputs, lstm_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model





class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def positional_encoding(self, sequence_length, d_model):
        angle_rads = self.get_angles(
            tf.cast(tf.range(sequence_length)[:, tf.newaxis], tf.float32),
            tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32),
            d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        angle_rads = tf.concat([sines, cosines], axis=-1)

        pos_encoding = angle_rads[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def create_tst_model(trial, input_shape, output_shape):
    # Hyperparameter suggestions
    d_model = trial.suggest_int('d_model', 32, 128)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', 2, 6)
    ff_dim = trial.suggest_int('ff_dim', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Input layer
    inputs = Input(shape=input_shape)

    # Linear projection to match d_model
    x = Dense(d_model)(inputs)

    # Positional encoding
    x = PositionalEncoding(input_shape[0], d_model)(x)

    # Transformer layers
    for _ in range(num_transformer_layers):
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        attn_output = Add()([x, attn_output])  # Residual connection
        attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(attn_output)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([attn_output, ffn_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)

    # Reduce the sequence length according to output_shape
    x = TimeDistributed(Dense(1, activation='linear'))(x)
    x = x[:, -output_shape:, :]  # Select the last 'output_shape' time steps

    # Model creation
    model = Model(inputs, x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model




def optimize_and_save_model(data, df_name, n_trials=50, epochs=100, verbosity=1, model_type='LSTM'):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Create the subfolder in Models and Plots directories if it doesn't exist
    result_dir = "./Results"
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f"./Results/Models/{df_name}"
    plot_dir = f"./Results/Plots/{df_name}"
    hyperparameters_file = "./Results/best_hyperparameters.csv"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Choose the model type
    if model_type == 'LSTM':
        model_creator = create_lstm_model
    elif model_type == 'GRU':
        model_creator = create_gru_model
    elif model_type == 'BLSTM':
        model_creator = create_blstm_model   
    elif model_type == 'BGRU':
        model_creator = create_bgru_model   
    elif model_type == 'CNN':
        model_creator = create_cnn_model
    elif model_type == 'TCN':
        model_creator = create_tcn_model
    elif model_type == 'DCNN':
        model_creator = create_dilated_cnn_model
    elif model_type == 'WaveNet':
        model_creator = create_wavenet_model
    elif model_type == 'TFT':
        model_creator = create_tft_model
    elif model_type == 'TST':
        model_creator = create_tst_model 
    else:
        raise ValueError("Invalid model type")

    def objective(trial):
        model = model_creator(trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=verbosity, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=verbosity)
        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=verbosity, callbacks=[early_stopping, reduce_lr])
        val_loss = model.evaluate(X_val, y_val, verbose=verbosity)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_model = model_creator(best_trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=verbosity, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=verbosity)
    # Fit the best model
    best_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=verbosity, callbacks=[early_stopping, reduce_lr])

    # Save the best model in TensorFlow SavedModel format
    best_model.save(f"{model_dir}/{model_type}.keras")

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
    r2 = round(r2_score(y_test, y_pred), 4)

    # print(f'MSE: {mse_score}')
    # print(f'RMSE: {rmse_score}')
    # print(f'MAE: {mae_score}')
    # print(f'MAPE: {mape_score}%')
    # print(f'SMAPE: {smape_score}%')
    # print(f'R²: {r2}')


    
    # Plot the performance with white background and border
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set the Y-axis limit based on the range of the data
    plt.ylim([min(y_test[:len(y_test)//10]) - 0.5, max(y_test[:len(y_test)//10]) + 0.5])
    
    # Set the X-axis limit based on the number of time steps plotted
    plt.xlim([0, len(y_test[:len(y_test)//10])])
    
    # Plot with custom colors
    plt.plot(y_test[:len(y_test)//10], label='True', color='#ed8787', alpha=0.99)  # Custom red color with transparency
    plt.plot(y_pred[:len(y_pred)//10], label='Predicted', color='#1f77b4', alpha=0.99)  # Custom blue color with transparency

    # Add the metrics to the title
    plt.title(f'Model={model_type}, MSE={mse_score:.2f}, RMSE={rmse_score:.2f}, MAE={mae_score:.2f}, MAPE={mape_score:.2f}%, SMAPE={smape_score:.2f}%, R²={r2:.2f}')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    
    # Add border lines (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # Optional: Adjust the position of the vertical and horizontal lines
    plt.axvline(x=0, color='black', linewidth=1)  # At x=0
    
    # Set the legend with default light gray background
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_facecolor('lightgray')
    
    # Save the plot as a PDF file with a white background
    plt.savefig(f"{plot_dir}/{model_type}_PerformancePlot.pdf", format='pdf', facecolor='white', bbox_inches='tight')
    
    plt.show()

    # Explicitly close the plot to avoid issues with subsequent plots
    plt.close()
    
    # Plot the optimization history with a larger figure size and white background
    fig = optuna.visualization.matplotlib.plot_optimization_history(study).figure

    # Adjust the figure settings
    fig.set_size_inches(10, 6)
    fig.patch.set_facecolor('white')

    # Access the axes to adjust spines
    ax = fig.gca()
    ax.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Adjust legend position inside the plot
    legend = ax.legend(loc='upper right')
    legend.get_frame().set_facecolor('lightgray')
    
    # Add a title to the plot
    plt.title('Optimization History')
    
    # Save the plot as a PDF file with a white background
    fig.savefig(f"{plot_dir}/{model_type}_OptimizationHistory.pdf", format='pdf', facecolor='white', bbox_inches='tight')
    
    # Show the plot
    plt.show()

    # Explicitly close the plot to avoid issues with subsequent plots
    plt.close(fig)


    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        "type": [model_type],
        "dataset_name": [df_name],
        "best_params": [str(best_trial.params)],
        "duration": [best_trial.duration.total_seconds()],
        "mse_score": [mse_score],
        "rmse_score": [rmse_score],
        "mae_score": [mae_score],
        "smape_score": [smape_score],
        "mape_score": [mape_score],
        "r2_score": [r2],
    })

    # Save results to 'best_hyperparameters.csv' in the Results folder
    try:
        existing_df = pd.read_csv(hyperparameters_file)
        if df_name in existing_df['dataset_name'].values:
            existing_df = existing_df[(existing_df['dataset_name'] != df_name) | (existing_df['type'] != model_type)]
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        combined_df = results_df

    combined_df.to_csv(hyperparameters_file, index=False)

    return best_model








import os
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.samplers import TPESampler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib

def optimize_and_save_rf_model(data, df_name, n_trials=10, verbosity=1, model_type='LSTM'):

    X_train, X_test, y_test = data

    # Create the subfolder in Models and Plots directories if it doesn't exist
    result_dir = "./Results"
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f"./Results/Models/{df_name}"
    plot_dir = f"./Results/Plots/{df_name}"
    hyperparameters_file = "./Results/best_hyperparameters.csv"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load the previously saved model
    best_model = load_model(f"{model_dir}/{model_type}.keras")

    # Use the model's predictions as the target (y_train)
    y_train = best_model.predict(X_train)[:, 0]  # Use the prediction of your model

    # Reshape X_train and X_test to 2D
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        # Define the model
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Evaluate model using cross-validation
        scores = cross_val_score(rf, X_train_reshaped, y_train, cv=5, scoring='neg_mean_squared_error')

        # Optuna minimizes the objective, so return the negative of the score
        return -scores.mean()

    # Create a study with TPE Sampler (Bayesian optimization)
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_trial = study.best_trial

    # Train the final model with the best parameters
    best_rf = RandomForestRegressor(**best_trial.params, random_state=42)
    best_rf.fit(X_train_reshaped, y_train)

    # Save the best Random Forest model
    joblib.dump(best_rf, f"{model_dir}/{model_type}_RandomForest.pkl")

    # Make predictions on the test set
    y_test_pred_flat = best_rf.predict(X_test_reshaped).flatten()

    # Calculate evaluation metrics
    y_test_flat = y_test[:, 0].flatten()

    mape = np.mean(np.abs((y_test_flat - y_test_pred_flat) / y_test_flat)) * 100
    smape = 100 * np.mean(2 * np.abs(y_test_pred_flat - y_test_flat) / (np.abs(y_test_flat) + np.abs(y_test_pred_flat)))
    mae = mean_absolute_error(y_test_flat, y_test_pred_flat)
    mse = mean_squared_error(y_test_flat, y_test_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_test_pred_flat)

    # print(f'MAPE: {mape:.2f}%')
    # print(f'SMAPE: {smape:.2f}%')
    # print(f'MAE: {mae:.4f}')
    # print(f'MSE: {mse:.4f}')
    # print(f'RMSE: {rmse:.4f}')
    # print(f'R²: {r2:.4f}')

    # Plot the optimization history with a white background and border lines
    fig = optuna.visualization.matplotlib.plot_optimization_history(study).figure

    # Adjust the figure settings
    fig.set_size_inches(10, 6)
    fig.patch.set_facecolor('white')

    # Access the axes to adjust spines
    ax = fig.gca()
    ax.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Adjust legend position inside the plot
    legend = ax.legend(loc='upper right')
    legend.get_frame().set_facecolor('lightgray')

    # Add a title to the plot
    plt.title('Optimization History')

    # Save the plot as a PDF file with a white background
    fig.savefig(f"{plot_dir}/{model_type}_RandomForest_OptimizationHistory.pdf", format='pdf', facecolor='white', bbox_inches='tight')

    plt.show()

    # Explicitly close the plot to avoid issues with subsequent plots
    plt.close()

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        "type": [f"{model_type}_RandomForest"],
        "dataset_name": [df_name],
        "best_params": [str(best_trial.params)],
        "duration": [best_trial.duration.total_seconds()],
        "mse_score": [mse],
        "rmse_score": [rmse],
        "mae_score": [mae],
        "mape_score": [mape],
        "smape_score": [smape],
        "r2_score": [r2],
    })

    # Append or replace the results in the CSV file
    hyperparameters_file = os.path.join(result_dir, "best_hyperparameters.csv")
    try:
        existing_df = pd.read_csv(hyperparameters_file)
        if df_name in existing_df['dataset_name'].values:
            existing_df = existing_df[(existing_df['dataset_name'] != df_name) | (existing_df['type'] != f"{model_type}_RandomForest")]
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        combined_df = results_df

    combined_df.to_csv(hyperparameters_file, index=False)

    return best_rf
