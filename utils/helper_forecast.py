import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from prophet import Prophet
import itertools
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator



def test_stationarity(data):
    result = adfuller(data)
    p_value = result[1]

    if p_value <= 0.05:
        print('Data is stationary, nothing returned')
        return None
    else:
        print('Data is not stationary so differencing applied')
        return data.diff().dropna()
        

def rmse(val, pred):
    return np.sqrt(mean_squared_error(val, pred))

def mape(val, pred):

    if len(val) != len(pred):
        raise ValueError("Input arrays must have the same length")

    # find the absolute percentage errors for each data point
    absolute_percentage_errors = np.abs((np.array(val) - np.array(pred)) / np.array(val))

    # find the mean absolute percentage error
    mape = np.mean(absolute_percentage_errors) * 100

    return mape

def find_best_sarima(train, val):

    # define ranges for model paramyers p, d, q, P, D, Q, and S
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    s_values = [12]

    # list of all possible parameter combinations
    param_combos = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values))

    top_results_rmse = []
    top_results_mape = []

    # iterate through parameter combinations and fit SARIMA models
    for params in param_combos:
        p, d, q, P, D, Q, s = params
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        
        try:
            model = sm.tsa.statespace.SARIMAX(train['Total_peak_demand'], order=order, seasonal_order=seasonal_order)
            results = model.fit()
            val_predictions = results.predict(start=len(train), end=len(train) + len(val) - 1, dynamic=False)
            val_rmse = rmse(val['Total_peak_demand'], val_predictions)
            val_mape = mape(val['Total_peak_demand'], val_predictions)

            top_results_rmse.append((val_rmse, order, seasonal_order))
            top_results_mape.append((val_mape, order, seasonal_order))
        
        except:
            continue

    # sorting top 3 top_results by RMSE 
    top_results_rmse.sort()
    
    # sorting top 3 top_results by MAPE
    top_results_mape.sort()

    return (top_results_rmse[:3], top_results_mape[:3])


def find_best_prophet(pr_train, pr_val):

    # define hyperparameter search space
    seasonality_prior_scale_values = [0.001, 0.01, 0.1, 1.0]
    changepoint_prior_scale_values = [0.001, 0.01, 0.1, 1.0]
    holidays_prior_scale_values = [0.001, 0.01, 0.1, 1.0]

    # list of all possible parameter combinations
    param_combinations = list(itertools.product(seasonality_prior_scale_values, changepoint_prior_scale_values, holidays_prior_scale_values))

    top_results_rmse = []
    top_results_mape = []

    # iterate through parameter combinations and fit Prophet 
    for params in param_combinations:
        seasonality_prior_scale, changepoint_prior_scale, holidays_prior_scale = params
        
        try:
            model = Prophet(seasonality_prior_scale=seasonality_prior_scale, changepoint_prior_scale=changepoint_prior_scale, holidays_prior_scale=holidays_prior_scale)
            model.fit(pr_train.rename(columns={'date': 'ds', 'Total_peak_demand': 'y'}))
            
            future_val = pd.DataFrame(pr_val['date'].values, columns=['ds'])
            forecast = model.predict(future_val)
            
            val_rmse = rmse(pr_val['Total_peak_demand'], forecast['yhat'])
            val_mape = mape(pr_val['Total_peak_demand'], forecast['yhat'])

            top_results_rmse.append((val_rmse, params)) 
            top_results_mape.append((val_mape, params))          
        except:
            continue

    # sorting top 3 top_results by RMSE 
    top_results_rmse.sort()
    
    # sorting top 3 top_results by MAPE
    top_results_mape.sort()

    return (top_results_rmse[:3], top_results_mape[:3])


def find_best_LSTM(lstm_train, lstm_val):
        
    # define hyperparameter search space
    lstm_units_values = [64, 128, 256]
    batch_size_values = [32, 64, 128]
    learning_rate_values = [0.001, 0.01, 0.1]
    
    # list of all possible parameter combinations
    param_combinations = list(itertools.product(lstm_units_values, batch_size_values, learning_rate_values))

    top_results_rmse = []
    top_results_mape = []
    
    # extract the 'seasonal_adj_demand' column from your NumPy arrays
    train_demand = lstm_train['seasonal_adj_demand'].values.reshape(-1, 1)
    val_demand = lstm_val['seasonal_adj_demand'].values.reshape(-1, 1)

       

    # fit the scaler on your training data and transform both training and validation data
    scaler = MinMaxScaler()
    train_demand_scaled = scaler.fit_transform(train_demand)
    val_demand_scaled = scaler.transform(val_demand)

    # iterate through parameter combinations and fit LSTM 
    for params in param_combinations:
        lstm_units, batch_size, learning_rate = params
        
        try:
            model = Sequential()
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(8, 1)))
            model.add(Dropout(0.2)) 
            model.add(Dense(1))
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            
            # define early stopping to prevent overfitting
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
            #y_train = lstm_train['seasonal_adj_demand'].values
            #y_val = lstm_val['seasonal_adj_demand'].values
            
            history = model.fit(
                lstm_train.drop(columns=['date']), train_demand_scaled,  
                batch_size=batch_size,
                epochs=300,
                validation_data=(lstm_val.drop(columns=['date']), val_demand_scaled), 
                callbacks=[early_stopping],
                verbose=0
            )
            
            # extract X_val from lstm_val
            X_val = lstm_val.drop(columns=['date']).values
            
            # make predictions on the validation set
            val_predictions = model.predict(X_val)


            # Inverse transform the validation predictions using a scaler fitted to the training data
            scaler = MinMaxScaler()
            scaler.fit(val_demand)  # Use the numpy array for fitting
            val_predictions = scaler.inverse_transform(val_predictions)
            
            val_rmse = rmse(val_demand_scaled, val_predictions)
            val_mape = mape(val_demand_scaled, val_predictions)
            
            top_results_rmse.append((val_rmse, params))
            top_results_mape.append((val_mape, params))
        
        except Exception as e:
            print(f"Error: {e}")
            continue

    # sorting top 3 top_results by RMSE 
    top_results_rmse.sort()
    
    # sorting top 3 top_results by MAPE
    top_results_mape.sort()

    return (top_results_rmse[:3], top_results_mape[:3])

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        X = data[i:i+seq_length]
        y = data[i+seq_length]
        print(X.shape)
        print(y.shape)
        
        # Check for NaN values in sequences
        if not np.isnan(X).any() and not np.isnan(y).any():
            # Flatten y to have the same shape as X
            y = np.array([y])
            sequences.append((X, y))
        else:
            print(f"Skipping sequence at index {i} due to NaN values.")
    return np.array(sequences)


def data_transform(ssn_train, ssn_val):

    ssn_train['date'] = pd.to_datetime(ssn_train['date'])
    train_demand = ssn_train['seasonal_adj_demand'].values.reshape(-1,1)
    val_demand = ssn_val['seasonal_adj_demand'].values.reshape(-1,1)


    scaler = MinMaxScaler()
    train_demand = scaler.fit_transform(train_demand)
    val_demand = scaler.fit_transform(val_demand)

    sequences = create_sequences(train_demand, 7)
    #X_train, = sequences[:, :-1],
    y_train = sequences[:, -1]
    X_train = np.array([sequence.reshape(7, 1) for sequence in sequences])


    sequences = create_sequences(val_demand, 7)
    X_val, y_val = sequences[:, :-1], sequences[:, -1]


    return (X_train, y_train, X_val, y_val)



def series_gen(train_demand, val_demand, batch_size):
    seq_length = 7

    
    # Extract the 'seasonal_adj_demand' column from your NumPy arrays
    train_demand = train_demand['seasonal_adj_demand'].values.reshape(-1, 1)
    val_demand = val_demand['seasonal_adj_demand'].values.reshape(-1, 1)

   
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on your training data and transform both training and validation data
    train_demand_scaled = scaler.fit_transform(train_demand)
    val_demand_scaled = scaler.transform(val_demand)

    # Create a TimeseriesGenerator for training data
    train_generator = TimeseriesGenerator(
        train_demand_scaled,  # Your scaled training data
        train_demand_scaled,  # Your target data (can be the same as the input for forecasting)
        length=seq_length,    # Length of the sequences
        batch_size=batch_size # Batch size for training
    )

    # Create a TimeseriesGenerator for validation data
    val_generator = TimeseriesGenerator(
        val_demand_scaled,    # Your scaled validation data
        val_demand_scaled,    # Your target data (can be the same as the input for forecasting)
        length=seq_length,    # Length of the sequences
        batch_size=batch_size # Batch size for validation
    )
    print(val_generator)
    return (train_generator, val_generator)



def find_best_LSTM_transform(train_data, val_data):
    # Define hyperparameter search space
    lstm_units_values = [64, 128, 256]
    batch_size_values = [32, 64, 128]
    learning_rate_values = [0.001, 0.01, 0.1]

    # List of all possible parameter combinations
    param_combinations = list(itertools.product(lstm_units_values, batch_size_values, learning_rate_values))

    top_results_rmse = []
    top_results_mape = []

    # Iterate through parameter combinations and fit LSTM
    for params in param_combinations:
        lstm_units, batch_size, learning_rate = params

        train_gen, val_gen = series_gen(train_data, val_data, batch_size)

        try:
            model = Sequential()
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(7, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)

            # Define early stopping to prevent overfitting
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

            history = model.fit(
                train_gen,
                epochs=250,
                validation_data=(val_gen),
                callbacks=[early_stopping],
                verbose=0
            )

            # Make predictions on the validation set using the generator
            val_predictions = model.predict(val_gen['seasonal_adj_demand'])

            # Inverse transform the validation predictions using a scaler fitted to the training data
            scaler = MinMaxScaler()
            scaler.fit(train_data['seasonal_adj_demand'])  # Use the numpy array for fitting
            val_predictions = scaler.inverse_transform(val_predictions)

            # Extract actual validation data (ground truth)
            actual_val_data = val_data[7:,'seasonal_adj_demand']  # Assuming first 7 data points were used for the initial sequences

            val_rmse = rmse(actual_val_data, val_predictions)
            val_mape = mape(actual_val_data, val_predictions)

            top_results_rmse.append((val_rmse, params))
            top_results_mape.append((val_mape, params))

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Sorting top 3 top_results by RMSE
    top_results_rmse.sort()

    # Sorting top 3 top_results by MAPE
    top_results_mape.sort()

    return (top_results_rmse[:3], top_results_mape[:3])
