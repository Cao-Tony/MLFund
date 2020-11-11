import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow import keras

from securities.calculations import *

dataFile = '../dataFiles/EURUSD=X (1).csv'
df = pd.read_csv(dataFile)
df_predict = pd.read_csv(dataFile)

# clean out NaN rows
df = df.dropna()
df_predict = df_predict.dropna()

# set index of df
df = df.set_index(pd.DatetimeIndex(df['Date'].values))

# create bollinger bands
indicator_bb = ta.volatility.BollingerBands(close=df["Close"], n=20, ndev=2, fillna=True)
df['bb_mid'] = indicator_bb.bollinger_mavg()
df['bb_high'] = indicator_bb.bollinger_hband()
df['bb_low'] = indicator_bb.bollinger_lband()

# Bollinger band indicators
df['bb_hindicator'] = indicator_bb.bollinger_hband_indicator()
df['bb_lindicator'] = indicator_bb.bollinger_lband_indicator()

# create RSI indicator
indicator_rsi = ta.momentum.RSIIndicator(close=df['Close'], n=14, fillna=True)
df['rsi'] = indicator_rsi.rsi()


########################################################################################################################

# Time series analysis
# - can be used to predict future stock prices

########################################################################################################################

# create new df with only close price
data = df.filter(["Close"])
data_predict = df_predict.filter(["Close"])

# convert df to numpy array
dataset = data.values

# get the number of rows to train the model
# Split df into 3 sets of data
training_data_len = math.ceil(len(dataset) * .6)  # 60%
valid_data_len = math.ceil(len(dataset) * .2)     # 20%
test_data_len = math.ceil(len(dataset) * .2)      # 20%

# scale the data [0,1] inclusive. preprocessing, scaling, or normalization of input data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

# split data into x_train and y_train
x_train = []
y_train = []

# append the last 60 data values (days) to our training data set
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape x_train data set because LSTM expects 3 dimensions and x_train is originally 2 dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#######################################################################################

# build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#######################################################################################

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

########################################################################################################################

### create the testing data set ###

# create new array containing scaled value
valid_data = scaled_data[training_data_len - 60:training_data_len + valid_data_len, :]

# create test data set x_test and y_test
x_valid = []
y_valid = dataset[training_data_len:training_data_len + valid_data_len, :]  # values we want our model to predict

for i in range(60, len(valid_data)):
    x_valid.append(valid_data[i-60:i, 0])

# convert data to numpy array
x_valid = np.array(x_valid)

# reshape data for LSTM model
x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

# get model's predicted price values
predictions_valid = model.predict(x_valid)

# unscaling the values
# we want prediction to contain the same values as our y_test data set
predictions_valid = scaler.inverse_transform(predictions_valid)

# get ROOT MEAN SQUARED ERROR ( RMSE ) to see how far our predictions are to y_test
rmse_valid = np.sqrt(np.mean(((predictions_valid - y_valid)**2)))

# plot the data
train = data[:training_data_len]
valid = data[training_data_len:training_data_len + valid_data_len]
valid['Predictions_valid'] = predictions_valid

########################################################################################################################

# make predictions for last 20% of the data
# create new array containing scaled value
test_data = scaled_data[(training_data_len - 60) + valid_data_len:, :]

# create test data set x_test and y_test
x_test = []
y_test = dataset[training_data_len + valid_data_len:, :]  # values we want our model to predict

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# convert data to numpy array
x_test = np.array(x_test)

# reshape data for LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get model's predicted price values
predictions_test = model.predict(x_test)

# unscaling the values
# we want prediction to contain the same values as our y_test data set
predictions_test = scaler.inverse_transform(predictions_test)

# get ROOT MEAN SQUARED ERROR ( RMSE ) to see how far our predictions are to y_test
rmse_test = np.sqrt(np.mean(((predictions_test - y_test)**2)))

# plot the data
test = data[training_data_len + valid_data_len:]
test['Predictions_test'] = predictions_test

########################################################################################################################

# TODO
# get predictions for the next year --> 261 days

new_df = data_predict  # pd df
end = len(new_df)
print(new_df)

for i in range(0, 261):
    next_day_pred = get_next_year_pred(new_df, model, scaler, np)
    new_df = new_df.append({'Close': next_day_pred[0, 0]}, ignore_index=True)

print(new_df[end:])

train_f = data_predict

# visualize the data
plt.figure(figsize=(12.2, 4.5))
plt.plot(train_f['Close'])
plt.plot(new_df[end:])
plt.title('EUR/USD Adj Price History (Model)')
plt.xlabel('Dates', fontsize=18)
plt.ylabel('Price ($)', fontsize=18)
plt.legend(['Train set', 'next year predictions'])
plt.show()


