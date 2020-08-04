import datetime
import time
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from keras import metrics
import math
from statsmodels.compat.pandas import deprecate_kwarg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# additive decompose a contrived additive time series
from random import randrange
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# the main library has a small set of functionality
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,drift, mean,seasonal_naive)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
         # find the end of this pattern
         end_ix = i + n_steps_in
         out_end_ix = end_ix + n_steps_out
         # check if we are beyond the sequence
         if out_end_ix > len(sequence):
             break
         # gather input and output parts of the pattern
         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
         X.append(seq_x)
         y.append(seq_y)
    return np.array(X), np.array(y)



def get_forcast_per_component(series, st_in, st_out, train_test_size):

    X, y = split_sequence(series, st_in, st_out)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    y = y.reshape((y.shape[0], y.shape[1], n_features))

    train_X, test_X = X[:train_test_size], X[train_test_size:]
    train_y, test_y = y[:train_test_size], y[train_test_size:]

    # define model
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(st_in, n_features)))
    model.add(RepeatVector(st_out))
    model.add(LSTM(20, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse',metrics=[metrics.mae, 'accuracy'] )

    # fit model
    model.fit(train_X, train_y, epochs=8, verbose=0)

    # predict
    predicted = []
    for i in range(len(test_X)):
        x_input = test_X[i].reshape(1, st_in, n_features)
        yhat = model.predict(x_input, verbose=0)

        #predicted.append(np.rint(yhat[0]))
        predicted.append(np.around(yhat[0], decimals=1))
    predicted = np.array(predicted)
    return predicted

dict2data = {}
dict2MAPE = {}
dict2RMSE = {}

data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv', index_col = 0,
                   header=0, parse_dates=True)

agg_by_cell = data.groupby(by = ['cell_num'])
print(len(agg_by_cell), 'celle')

n_steps_in, n_steps_out = 168, 6
train_test_size = 2600

counter = 0
for cell_i, k_frame in agg_by_cell:
    k_frame = k_frame.iloc[::4,:]
    dates4dec = []
    cell_values = []
    counter +=1
    print(counter)
    #if counter > 3: break

    for index, row in k_frame.iterrows():

        date = row['date']
        h = str(row['hours'])

        h = h.split('.')

        if len(h[0])<2:
            h = h[1]+h[0]
        else:
            h = h[0]

        minutes = str(row['minutes'])
        m = ''
        minutes = minutes.split('.')
        if len(minutes[0])<2:
            m = minutes[0] +'0'
        else:
            m = minutes[0]
        #print(date, h, m)
        data_f = date+' '+h+':'+m+':'+'00'
        #print(data_f)
        cell_values.append(row['nr_people'])
        dates4dec.append(data_f)

    dict_i = {'ds': dates4dec, 'y':cell_values}
    data4deco = pd.DataFrame(dict_i, index=None, columns=None)

    data4deco['ds'] = pd.to_datetime(data4deco['ds'])
    data4deco = data4deco.set_index('ds')

    decomp = decompose(data4deco['y'], period=24)

    trend = decomp.trend.values
    seasonal = decomp.seasonal.values
    residual = decomp.resid.values

    forcasted_trend = get_forcast_per_component(trend, n_steps_in, n_steps_out, train_test_size)
    forcasted_residual = get_forcast_per_component(residual, n_steps_in, n_steps_out, train_test_size)
    forcasted_season = get_forcast_per_component(seasonal, n_steps_in, n_steps_out, train_test_size)

    final_prediction = forcasted_trend + forcasted_residual + forcasted_season

    X, y = split_sequence(k_frame['nr_people'].values, n_steps_in, n_steps_out)

    train_X, train_y = X[:train_test_size], X[train_test_size:]
    train_y, test_y = y[:train_test_size], y[train_test_size:]
    expected = test_y
    difference = abs(expected - final_prediction)

    mean_error = np.reshape(difference, difference.shape[0] * difference.shape[1])

    minimum = np.amin(mean_error)
    per75 = np.percentile(mean_error, 75)
    per50 = np.percentile(mean_error, 50)
    per25 = np.percentile(mean_error, 25)
    maximum = np.amax(mean_error)
    l5i = [minimum, per25, per50, per75, maximum]
    dict2data[cell_i] = l5i

    power_err = np.power(mean_error, 2)

    rmse = math.sqrt(np.mean(power_err))
    print(rmse)
    dict2RMSE[cell_i] = rmse

    percent_err = 100 * (difference/final_prediction)
    MAPE = np.mean(abs(percent_err))
    dict2MAPE[cell_i] = MAPE

with open('MAE_error_data_4_LSTM_with_STL_Decomposition_in6.csv', 'w') as f:
    for key, value in dict2data.items():
        f.write('%s:%s\n' % (key, value))
with open('MAPE_error_data_4_LSTM_with_STL_Decomposition_in6.csv', 'w') as f:
    for key, value in dict2MAPE.items():
        f.write('%s:%s\n' % (key, value))
with open('RMSE_error_data_4_LSTM_with_STL_Decomposition_in6.csv', 'w') as f:
    for key, value in dict2RMSE.items():
        f.write('%s:%s\n' % (key, value))
