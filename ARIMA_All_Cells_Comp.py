from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings('ignore')

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


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


def get_forcast(series, st_in, st_out, params_per_cell):
    
    # split into samples
    X, y = split_sequence(series, st_in, st_out)
    ar, ii, ma = params_per_cell.split(',')
    ar = int(ar)
    ii = int(ii)
    ma = int(ma)
    days_in_year = 24
    count = 0
    result = []
    for i in range(len(X)):
        count += 1
        differenced = difference(X[i], days_in_year)
        #print(count)
        # fit model
        model = ARIMA(differenced, order=(ar,ii, ma))
        model_fit = model.fit(disp=0)

        # multi-step out-of-sample forecast
        forecast = model_fit.forecast(steps=st_out)[0]

        # invert the differenced forecast to something usable
        history = [x for x in X[i]]
        day = 1
        
        predicted = []
        
        for yhat in forecast:
            inverted = inverse_difference(history, yhat, days_in_year)
            history.append(inverted)
            day += 1
            predicted.append(inverted)
        
        predicted = np.array(predicted)    
        result.append(predicted)
    return result

# prepare parameters
parameters = pd.read_csv('BestARIMA_config_parametres_V2.csv', header = None, sep = ':')
par_per_cell = {}
for index, row in parameters.iterrows(): 
    cell = row[0]
    rest = row[1]
    #print(rest)
    par = rest.split('),')
    #print(par[0])
    f_s = par[0]
    fs = f_s[2:]
    #print(fs)
    par_per_cell[cell] = fs

# set rolling window    
step_in, step_out = 2600, 6

data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv', index_col = 0, 
                   header=0, parse_dates=True)
agg_by_cell = data.groupby(by = ['cell_num'])

count = 0
dict2data = {}
dict2MAPE = {}
dict2RMSE = {}
for i, k in agg_by_cell:
    predicted = []
    cell = i
    param_per_cell_i = par_per_cell[cell]
    
    print('Calcolo cella ', count)
    count +=1
    if count < 61: continue
    series_i = k.iloc[::4, :]
    
    series_i = series_i['nr_people'].values
   
    X_data, y_data = split_sequence(series_i, step_in, step_out)
    
    forcasted = get_forcast(series_i, step_in, step_out, param_per_cell_i)
    error = abs(forcasted - y_data)
    pow_err = np.power(error, 2)
    rmse = math.sqrt(np.mean(pow_err))
    
    mape_i = np.mean(100 * np.divide(error, y_data))
    
    # collect data 2 dictionary
    minimum = np.amin(error)   
    per75 = np.percentile(error, 75)
    per50 = np.percentile(error, 50)
    per25 = np.percentile(error, 25)
    maximum = np.amax(error)
    l5i = [minimum, per25, per50, per75, maximum]
    
    dict2data[cell] = l5i
    dict2MAPE[cell] = mape_i
    dict2RMSE[cell] = rmse

with open('MAE_error_data_4_ARIMA_MultistepForcast_6_V3.csv', 'w') as f:
    for key, value in dict2data.items():
        f.write('%s:%s\n' % (key, value))
        
with open('MAPE_error_data_4_ARIMA_MultistepForcast_6_V3.csv', 'w') as f:
    for key, value in dict2MAPE.items():
        f.write('%s:%s\n' % (key, value))  
        
with open('RMSE_error_data_4_ARIMA_MultistepForcast_6_V3.csv', 'w') as f:
    for key, value in dict2RMSE.items():
        f.write('%s:%s\n' % (key, value))  