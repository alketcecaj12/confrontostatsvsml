import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv',index_col = 0)

gbc = data.groupby(by = data['cell_num'])
cell_1 = gbc.get_group('486-1258')
cell_1 = cell_1.iloc[::4, :]

series1 = cell_1['nr_people'].values

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

# one-step Holt Winters Exponential Smoothing forecast
def exp_smoothing_forecast(history, config): 
    t,d,s,p,b,r = config
    model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    
    #  make one step forecast
    yhat = model_fit.predict(len(history), len(history)) 
    return yhat[0]

dict2data = {}
dict2MAPE = {}
dict2RMSE = {}
step_in, step_out = 2600, 48
cfg_par = [None, False, 'add', 24, False, False]

counter = 0
for index, k_frame in gbc:
    cell = index
    counter += 1
    print(counter)
    #if counter > 3: break
        
    k_frame = k_frame.iloc[::4, :]
    series_i = k_frame['nr_people'].values
    
    X, y = split_sequence(series_i, step_in, step_out)
    #print(X.shape)
    
    yHat = []
    expected = []

    for k in range(len(X)):
       
        predictions = []
    
        history = X[k]
        #print(history[:10])
        for i in range(len(y[k])):
       
            yhat = exp_smoothing_forecast(history, cfg_par)
            predictions.append(yhat)
            history = np.append(history, y[i])
        
        expected.append(y[k])
        yHat.append(predictions)
   
    expected = np.array(expected)
    yHat = np.array(yHat)
    
    error = abs(expected - yHat)
    pow_err = np.power(error, 2)
    rmse = sqrt(np.mean(pow_err))
    
    mape_i = np.mean(np.divide(100 * error, expected))
    
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

with open('MAE_error_data_4_ETS_MultistepForcast_48.csv', 'w') as f:
    for key, value in dict2data.items():
        f.write('%s:%s\n' % (key, value))
        
with open('MAPE_error_data_4_ETS_MultistepForcast_48.csv', 'w') as f:
    for key, value in dict2MAPE.items():
        f.write('%s:%s\n' % (key, value))  
        
with open('RMSE_error_data_4_ETS_MultistepForcast_48.csv', 'w') as f:
    for key, value in dict2RMSE.items():
        f.write('%s:%s\n' % (key, value))  
