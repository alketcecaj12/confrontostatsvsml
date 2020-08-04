import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv',index_col = 0)

gbc = data.groupby(by = data['cell_num'])


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
    
    model = ExponentialSmoothing(history, trend=t, damped=False, seasonal=s, seasonal_periods=p)
    
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    
    #  make one step forecast
    yhat = model_fit.predict(len(history), len(history)) 
    return yhat[0]


ETS_params = pd.read_csv('Params_data_4_All_Cells_ETS.csv', header = None, sep = ':')
counter = 0
par_per_cell = {}
nonetype = None
for index, row in ETS_params.iterrows(): 
    cell = row[0]
    rest = row[1]
    parameters_list = list()
    par = rest.split('],')
    parameters = par[0][1:len(par[0])]
   
    params = parameters.split(',')

    parameters_list.insert(0, None) 
    parameters_list.insert(1, bool(params[1]))
    sist = params[2]
    par_i = ''
   
    if len(sist) == 5: 
        par_i = sist[1:len(sist)]
        
    else:
        par_i = sist[2:len(sist)-1]
       
    parameters_list.insert(2, par_i)
    parameters_list.insert(3, 24)
    parameters_list.insert(4, bool(params[4]))
    parameters_list.insert(5,bool(params[5]))
    
    par_per_cell[cell] = parameters_list
    
dict2data = {}
dict2MAPE = {}
dict2RMSE = {}
step_in, step_out = 2600, 96


counter = 0
for index, k_frame in gbc:
    cell = index
    
    print('Previsione per cella ', counter)
    counter += 1
    cfg_params = par_per_cell[cell] 
    
    k_frame = k_frame.iloc[::4, :]
    series_i = k_frame['nr_people'].values
    
    X, y = split_sequence(series_i, step_in, step_out)
 
    yHat = []
    expected = []
    
    for k in range(len(X)):
       
        predictions = []
    
        history = X[k]
        #print(history[:10])
        for i in range(len(y[k])):
       
            yhat = exp_smoothing_forecast(history, cfg_params)
            predictions.append(yhat)
            history = np.append(history, y[i])
            #print('i', i)
        expected.append(y[k])
        yHat.append(predictions)
        #print(k)
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

with open('MAE_error_data_4_ETS_MultistepForcast_96.csv', 'w') as f:
    for key, value in dict2data.items():
        f.write('%s:%s\n' % (key, value))
        
with open('MAPE_error_data_4_ETS_MultistepForcast_96.csv', 'w') as f:
    for key, value in dict2MAPE.items():
        f.write('%s:%s\n' % (key, value))  
        
with open('RMSE_error_data_4_ETS_MultistepForcast_96.csv', 'w') as f:
    for key, value in dict2RMSE.items():
        f.write('%s:%s\n' % (key, value))    