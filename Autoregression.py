import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from pandas import datetime
from random import random
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AR
import warnings
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
warnings.filterwarnings('ignore')
data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv',index_col = 0)

gbc = data.groupby(by = data['cell_num'])
cell_1 = gbc.get_group('486-1258')
cell_2 = gbc.get_group('498-1268')
series1 = cell_1['nr_people'].tolist()

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


def get_AR_prediction(X, st_in, st_out):
    
    # split into samples
    X_data, y_data = split_sequence(X, st_in, st_out)    
    tot_predictions = []
    
    for i in range(len(X_data)):
        model = AR(X_data[i])
        model_fit = model.fit()
        window = model_fit.k_ar
        window = 12
        #print(window)
        coef = model_fit.params

        # walk forward over time steps in test
        history = X_data[i][len(X_data[i])-window:]
        history = [history[i] for i in range(len(history))]
        predictions = list()
        for t in range(len(y_data[i])):
            length = len(history)
            lag = [history[i] for i in range(length-window,length)]
            yhat = coef[0]
            
            for d in range(window):
                yhat += coef[d+1] * lag[window-d-1]
            
            predictions.append(yhat)
            
            
        #print(predictions)
        predictions = np.asarray(predictions)
        tot_predictions.append(predictions)
    return tot_predictions


dict2data = {}
dict2MAPE = {}
dict2RMSE = {}
counter = 0
step_in, step_out = 2680, 12

for i, k in gbc:

    cell = i
    print(counter)
    counter +=1
    #if counter > 3: break
    cell_data_i = k['nr_people'].values
    predicted = get_AR_prediction(cell_data_i, step_in, step_out)
    
    X_data, y_data = split_sequence(cell_data_i, step_in, step_out)
    
    difference = abs(predicted - y_data)
    
    pow_err = np.power(difference, 2)
    rmse = math.sqrt(np.mean(pow_err))
    
    # collect data 2 dictionary
    minimum = np.amin(difference)   
    per75 = np.percentile(difference, 75)
    per50 = np.percentile(difference, 50)
    per25 = np.percentile(difference, 25)
    maximum = np.amax(difference)
    l5i = [minimum, per25, per50, per75, maximum]
    dict2data[cell] = l5i
    dict2RMSE[cell] = rmse
    MAPE = np.mean(abs(100 * (difference/y_data)))
    dict2MAPE[cell] = MAPE
    print(rmse, MAPE, per50)
    
with open('error_data_4_Autoregression_12_StepForecast.csv', 'w') as f:
    for key, value in dict2data.items():
        f.write('%s:%s\n' % (key, value))    
        
with open('MAPE_data_4_Autoregression_12_StepForecast.csv', 'w') as f:
    for key, value in dict2MAPE.items():
        f.write('%s:%s\n' % (key, value))   
        
with open('RMSE_data_4_Autoregression_12_StepForecast.csv', 'w') as f:
    for key, value in dict2RMSE.items():
        f.write('%s:%s\n' % (key, value))     