#%%
# code source: https://www.kaggle.com/code/robikscube/time-series-forecasting-with-prophet-yt
# documentation: https://facebook.github.io/prophet/docs/quick_start.html#python-api
# paper in reference 1: https://peerj.com/preprints/3190.pdf
## IMPORTING LIBRARIES
from WMT import stock_hist,stock_hist_daily # stock hist is the monthly one
import pandas as pd 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
stock = 'WMT'


# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# split in X,y
def split_features_target(fn_df):
    X = fn_df[['day','day_of_week','day_of_year','week_of_year',
               'month','quarter','year']]
    y = fn_df.iloc[:,0] #first column is stock price
    return X,y 

print(stock_hist.head())

#%%
### DAILY STOCKS
from function_filePrep import _add_dayMonthYear 

# add month, year, day of week, day of year...
stock_hist_daily1=_add_dayMonthYear(stock_hist_daily)

X, y = split_features_target(stock_hist_daily1)
features_and_target = pd.concat([X,y],axis=1)

#%%

#%%
# ADDING EXOGENOUS VARIABLES 
# Python

stock_hist_daily2=stock_hist_daily1.reset_index().rename(columns={'Date':'ds',stock:'y'}).copy()
date_train_test = '2024-08-31'
stock_train2=stock_hist_daily2.loc[stock_hist_daily2.ds<=date_train_test].copy()
stock_test2=stock_hist_daily2.loc[stock_hist_daily2.ds>date_train_test].copy()

## COVID
# Python
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'}
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
lockdowns
## END COVID 

m = Prophet(holidays=lockdowns, changepoint_range=0.9, changepoint_prior_scale=0.2) #changepoint range dice que encuentra cambios en trend hasta el 90% inicial de la data.
# changepoint prior scale es que tan flexible es. valores entre 0.05 y 0.5
m.add_regressor('day')
m.add_regressor('day_of_week')
m.add_regressor('day_of_year')
m.add_regressor('week_of_year')
m.add_regressor('month')
m.add_regressor('quarter')
m.add_regressor('year')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(stock_train2)


forecast = m.predict(stock_test2)

#%%
from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


#%%


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(stock_test2.ds, stock_test2['y'], color='r', label= 'Actual Test Values')
fig = m.plot(forecast, ax=ax)
ax.set_xbound(lower=pd.to_datetime('2024-07-01'),
              upper=pd.to_datetime('2024-09-30'))
# ax.set_ylim(55, 90)
plt.xticks(rotation=45)
ax.legend(loc='lower left') 
plot = plt.suptitle('Actual vs Forecasted')

#%%
# EVALUATE WITH ERROR METRICS 
mse=np.sqrt(mean_squared_error(y_true=stock_test2['y'],y_pred=forecast['yhat']))

mae = mean_absolute_error(y_true=stock_test2['y'],
                   y_pred=forecast['yhat'])

mape = mean_absolute_percentage_error(y_true=stock_test2['y'],
                   y_pred=forecast['yhat'])

print(mse,mae, mape)
print("For the month of April (the predicted one), the error rate is very low")
#%%
# PRINTING COMPONENTS OF MODEL
m.plot_components(forecast)
plt.show()

#%%

# HACIENDOLO BASADO EN EL RETORNO DE LA ACCION 


stock_hist_daily2=stock_hist_daily1.reset_index().rename(columns={'Date':'ds',stock:'y'}).copy()
stock_hist_daily2['y']=stock_hist_daily2.y.pct_change()*100
stock_hist_daily2.dropna(inplace=True)

stock_train2=stock_hist_daily2.loc[stock_hist_daily2.ds<='2024-03-31'].copy()
stock_test2=stock_hist_daily2.loc[stock_hist_daily2.ds>'2024-03-31'].copy()

m = Prophet(growth='logistic', changepoint_range=0.95)
m.add_regressor('day')
m.add_regressor('day_of_week')
m.add_regressor('day_of_year')
m.add_regressor('week_of_year')
m.add_regressor('month')
m.add_regressor('quarter')
m.add_regressor('year')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(stock_train2)


forecast = m.predict(stock_test2)

#%%


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stock_test2.ds, stock_test2['y'], color='r', label= 'Actual Test Values')
fig = m.plot(forecast, ax=ax)
ax.plot(stock_train2.ds, stock_train2['y'], color='b', label= 'Actual Train Values')
ax.set_xbound(lower=pd.to_datetime('2023-10-01'),
              upper=pd.to_datetime('2024-09-30'))
ax.set_ylim(-10, 10)
plt.xticks(rotation=45)
ax.legend(loc='upper left') 
plot = plt.suptitle('Actual vs Forecasted')

#%%
# EVALUATE WITH ERROR METRICS 
mse=np.sqrt(mean_squared_error(y_true=stock_test2.loc[stock_test2.ds<='2024-04-30']['y'],y_pred=forecast['yhat'].loc[forecast.ds<='2024-04-30']))

mae = mean_absolute_error(y_true=stock_test2.loc[stock_test2.ds<='2024-04-30']['y'],
                   y_pred=forecast['yhat'].loc[forecast.ds<='2024-04-30'])

mape = mean_absolute_percentage_error(y_true=stock_test2['y'].loc[stock_test2.ds<='2024-04-30'],
                   y_pred=forecast['yhat'].loc[forecast.ds<='2024-04-30'])

print(mse,mae, mape)

#%%


#%%

stock_test2['pred'] = forecast.yhat.values

#%%

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stock_test2.ds, stock_test2['y'], color='r', label= 'Actual Test Values')
fig = m.plot(forecast, ax=ax)
ax.plot(stock_train2.ds, stock_train2['y'], color='b', label= 'Actual Train Values')
ax.set_xbound(lower=pd.to_datetime('2024-04-01'),
              upper=pd.to_datetime('2024-04-30'))
ax.set_ylim(-1.5, 1.5)
plt.xticks(rotation=45)
ax.legend(loc='upper left') 
plot = plt.suptitle('Actual vs Forecasted')

#%%

stock_test2[['y','pred']][:30].plot()

#%%
stock_test2[['y','pred']].apply(lambda x: np.std(x))

#%%

#%%
# MONTHLY DATA
monthly_data=stock_hist.reset_index().rename(columns={'Date': 'ds',stock:'y'})
monthly_train=monthly_data.loc[monthly_data.ds<='2024-03-31'].copy()
monthly_test=monthly_data.loc[monthly_data.ds>'2024-03-31'].copy()

#from dask.distributed import Client
#from dask import delayed, compute
#client = Client()  # connect to the cluster
#from joblib import Parallel, delayed
#from multiprocessing import Pool
## COVID
# Python
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'}
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days


param_grid = {  
    'changepoint_prior_scale': [0.001,0.005, 0.01, 0.05, 0.1,0.5],
    'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'changepoint_range': [0.8,0.9,0.95]    
}
import itertools
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here
## END COVID 
cutoffs = pd.to_datetime(['2023-08-31','2023-07-31','2023-06-30'])


def fit_and_evaluate(params):
    m = Prophet(**params, holidays=lockdowns)
    m.add_regressor('month')
    m.add_regressor('year')
    m.fit(monthly_train)
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='180 days')
    df_p = performance_metrics(df_cv, rolling_window=1, metrics=['rmse', 'mape'])
    return df_p['rmse'].values[0]

# List to hold delayed tasks
tasks = [fit_and_evaluate(params) for params in all_params]

# Execute tasks and gather results
rmses = tasks
# Use cross validation to evaluate all parameters
# Run in parallel using Joblib

# Run in parallel using Multiprocessing




#%%

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
tuning_results.sort_values('rmse').head(1)

#%%


best_results = tuning_results.sort_values('rmse').head(1)

param_grid = {  
    'changepoint_prior_scale': best_results['changepoint_prior_scale'].values,
    'seasonality_prior_scale': best_results['seasonality_prior_scale'].values,
    'changepoint_range': best_results['changepoint_range'].values
    
}
import itertools
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

#%%

# Training Best Model

m = Prophet(**all_params[0], holidays=lockdowns)   

#%%

m.add_regressor('month')
m.add_regressor('year')
m.fit(monthly_train)


forecast = m.predict(monthly_test)

#%%

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(monthly_test.ds, monthly_test['y'], color='r', label= 'Actual Test Values')
fig = m.plot(forecast, ax=ax)
# ax.set_xbound(lower=pd.to_datetime('2024-04-01'),
#               upper=pd.to_datetime('2024-04-30'))
# ax.set_ylim(55, 65)
plt.xticks(rotation=45)
ax.legend(loc='upper left') 
plot = plt.suptitle('Actual vs Forecasted')

#%%
# EVALUATE WITH ERROR METRICS 
mse=np.sqrt(mean_squared_error(y_true=monthly_test['y'],y_pred=forecast['yhat']))

mae = mean_absolute_error(y_true=monthly_test['y'],
                   y_pred=forecast['yhat'])

mape = mean_absolute_percentage_error(y_true=monthly_test['y'],
                   y_pred=forecast['yhat'])

print(mse,mae, mape)


#%%
### CROSS VALIDATION ON 30 DAY HORIZON 


# Python
from prophet.diagnostics import cross_validation
cutoffs = pd.to_datetime(['2024-03-31','2023-12-31','2023-09-30','2023-06-30'])
df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='90 days')

#%%

# Python
from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv2)
df_p.head()

#%%
df_p.loc[df_p.horizon.isin(['30 days','60 days','90 days'])]

#%%
from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv2, metric='mape')
plt.title('MAPE at Different Horizons')
plt.show()

#%%

# CROSS VALIDATION FOR HYPERPARAMETER TUNING:
# changepoint_prior_scale: default 0.05. Recommended to try all values betweeen 0.001 and 0.5. this affects how well it adjusts to trend.
    # [0.001,0.005, 0.01, 0.05, 0.1,0.5]
# seasonality_prior_scale: defaults to 10. recommended to try different values in the range 0.01 to 10. 
    # [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
# seasonality_mode: ['additive','multiplicative']
# changepoint_range: defaults to 0.8. This is, only the first 80% of the data is used to understand changes in trends
    # [0.8,0.9,0.95]

stock_hist_daily2=stock_hist_daily1.reset_index().rename(columns={'Date':'ds',stock:'y'}).copy()
date_train_test = '2024-08-31'
stock_train2=stock_hist_daily2.loc[stock_hist_daily2.ds<=date_train_test].copy()
stock_test2=stock_hist_daily2.loc[stock_hist_daily2.ds>date_train_test].copy()

## COVID
# Python
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'}
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

param_grid = {  
    'changepoint_prior_scale': [0.01, 0.05, 0.1,0.2,0.5],
    'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'changepoint_range': [0.8,0.9,0.95]
    
}
import itertools
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here
## END COVID 
cutoffs = pd.to_datetime(['2024-03-31','2023-12-31','2023-09-30','2023-06-30'])

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params, holidays=lockdowns)    
    m.add_regressor('day')
    m.add_regressor('day_of_week')
    m.add_regressor('day_of_year')
    m.add_regressor('week_of_year')
    m.add_regressor('month')
    m.add_regressor('quarter')
    m.add_regressor('year')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(stock_train2)

    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days')
    df_p = performance_metrics(df_cv, rolling_window=1, metrics=['rmse','mape'])
    rmses.append(df_p['rmse'].values[0])

#%%

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)


#%%


stock_hist_daily2=stock_hist_daily1.reset_index().rename(columns={'Date':'ds',stock:'y'}).copy()
date_train_test = '2024-08-31'
stock_train2=stock_hist_daily2.loc[stock_hist_daily2.ds<=date_train_test].copy()
stock_test2=stock_hist_daily2.loc[stock_hist_daily2.ds>date_train_test].copy()

## COVID
# Python
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'}
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days


best_results = tuning_results.sort_values('rmse').head(1)

param_grid = {  
    'changepoint_prior_scale': best_results['changepoint_prior_scale'].values,
    'seasonality_prior_scale': best_results['seasonality_prior_scale'].values,
    'changepoint_range': best_results['changepoint_range'].values
    
}
import itertools
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

#%%

# Training Best Model

m = Prophet(**all_params[0], holidays=lockdowns, seasonality_mode='additive')    
m.add_regressor('day', mode='additive')
m.add_regressor('day_of_week',mode='additive')
m.add_regressor('day_of_year',mode='additive')
m.add_regressor('week_of_year',mode='additive')
m.add_regressor('month',mode='additive')
m.add_regressor('quarter',mode='additive')
m.add_regressor('year',mode='additive')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(stock_train2)

forecast = m.predict(stock_test2)

#%%

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(stock_test2.ds, stock_test2['y'], color='r', label= 'Actual Test Values')
fig = m.plot(forecast, ax=ax)
ax.set_xbound(lower=pd.to_datetime('2024-08-01'),
              upper=pd.to_datetime('2024-09-30'))
ax.set_ylim(70, 90)
plt.xticks(rotation=45)
ax.legend(loc='upper left') 
plot = plt.suptitle('Actual vs Forecasted')

#%%
# EVALUATE WITH ERROR METRICS 
mse=np.sqrt(mean_squared_error(y_true=stock_test2['y'],y_pred=forecast['yhat']))

mae = mean_absolute_error(y_true=stock_test2['y'],
                   y_pred=forecast['yhat'])

mape = mean_absolute_percentage_error(y_true=stock_test2['y'],
                   y_pred=forecast['yhat'])

print(mse,mae, mape)


#%%