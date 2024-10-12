#%%
# tutorials: https://github.com/ourownstory/neural_prophet/blob/main/docs/source/quickstart.ipynb
# docs https://github.com/ourownstory/neural_prophet/blob/main/docs/source/tutorials/tutorial01.ipynb
# main web: https://neuralprophet.com/
from neuralprophet import NeuralProphet

from WMT import stock_hist,stock_hist_daily # stock hist is the monthly one
import pandas as pd 
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

from function_filePrep import _add_dayMonthYear 

# add month, year, day of week, day of year...
stock_hist_daily1=_add_dayMonthYear(stock_hist_daily)

#%%


stock_hist_daily2=stock_hist_daily1.reset_index().rename(columns={'Date':'ds',stock:'y'}).copy()
date_train_test = '2024-07-31'
stock_train2=stock_hist_daily2.loc[stock_hist_daily2.ds<=date_train_test].copy()
stock_test2=stock_hist_daily2.loc[stock_hist_daily2.ds>date_train_test].copy()

## COVID
# Python

# Define the lockdown periods

# Create a date range from March 21, 2022 through June 10, 2021
date_range = pd.date_range(start='2020-03-21', end='2021-06-10', freq='D')

# Create DataFrame
lockdowns = pd.DataFrame({
    'event': 'COVID',
    'ds': date_range
})



#%%
lockdowns['ds'] = pd.to_datetime(lockdowns['ds'])

#%%


#%%
#changepoint range dice que encuentra cambios en trend hasta el 90% inicial de la data.
#m = NeuralProphet(holidays=lockdowns, changepoint_range=0.9, changepoint_prior_scale=0.2) 
m = NeuralProphet(changepoints_range=0.9,
                  trend_reg=0.2
                  )
# changepoint prior scale es que tan flexible es. valores entre 0.05 y 0.5
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.add_events('COVID')
#m.add_events(lockdowns)
m.add_future_regressor('day')
m.add_future_regressor('day_of_week')
m.add_future_regressor('day_of_year')
m.add_future_regressor('week_of_year')
m.add_future_regressor('month')
m.add_future_regressor('quarter')
m.add_future_regressor('year')

#%%
# Fit the model
metrics = m.fit(m.create_df_with_events(stock_train2, lockdowns), freq='D')

forecast = m.predict(m.create_df_with_events(stock_test2, lockdowns))

#%%

forecast.head()
m.plot(forecast)

#%%
# EVALUATE WITH ERROR METRICS 
mse=np.sqrt(mean_squared_error(y_true=forecast['y'],y_pred=forecast['yhat1']))

mae = mean_absolute_error(y_true=forecast['y'],
                   y_pred=forecast['yhat1'])

mape = mean_absolute_percentage_error(y_true=forecast['y'],
                   y_pred=forecast['yhat1'])

print(mse,mae,mape)


#%%
df_train, df_val = m.split_df(m.create_df_with_events(stock_train2, lockdowns), valid_p=0.2)


#%%

metrics = m.fit(df_train, validation_df=df_val, progress=None)
metrics
#%%
forecast = m.predict(m.create_df_with_events(stock_test2, lockdowns))
#m.plot(forecast)
#%%
forecast[['y','yhat1']].plot()