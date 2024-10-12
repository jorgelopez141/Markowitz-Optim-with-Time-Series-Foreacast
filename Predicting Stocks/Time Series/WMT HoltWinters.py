#%%
from sklearn.metrics import mean_absolute_error,mean_squared_error
from WMT import stock_hist
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from WMT import stock_hist,stock_hist_daily # stock hist is the monthly one
stock = 'WMT'
# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(stock_hist.head())
#%%

date_split = '2023-06-30'
#months_to_forecast = 12 
stock_train = stock_hist[stock][:date_split]
stock_test = stock_hist[stock][date_split:]
# forecasting 12 months if we can forecast 12 months, othewise the max allowed to forecast
months_to_forecast = 15#12 if (stock_test.index.max().year-pd.to_datetime(date_split).year)*12 + (stock_test.index.max().month - pd.to_datetime(date_split).month) >= 12 else (stock_test.index.max().year-pd.to_datetime(date_split).year)*12 + (stock_test.index.max().month - pd.to_datetime(date_split).month)

model = ExponentialSmoothing(stock_train, trend='mul', seasonal='add', seasonal_periods=12)
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=months_to_forecast)

#%%

# EVALUATE WITH ERROR METRICS 
mse=np.sqrt(mean_squared_error(y_true=stock_test.loc[stock_test.index.isin(forecast.index)],y_pred=forecast))

mae = mean_absolute_error(y_true=stock_test.loc[stock_test.index.isin(forecast.index)],y_pred=forecast)

mape = mean_absolute_percentage_error(y_true=stock_test.loc[stock_test.index.isin(forecast.index)],y_pred=forecast)

print(mse,mae, mape)

#%%

fig, ax = plt.subplots(figsize=(10, 5))
#ax.scatter(stock_test2.ds, stock_test2['y'], color='r', label= 'Actual Test Values')
forecast.plot(label='forecast',ax=ax,color='b')
fig = stock_test[1:13].plot(label='actual',ax=ax,color='y')
stock_train.plot(label='past values',ax=ax,color='r')
# ax.set_xbound(lower=pd.to_datetime('2024-07-01'),
#               upper=pd.to_datetime('2024-09-30'))
# ax.set_ylim(55, 90)
plt.xticks(rotation=45)
ax.legend(loc='upper left') 
plot = plt.suptitle('Actual vs Forecasted')

