#%%
from function_filePrep import tickerList, download_data,missing_days_andIndexTimeZone, to_month_and_add_monthYear_columns
import yfinance
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dateutil.relativedelta import relativedelta
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.stattools import adfuller
# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


df_ticker_price = download_data(list_stocks=tickerList,start_date = '2016-01-01', end_date = '2024-09-30')
df_ticker_price1=missing_days_andIndexTimeZone(df_ticker_price)
monthly_data=to_month_and_add_monthYear_columns(df_ticker_price1)

#%%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

#%%


adfuller( monthly_data['WMT'][:"2023-06"].diff().dropna())[1]<0.05 # if true then is stationary

model_basic=ARIMA(monthly_data['WMT'][:"2023-06"],order=(2,1,2)).fit()

#%%
modelS=SARIMAX(monthly_data['WMT'][:"2023-06"],order=(6,1,2),seasonal_order=(3,1,1,12)).fit()

#%%

sgt.plot_acf(modelS.resid,zero=False)

#%%
sgt.plot_acf(model_basic.resid,zero=False)
#%%
modelS.summary()

#%%
fig, ax = plt.subplots(figsize=(10, 5))
monthly_data['WMT']["2023-06":"2024-06"].plot(label='real')
modelS.predict(start='2023-06-30', end='2024-06-30').plot(label='forecast')
ax.legend(loc='upper left')

#%%
fig, ax = plt.subplots(figsize=(10, 5))
monthly_data['WMT']["2023-06":"2024-06"].plot(label='real')
model_basic.predict(start='2023-06-30', end='2024-06-30').plot(label='forecast')
ax.legend(loc='upper left')

#%%
from pmdarima.arima import auto_arima
model_auto = auto_arima(monthly_data['WMT'][:"2023-06"],
                        #exogenous = df_train1[['Covid']],
                        m=12, #seasonal cycle length
                        max_order=None, #max sum of p and q for ARMA
                        max_p = 6,
                        max_q=6,
                        max_d = 1, # max differentiation 
                        # seasonal orderes are P Q D 
                        max_P = 3, 
                        max_Q= 1, 
                        max_D= 1,
                        maxiter=300,
                        alpha=.05,
                        #trend='ct', #ct= constant + trend
                        n_jobs=-1,
                        information_criterion='oob', # out of bag
                        out_of_sample_size= int(.1*len(monthly_data[:"2023-06"])) # use the last 20% for validation
                        )

#%%



fig, ax = plt.subplots(figsize=(10, 5))
monthly_data['WMT']["2023-07":"2024-06"].plot(label='real',ax=ax)
pd.DataFrame(model_auto.predict(12), index=monthly_data["2023-07":"2024-06"].index).plot(label='forecast',ax=ax)
ax.legend(loc='upper left')

#%%
for p in range(1,7):
    for q in range(1,7):
        try:
            modelS=SARIMAX(monthly_data['WMT'][:"2023-06"],order=(p,1,q),seasonal_order=(3,1,1,12)).fit()            
            sgt.plot_acf(modelS.resid,zero=False)
            plt.title(f'p = {p} and q = {q}')
            plt.show()
        except: 
            continue

#%%

#modelS=SARIMAX(monthly_data['WMT'][:"2023-06"],order=(2,1,2),seasonal_order=(2,0,1,12)).fit()

#modelS=SARIMAX(monthly_data['MSFT'][:"2023-06"],order=(2,1,2),seasonal_order=(2,0,1,12)).fit()

#modelS=SARIMAX(monthly_data['CAT'][:"2023-06"],order=(2,1,2),seasonal_order=(2,0,1,12)).fit()

# modelS=SARIMAX(monthly_data['AXP'][:"2023-06"],order=(2,1,2),seasonal_order=(2,0,1,12)).fit()

modelS=SARIMAX(monthly_data['MRK'][:"2023-06"],order=(2,1,2),seasonal_order=(2,0,1,12)).fit()
#%%

sgt.plot_acf(modelS.resid,zero=False)
plt.title('abc')
plt.show()

#%%
fig, ax = plt.subplots(figsize=(10, 5))
monthly_data['MRK']["2023-06":"2024-06"].plot(label='real')
modelS.predict(start='2023-06-30', end='2024-06-30').plot(label='forecast')
ax.legend(loc='upper left')

#%%

monthly_data['MRK'].plot()