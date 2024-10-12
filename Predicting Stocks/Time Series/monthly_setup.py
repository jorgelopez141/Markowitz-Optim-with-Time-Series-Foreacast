#%%
from function_filePrep import tickerList, download_data,missing_days_andIndexTimeZone
import yfinance
import pandas as pd 

df_ticker_price = download_data(list_stocks=tickerList,start_date = '2018-01-01', end_date = '2024-09-30')

df_ticker_price1=missing_days_andIndexTimeZone(df_ticker_price)

monthly_data=df_ticker_price1.resample(rule='M').first()


