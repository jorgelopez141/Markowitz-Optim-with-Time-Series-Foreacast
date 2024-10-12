# importing libraries
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from arch import arch_model
import yfinance
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings("ignore")
sns.set_theme()

# making current file location's the working directory
import os
import sys

# Gets the directory of the running script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Sets it as the current working directory
os.chdir(script_dir)

print(f"The current working directory is: {os.getcwd()}")

# 

#Excel file con nombres de los stocks
df_tickers=pd.read_excel('./marketInsider.xlsx',sheet_name= 'automatically') #current 30 stocks of DJI
tickerList=df_tickers.ticker.to_list()
tickerList=[x.strip() for x in tickerList]


def download_data(list_stocks=tickerList,start_date:str = '2018-01-01', end_date:str = '2024-09-30'): 

    #Download stock data
    data = yfinance.download(list_stocks, start=start_date, end=end_date,auto_adjust=True,group_by='ticker')

    # getting close prices only 
    for ticker in list_stocks:
        nameColumn = ticker.lower() + ' price'
        data[nameColumn] = data[ticker].Close
        del data[ticker]

    #standard column names
    data.columns = [x[0].split(' ')[0].upper() for x in data.columns]

    return data 

def missing_days_andIndexTimeZone(fn_df:pd.DataFrame):
    fn_df=fn_df.asfreq('b')
    fn_df=fn_df.fillna(method='ffill')
    fn_df.index = fn_df.index.tz_localize(None)
    return fn_df

def _add_dayMonthYear(fn_df1):
    fn_df=fn_df1.copy()
    fn_df['day']=fn_df.index.day
    fn_df['day_of_week']=fn_df.index.day_of_week
    fn_df['day_of_year']=fn_df.index.day_of_year
    fn_df['week_of_year']=fn_df.index.isocalendar()['week'].to_list()
    fn_df['month']=fn_df.index.month  
    fn_df['quarter']=fn_df.index.quarter
    fn_df['year']=fn_df.index.year
    return fn_df


def to_month_and_add_monthYear_columns(fn_df):
    monthly_data=fn_df.resample(rule='M').first()
    monthly_data['month']=monthly_data.index.month  
    monthly_data['year']=monthly_data.index.year
    return monthly_data