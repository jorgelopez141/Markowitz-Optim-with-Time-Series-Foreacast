#%%

from function_filePrep import tickerList, download_data,missing_days_andIndexTimeZone, to_month_and_add_monthYear_columns
import yfinance
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

df_ticker_price = download_data(list_stocks=tickerList,start_date = '2016-01-01', end_date = '2024-09-30')
stock = 'WMT'
df_ticker_price1=missing_days_andIndexTimeZone(df_ticker_price)
monthly_data=to_month_and_add_monthYear_columns(df_ticker_price1)


# selecting stock only - daily
stock_hist_daily = df_ticker_price1[[stock]]

#selecting stock only - monthly 
stock_hist = monthly_data[[stock,'month','year']].copy()

from sklearn.metrics import mean_absolute_error,mean_squared_error

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(stock_hist.head())
#%%

date_split = '2023-12-31'
#months_to_forecast = 12 
stock_train = stock_hist[stock][:date_split]
stock_test = stock_hist[stock][date_split:]
# forecasting 12 months if we can forecast 12 months, othewise the max allowed to forecast
months_to_forecast = 12 if (stock_test.index.max().year-pd.to_datetime(date_split).year)*12 + (stock_test.index.max().month - pd.to_datetime(date_split).month) >= 12 else (stock_test.index.max().year-pd.to_datetime(date_split).year)*12 + (stock_test.index.max().month - pd.to_datetime(date_split).month)

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

#%%

######### CREANDO CUTOFF POINTS Y TESTING 12 MONTHS INTO THE FUTURE 
from dateutil.relativedelta import relativedelta
cutoff_points = ['2023-09-30','2023-03-31','2022-09-30','2022-03-31']

df_allEntries = pd.DataFrame({
    'Date': [],
    stock: [],
    'Pred': [],
    'CutoffDate': []
})

def pred_and_error(stock,cutoff_date,months_forecast):
    date_split = pd.to_datetime(cutoff_date)
    nextMonth = date_split + relativedelta(months=1)
    month12 = date_split + relativedelta(months=months_forecast)
    #months_to_forecast = 12 
    stock_train = stock_hist[stock][:date_split]
    stock_test = stock_hist[stock][nextMonth:month12]
    # forecasting 12 months if we can forecast 12 months, othewise the max allowed to forecast
    model = ExponentialSmoothing(stock_train, trend='mul', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=months_forecast) 
    
    mini_df=stock_test.reset_index().merge(forecast.reset_index().rename(columns={'index':'Date',0:'Pred'}), on='Date')
    #df_allEntries=pd.concat([df_allEntries,mini_df])
    mini_df['CutoffDate'] = cutoff_date
    return mini_df

#%%
for cutoff_point in cutoff_points: 
    df_allEntries=pd.concat([df_allEntries,pred_and_error('AMZN',cutoff_point,12)])

#%%



df_allEntries['monthDiff']=df_allEntries.apply(lambda x: relativedelta( x['Date'], pd.to_datetime(x['CutoffDate'])).years *12 + relativedelta( x['Date'], pd.to_datetime(x['CutoffDate'])).months ,axis=1)

#%%

MAPE_calc = df_allEntries.groupby('monthDiff').apply(lambda x: mean_absolute_percentage_error(x[stock],x['Pred']))

mse_calc=df_allEntries.groupby('monthDiff').apply(lambda x: np.sqrt( mean_squared_error(x[stock],x['Pred']) ))

mae_calc=df_allEntries.groupby('monthDiff').apply(lambda x: mean_absolute_error(x[stock],x['Pred']) )

#%%
MAPE_calc.reset_index().rename(columns={0:'MAPE'}).merge(mse_calc.reset_index().rename(columns={0:'MSE'}), on='monthDiff').merge(mae_calc.reset_index().rename(columns={0:'MAE'}),on='monthDiff')

#%%

#### CREATING FUNCTION TO CREATE THAT DATAFRAME FROM THE TOP 
cutoff_points = ['2023-09-30','2023-03-31','2022-09-30','2022-03-31']
def Error_Rate_Horizon(stockToSelect: str, cutoff_points: list[str] = cutoff_points, horizon:int=12 ):    
    stock_hist = monthly_data[[stockToSelect,'month','year']].dropna().copy()
    
    df_allEntries = pd.DataFrame({
    'Date': [],
    stock: [],
    'Pred': [],
    'CutoffDate': []
    })

        
    def pred_and_error(stock,cutoff_date,months_forecast):
        date_split = pd.to_datetime(cutoff_date)
        nextMonth = date_split + relativedelta(months=1)
        month12 = date_split + relativedelta(months=months_forecast)
        #months_to_forecast = 12 
        stock_train = stock_hist[stock][:date_split]
        stock_test = stock_hist[stock][nextMonth:month12]
        # forecasting 12 months if we can forecast 12 months, othewise the max allowed to forecast
        model = ExponentialSmoothing(stock_train, trend='mul', seasonal='add', seasonal_periods=12)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=months_forecast) 
        
        mini_df=stock_test.reset_index().merge(forecast.reset_index().rename(columns={'index':'Date',0:'Pred'}), on='Date')
        #df_allEntries=pd.concat([df_allEntries,mini_df])
        mini_df['CutoffDate'] = cutoff_date
        return mini_df

    for cutoff_point in cutoff_points: 
        df_allEntries=pd.concat([df_allEntries,pred_and_error(stockToSelect,cutoff_point,horizon)])

    df_allEntries['monthDiff']=df_allEntries.apply(lambda x: relativedelta( x['Date'], pd.to_datetime(x['CutoffDate'])).years *12 + relativedelta( x['Date'], pd.to_datetime(x['CutoffDate'])).months ,axis=1)


    MAPE_calc = df_allEntries.groupby('monthDiff').apply(lambda x: mean_absolute_percentage_error(x[stockToSelect],x['Pred']))

    mse_calc=df_allEntries.groupby('monthDiff').apply(lambda x: np.sqrt( mean_squared_error(x[stockToSelect],x['Pred']) ))

    mae_calc=df_allEntries.groupby('monthDiff').apply(lambda x: mean_absolute_error(x[stockToSelect],x['Pred']) )

    return MAPE_calc.reset_index().rename(columns={0:'MAPE'}).merge(mse_calc.reset_index().rename(columns={0:'MSE'}), on='monthDiff').merge(mae_calc.reset_index().rename(columns={0:'MAE'}),on='monthDiff')

#%%
cutoff_points = ['2023-09-30','2023-03-31','2022-09-30','2022-03-31']
Error_Rate_Horizon('NKE')