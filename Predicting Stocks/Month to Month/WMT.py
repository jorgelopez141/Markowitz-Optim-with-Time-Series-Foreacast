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
printGraphs = 0 # 1 to show graphs

# selecting stock only - daily
stock_hist_daily = df_ticker_price1[[stock]]

#selecting stock only - monthly 
stock_hist = monthly_data[[stock,'month','year']].copy()

if printGraphs ==1:
    stock_hist[stock].plot()
    plt.title(stock)
    plt.show()

    sns.boxplot(data=stock_hist,x='month',y=stock)
    plt.title(f'{stock} Seasonality per Month')
    plt.show()

    print('''
        We can notice WMTs seasonality a little higher than usual in months 4,5. 
        This could be due to tax returns happening every year.
        ''')


    # Creating the line plot
    sns.lineplot(data=stock_hist, x='month', y=stock, hue='year', palette='tab10')
    plt.title('Historical Prices broken by Year')
    plt.show()

    #seasonal decompose 
    result_season_decompose=seasonal_decompose(stock_hist[stock], model='additive',period=12)
    result_season_decompose.plot()
    plt.show()

    print("*"*100)

    print("""
        From breaking WMT's price into trend, and seasonal components, 
        it can be appreciated that there is seasonality happening every year.
        """)


##

