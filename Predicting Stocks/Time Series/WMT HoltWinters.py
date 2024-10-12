#%%
from WMT import stock_hist
import pandas as pd 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print(stock_hist.head())
