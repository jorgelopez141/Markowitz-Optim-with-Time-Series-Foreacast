#%%
import pandas as pd
import numpy as np
import yfinance as yf

# Step 1: Get stock data
tickers = ['AAPL', 'WMT', 'MSFT']
data = yf.download(tickers, start='2023-10-01', end='2024-10-01')['Adj Close']

#%%

(data.loc[data.index == data.index.max()].reset_index(drop=True)/data.loc[data.index == data.index.min()].reset_index(drop=True)).dot(weights)
#%%

# Step 2: Calculate daily returns
returns = data.pct_change().dropna()

# Step 3: Define portfolio weights
weights = np.array([0.2, 0.5, 0.3])

# Step 4: Calculate portfolio returns
portfolio_returns = returns.dot(weights)

# Step 5: Annualize returns and volatility
## notar que multilica por 252 para anualizar
annual_returns = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)

# Step 6: Define risk-free rate
risk_free_rate = 0.02

# Step 7: Calculate Sharpe Ratio
sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

# Step 8: Print Sharpe Ratio
print(f'The Sharpe Ratio of the portfolio is: {sharpe_ratio:.2f}')
