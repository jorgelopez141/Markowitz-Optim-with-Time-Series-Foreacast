#%%
from scipy.optimize import minimize
import pandas as pd
import numpy as np

# Example stock prices DataFrame
dates = pd.date_range('2023-01-01', '2024-01-01')
np.random.seed(42)
prices = pd.DataFrame(np.random.rand(len(dates), 5) * 100, index=dates, columns=['Stock1', 'Stock2', 'Stock3', 'Stock4', 'Stock5'])
returns = prices.pct_change().dropna()

#%%
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean()) * 252

def portfolio_risk(weights, returns):
    cov_matrix = returns.cov() * 252
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def sharpe_ratio(weights, returns, risk_free_rate=0):
    port_return = portfolio_return(weights, returns)
    port_risk = portfolio_risk(weights, returns)
    return - (port_return - risk_free_rate) / port_risk

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1) for _ in range(returns.shape[1])]


#%%
initial_guess = [1. / returns.shape[1]] * returns.shape[1]

result = minimize(sharpe_ratio, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimized Weights:", result.x)
print("Maximized Sharpe Ratio:", -result.fun)

