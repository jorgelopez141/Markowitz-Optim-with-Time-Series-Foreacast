# when pandemic started: https://www.npr.org/2021/03/11/975663437/march-11-2020-the-day-everything-changed: March 11, 2020
# when pandemic lockdown ended: pfize vaccine was administered for first time Dec 14, 2020: https://abcnews.go.com/US/us-administer-1st-doses-pfizer-coronavirus-vaccine/story?id=74703018
# 

#%%
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Download WMT data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years ago
wmt = yf.download('WMT', start=start_date, end=end_date, interval='1wk')

# Prepare data for Prophet
df = wmt.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Create dataset excluding COVID lockdown period
covid_start = '2020-03-01'
covid_end = '2020-12-31'
df_no_covid = df[~((df['ds'] >= covid_start) & (df['ds'] <= covid_end))].reset_index(drop=True)

# Split both datasets into training and testing sets (90/10)
train_size = int(len(df) * 0.9)
train_size_no_covid = int(len(df_no_covid) * 0.9)

# Full dataset split
train_data = df[:train_size]
test_data = df[train_size:]

# No-COVID dataset split
train_data_no_covid = df_no_covid[:train_size_no_covid]
test_data_no_covid = df_no_covid[train_size_no_covid:]

# Create and train Prophet model on full dataset
model_full = Prophet(yearly_seasonality=True, 
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05)
model_full.fit(train_data)

# Create and train Prophet model on no-COVID dataset
model_no_covid = Prophet(yearly_seasonality=True, 
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05)
model_no_covid.fit(train_data_no_covid)

# Make predictions for both models
future_dates_full = model_full.make_future_dataframe(periods=len(test_data), freq='W')
forecast_full = model_full.predict(future_dates_full)

future_dates_no_covid = model_no_covid.make_future_dataframe(periods=len(test_data), freq='W')
forecast_no_covid = model_no_covid.predict(future_dates_no_covid)

# Extract predictions for the test period
predictions_full = forecast_full.iloc[-len(test_data):]['yhat']
predictions_no_covid = forecast_no_covid.iloc[-len(test_data):]['yhat']

# Calculate error metrics for both models
def calculate_metrics(actual, predicted):
    mape = np.mean(np.abs((actual.values - predicted.values) / actual)) * 100
    rmse = np.sqrt(np.mean((actual.values - predicted.values) ** 2))
    return mape, rmse

mape_full, rmse_full = calculate_metrics(test_data['y'], predictions_full)
mape_no_covid, rmse_no_covid = calculate_metrics(test_data['y'], predictions_no_covid)


# Plotting
plt.figure(figsize=(15, 8))

# Plot training data
plt.plot(train_data['ds'], train_data['y'], label='Training Data', color='blue', alpha=0.5)

# Plot test data and predictions
plt.plot(test_data['ds'], test_data['y'], label='Actual Test Data', color='green', linewidth=2)
plt.plot(test_data['ds'], predictions_full, label='Predictions (Full Data)', color='red', linestyle='--')
plt.plot(test_data['ds'], predictions_no_covid, label='Predictions (Excluding COVID)', color='purple', linestyle='--')

# Highlight COVID period
plt.axvspan(pd.to_datetime(covid_start), pd.to_datetime(covid_end), 
            color='gray', alpha=0.2, label='COVID Lockdown Period')

plt.title('WMT Stock Price Prediction - Model Comparison')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Add error metrics to the plot
metrics_text = (f'Full Data Model:\n'
               f'MAPE: {mape_full:.2f}%\n'
               f'RMSE: ${rmse_full:.2f}\n\n'
               f'No-COVID Model:\n'
               f'MAPE: {mape_no_covid:.2f}%\n'
               f'RMSE: ${rmse_no_covid:.2f}')

plt.text(0.02, 0.98, metrics_text, 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True)
plt.tight_layout()
plt.show()

# Print model comparison
print("\nModel Performance Comparison:")
print("\nFull Data Model:")
print(f"Mean Absolute Percentage Error (MAPE): {mape_full:.2f}%")
print(f"Root Mean Square Error (RMSE): ${rmse_full:.2f}")
print("\nNo-COVID Model:")
print(f"Mean Absolute Percentage Error (MAPE): {mape_no_covid:.2f}%")
print(f"Root Mean Square Error (RMSE): ${rmse_no_covid:.2f}")