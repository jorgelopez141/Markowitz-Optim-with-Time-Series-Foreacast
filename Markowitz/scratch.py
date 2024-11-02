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
df.columns = ['ds', 'y']  # Prophet requires these specific column names

# Split data into training and testing sets (90/10)
train_size = int(len(df) * 0.9)
train_data = df[:train_size]
test_data = df[train_size:]

# Create and train Prophet model
model = Prophet(yearly_seasonality=True, 
               weekly_seasonality=True,
               daily_seasonality=False,
               changepoint_prior_scale=0.05)
model.fit(train_data)

# Make predictions for the test period
future_dates = model.make_future_dataframe(periods=len(test_data), freq='W')
forecast = model.predict(future_dates)

# Extract predictions for the test period
predictions = forecast.iloc[-len(test_data):]['yhat']

# Calculate error metrics
mape = np.mean(np.abs((test_data['y'] - predictions) / test_data['y'])) * 100
rmse = np.sqrt(np.mean((test_data['y'] - predictions) ** 2))

# Plotting
plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(train_data['ds'], train_data['y'], label='Training Data', color='blue')

# Plot test data and predictions
plt.plot(test_data['ds'], test_data['y'], label='Actual Test Data', color='green')
plt.plot(test_data['ds'], predictions, label='Predictions', color='red', linestyle='--')

plt.title('WMT Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Add error metrics to the plot
plt.text(0.02, 0.98, f'MAPE: {mape:.2f}%\nRMSE: ${rmse:.2f}', 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True)
plt.tight_layout()
plt.show()

# Print error metrics
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")