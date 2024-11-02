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

# Define COVID period as a holiday
covid_holidays = pd.DataFrame({
    'holiday': 'covid_period',
    'ds': pd.date_range(start='2020-03-11', end='2020-12-14'),  # WHO declaration to first US vaccine
    'lower_window': 0,
    'upper_window': 0
})

# Split data into training and testing sets (90/10)
train_size = int(len(df) * 0.9)
train_data = df[:train_size]
test_data = df[train_size:]

# Create and train regular Prophet model
model_regular = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=True,
                       daily_seasonality=False,
                       changepoint_prior_scale=0.05)
model_regular.fit(train_data)

# Create and train Prophet model with COVID as holiday
model_with_covid = Prophet(yearly_seasonality=True, 
                          weekly_seasonality=True,
                          daily_seasonality=False,
                          changepoint_prior_scale=0.05,
                          holidays=covid_holidays)
model_with_covid.fit(train_data)

# Make predictions for both models
future_dates = model_regular.make_future_dataframe(periods=len(test_data), freq='W')
forecast_regular = model_regular.predict(future_dates)
forecast_with_covid = model_with_covid.predict(future_dates)

# Extract predictions for the test period
predictions_regular = forecast_regular.iloc[-len(test_data):]['yhat']
predictions_with_covid = forecast_with_covid.iloc[-len(test_data):]['yhat']

# Calculate error metrics for both models
def calculate_metrics(actual, predicted):
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mape, rmse

mape_regular, rmse_regular = calculate_metrics(test_data['y'], predictions_regular)
mape_covid, rmse_covid = calculate_metrics(test_data['y'], predictions_with_covid)

# Plotting
plt.figure(figsize=(15, 8))

# Plot training data
plt.plot(train_data['ds'], train_data['y'], label='Training Data', color='blue', alpha=0.5)

# Plot test data and predictions
plt.plot(test_data['ds'], test_data['y'], label='Actual Test Data', color='green', linewidth=2)
plt.plot(test_data['ds'], predictions_regular, label='Regular Model', color='red', linestyle='--')
plt.plot(test_data['ds'], predictions_with_covid, label='Model with COVID Holiday', color='purple', linestyle='--')

# Highlight COVID period
plt.axvspan(pd.to_datetime('2020-03-11'), pd.to_datetime('2020-12-14'), 
            color='gray', alpha=0.2, label='COVID Period')

plt.title('WMT Stock Price Prediction - Model Comparison')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Add error metrics to the plot
metrics_text = (f'Regular Model:\n'
               f'MAPE: {mape_regular:.2f}%\n'
               f'RMSE: ${rmse_regular:.2f}\n\n'
               f'Model with COVID Holiday:\n'
               f'MAPE: {mape_covid:.2f}%\n'
               f'RMSE: ${rmse_covid:.2f}')

plt.text(0.02, 0.98, metrics_text, 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True)
plt.tight_layout()

# Plot the holiday effect
holiday_effect = model_with_covid.plot_components(forecast_with_covid)
plt.show()

# Print model comparison
print("\nModel Performance Comparison:")
print("\nRegular Model:")
print(f"Mean Absolute Percentage Error (MAPE): {mape_regular:.2f}%")
print(f"Root Mean Square Error (RMSE): ${rmse_regular:.2f}")
print("\nModel with COVID Holiday:")
print(f"Mean Absolute Percentage Error (MAPE): {mape_covid:.2f}%")
print(f"Root Mean Square Error (RMSE): ${rmse_covid:.2f}")

# Analyze holiday impact
covid_effect = forecast_with_covid[forecast_with_covid['covid_period'] != 0]['covid_period']
if not covid_effect.empty:
    print("\nCOVID Period Impact Analysis:")
    print(f"Average effect on stock price: ${covid_effect.mean():.2f}")
    print(f"Maximum effect: ${covid_effect.max():.2f}")
    print(f"Minimum effect: ${covid_effect.min():.2f}")