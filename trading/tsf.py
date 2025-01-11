import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fetch stock data (e.g., Apple stock)
ticker = 'AAPL'
data = pd.DataFrame(yf.download(ticker, start='2018-01-01', end='2023-01-01'))

# Create a function to add lagged features to handle time series dependencies
def create_lagged_features(df, lags):
    for lag in range(1, lags+1):
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    return df

# Apply lagged features (we'll use a 5-lag model for simplicity)
data = create_lagged_features(data, 5)

# Calculate the daily returns (percentage change)
data['Return'] = data['Close'].pct_change()

# Shift the return by 1 to predict the next day's return (target variable)
data['Target'] = data['Return'].shift(-1)

# Drop rows with missing values (due to shifting)
data.dropna(inplace=True)

# Features and target variable for machine learning
X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']]
y = data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize RandomForestClassifier for time series prediction
model = MLPRegressor()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
print(type(y_pred))
print(type(y_test))
compressed_y_test = pd.DataFrame([y_test.tolist()],columns=y_test.index)
def score(true, prediction):
    #closer to 0 the better
    score = 0
    count = 0
    for x in range(len(true)):
        score += abs(true - prediction)
        count += 1
    return score/count
accuracy = score(compressed_y_test, y_pred)
print(accuracy)

# Visualizing actual vs predicted returns (for testing)
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual Returns', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Returns', color='red', alpha=0.7)
plt.legend(loc='best')
plt.title(f'{ticker} - Actual vs Predicted Returns')
plt.show()

# Buy/Sell recommendations based on predicted returns
buy_signals = y_pred > 0
sell_signals = y_pred < 0

print(f"Number of Buy Signals: {buy_signals.sum()}")
print(f"Number of Sell Signals: {sell_signals.sum()}")


