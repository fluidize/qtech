import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fetch stock data (e.g., Apple stock)
ticker = 'AAPL'
data = pd.DataFrame(yf.download(ticker, start='2018-01-01', end='2023-01-01', progress=False))
# Create a function to add lagged features to handle time series dependencies
def create_lagged_features(df, lags):
    for lag in range(1, lags+1):
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    return df

# Apply lagged features (we'll use a 5-lag model for simplicity)
data = create_lagged_features(data, 5)


append_list = []
previous_price = pd.NA
for close_price in data['Close']:
    append_list.append((close_price-previous_price)/close_price)
    previous_price = close_price #pct change
print(append_list)
data['Return'] = append_list

data['Target'] = data['Return'].shift(-1)

data.dropna(inplace=True)

X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = MLPRegressor(activation='logistic', hidden_layer_sizes=(50,50))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


compressed_y_pred = y_pred.tolist()
compressed_y_test = y_test.tolist()

def score(true, prediction):
    #closer to 0 the better
    score = 0
    count = 0
    for x in range(len(true)):
        score += abs(true[x] - prediction[x])
        print(score) #add deviation
        count += 1
    return score/count
accuracy = score(compressed_y_test, compressed_y_pred)
print(f"accuracy: {accuracy} avg prediction: {sum(y_pred)/len(y_pred)}")

# Visualizing actual vs predicted returns (for testing)
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual Returns', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Returns', color='red', alpha=0.7)
print(compressed_y_pred[0:10])
print(compressed_y_test[0:10])
plt.legend(loc='best')
plt.title(f'{ticker} - Actual vs Predicted Returns')
plt.show()

# Buy/Sell recommendations based on predicted returns
# buy_signals = y_pred > 0
# sell_signals = y_pred < 0

# print(f"Number of Buy Signals: {buy_signals.sum()}")
# print(f"Number of Sell Signals: {sell_signals.sum()}")


