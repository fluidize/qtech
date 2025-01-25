import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

ticker = 'AAPL'
data = pd.DataFrame(yf.download(ticker, start='2018-01-01', end='2023-01-01', progress=False))

def score(true, prediction):
    #closer to 0 the better
    score = 0
    count = 0
    for x in range(len(true)):
        score += abs(true[x] - prediction[x]) #add deviation
        count += 1
    return score/count
def create_lagged_features(df, lags):
    for lag in range(1, lags+1):
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    return df

data = create_lagged_features(data, 5)
data['Return'] = data['Close'].pct_change()
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

accuracy = score(compressed_y_test, compressed_y_pred)
print(accuracy)