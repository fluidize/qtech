from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from datetime import datetime, timedelta
import requests
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rich import print
import pandas_indicators as ta

def fetch_data(ticker, chunks, interval, age_days, kucoin: bool = True):
    print("[green]DOWNLOADING DATA[/green]")
    if not kucoin:
        data = pd.DataFrame()
        times = []
        for x in range(chunks):
            chunksize = 1
            start_date = datetime.now() - timedelta(days=chunksize) - timedelta(days=chunksize*x) - timedelta(days=age_days)
            end_date = datetime.now() - timedelta(days=chunksize*x) - timedelta(days=age_days)
            temp_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)
            if data.empty:
                data = temp_data
            else:
                data = pd.concat([data, temp_data])
            times.append(start_date)
            times.append(end_date)
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"\n{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")

        data.sort_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
        data = pd.DataFrame(data)  
    elif kucoin:
        data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        times = []
        
        progress_bar = tqdm(total=chunks, desc="KUCOIN PROGRESS")
        for x in range(chunks):
            chunksize = 1440  # 1d of 1m data
            end_time = datetime.now() - timedelta(minutes=chunksize*x) - timedelta(days=age_days)
            start_time = end_time - timedelta(minutes=chunksize) - timedelta(days=age_days)
            
            params = {
                "symbol": ticker,
                "type": interval,
                "startAt": str(int(start_time.timestamp())),
                "endAt": str(int(end_time.timestamp()))
            }
            
            request = requests.get("https://api.kucoin.com/api/v1/market/candles", params=params).json()
            try:
                request_data = request["data"]  # list of lists
            except:
                raise Exception(f"Error fetching Kucoin. Check request parameters.")
            
            records = []
            for dochltv in request_data:
                records.append({
                    "Datetime": dochltv[0],
                    "Open": float(dochltv[1]),
                    "Close": float(dochltv[2]),
                    "High": float(dochltv[3]),
                    "Low": float(dochltv[4]),
                    "Volume": float(dochltv[6])
                })
            
            temp_data = pd.DataFrame(records)
            if data.empty:
                data = temp_data
            else:
                data = pd.concat([data, temp_data])
            times.append(start_time)
            times.append(end_time)

            progress_bar.update(1)
        progress_bar.close()
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds | {data.shape[0]} bars")
        
        data["Datetime"] = pd.to_datetime(pd.to_numeric(data['Datetime']), unit='s')
        data.sort_values('Datetime', inplace=True)
        data.reset_index(drop=True, inplace=True)
        
    return data

def prepare_data(data, lagged_length=5, train_split=True, scale_y=True):
    scalers = {
        'price': MinMaxScaler(feature_range=(0, 1)),
        'volume': QuantileTransformer(output_distribution='normal'),
        'technical': StandardScaler()
    }

    df = data.copy()

    # Calculate technical indicators using pandas_indicators
    df['Log_Return'] = ta.LOG(df['Close'])
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    df['MACD'], df['MACD_Signal'] = ta.MACD(df['Close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'])
    
    df['STOCH_K'], df['STOCH_D'] = ta.STOCH(df['High'], df['Low'], df['Close'])
    
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])
    
    df['ROC'] = ta.ROC(df['Close'])
    
    df['WillR'] = ta.WILLR(df['High'], df['Low'], df['Close'])
    
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'])
    
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = ta.ADX(df['High'], df['Low'], df['Close'])
    
    df['MFI'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Create lagged features
    lagged_features = []
    for col in df.columns:
        for i in range(1, lagged_length):
            lagged_features.append(pd.DataFrame({
                f'Prev{i}_{col}': df[col].shift(i)
            }))
    
    if lagged_features:
        df = pd.concat([df] + lagged_features, axis=1)
    
    df['Close_ZScore'] = (df['Close'] - ta.SMA(df['Close'], timeperiod=100)) / ta.STDDEV(df['Close'], timeperiod=100)
    df['MA10'] = ta.SMA(df['Close'], timeperiod=10) / df['Close']
    df['MA20'] = ta.SMA(df['Close'], timeperiod=20) / df['Close']
    df['MA50'] = ta.SMA(df['Close'], timeperiod=50) / df['Close']
    df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    
    df = df.bfill().ffill()
    df.dropna(inplace=True)

    if train_split:
        price_features = ['Open', 'High', 'Low', 'Close'] + [f'Prev{i}_{col}' for i in range(1, lagged_length) for col in ['Open', 'High', 'Low', 'Close']]
        volume_features = ['Volume', 'OBV'] + [f'Prev{i}_Volume' for i in range(1, lagged_length)]
        bounded_features = ['RSI', 'STOCH_K', 'STOCH_D', 'MFI', 'WillR']  # Features that are already bounded
        
        technical_features = [col for col in df.columns 
                            if col not in (price_features + volume_features + bounded_features + ['Datetime'])]
        
        if scale_y:
            df[price_features] = scalers['price'].fit_transform(df[price_features])
        else:
            df[price_features] = df[price_features]
        
        df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
        df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
        df[volume_features] = scalers['volume'].fit_transform(df[volume_features])
        
        if technical_features:
            df[technical_features] = scalers['technical'].fit_transform(df[technical_features])

        if 'Datetime' in df.columns:
            X = df.drop(['Datetime'], axis=1)
        else:
            X = df
        
        y = df['Close'].shift(-1)
        
        X = X[:-1]
        y = y[:-1]
        return X, y, scalers
    
    return df, scalers

def prepare_data_classifier(data, lagged_length=5, train_split=True, pct_threshold=0.05):
    pct_threshold = pct_threshold / 100

    scalers = {
        'price': MinMaxScaler(feature_range=(0, 1)),
        'volume': QuantileTransformer(output_distribution='normal'),
        'technical': StandardScaler()
    }

    df = data.copy()
    
    if 'Datetime' in df.columns:
        df.drop(columns=['Datetime'], inplace=True)

    df['Log_Return'] = ta.LOG(df['Close'])
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    macd, signal = ta.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = macd - signal
    
    upper, middle, lower = ta.BBANDS(df['Close'])
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    df['BB_Width'] = (upper - lower) / middle
    
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'])
    
    k, d = ta.STOCH(df['High'], df['Low'], df['Close'])
    df['STOCH_K'] = k
    df['STOCH_D'] = d
    
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])
    
    df['ROC'] = ta.ROC(df['Close'])
    
    df['WillR'] = ta.WILLR(df['High'], df['Low'], df['Close'])
    
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'])
    
    adx, plus_di, minus_di = ta.ADX(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx
    df['PLUS_DI'] = plus_di
    df['MINUS_DI'] = minus_di
    
    df['MFI'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
    
    patterns = ['doji', 'hammer', 'engulfing', 'morning_star', 'evening_star']
    pattern_df = ta.get_candlestick_patterns(df, patterns)
    for pattern in patterns:
        df[pattern] = pattern_df[pattern]
    
    lagged_features = []
    for col in df.columns:
        for i in range(1, lagged_length):
            lagged_features.append(pd.DataFrame({
                f'Prev{i}_{col}': df[col].shift(i)
            }))
    
    if lagged_features:
        df = pd.concat([df] + lagged_features, axis=1)
    
    df['Close_ZScore'] = (df['Close'] - ta.SMA(df['Close'], timeperiod=100)) / ta.STDDEV(df['Close'], timeperiod=100)
    df['MA10'] = ta.SMA(df['Close'], timeperiod=10) / df['Close']
    df['MA20'] = ta.SMA(df['Close'], timeperiod=20) / df['Close']
    df['MA50'] = ta.SMA(df['Close'], timeperiod=50) / df['Close']
    df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    
    df = df.bfill().ffill()
    df.dropna(inplace=True)
    

    price_features = ['Open', 'High', 'Low', 'Close'] + [f'Prev{i}_{col}' for i in range(1, lagged_length) for col in ['Open', 'High', 'Low', 'Close']]
    volume_features = ['Volume', 'OBV'] + [f'Prev{i}_Volume' for i in range(1, lagged_length)]
    bounded_features = ['RSI', 'STOCH_K', 'STOCH_D', 'MFI', 'WillR']  # Features that are already bounded
    pattern_features = patterns
    
    technical_features = [col for col in df.columns 
                        if col not in (price_features + volume_features + bounded_features + 
                                        pattern_features + ['Datetime'])]
    
    # df[price_features] = scalers['price'].fit_transform(df[price_features])
    # df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
    # df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
    # df[volume_features] = scalers['volume'].fit_transform(df[volume_features])
    
    # if technical_features:
    #     df[technical_features] = df[technical_features].replace([np.inf, -np.inf], np.nan)
    #     df[technical_features] = df[technical_features].fillna(df[technical_features].mean())
    #     df[technical_features] = scalers['technical'].fit_transform(df[technical_features])
        
    X = df
    
    pct_change = df['Close'].pct_change().shift(-1)

    # 0 = below threshold | 1 = within threshold | 2 = above threshold
    y = pd.Series(1, index=df.index)
    y[pct_change < -pct_threshold] = 0
    y[pct_change > pct_threshold] = 2
    
    X = X[:-1]
    y = y[:-1]

    return X, y

def prediction_plot(actual, predicted):
    difference = len(actual)-len(predicted)
    actual = actual[difference:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual['Datetime'], y=actual['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=actual['Datetime'], y=predicted, mode='lines', name='Predicted'))
    fig.update_layout(title='Price Prediction', xaxis_title='Date', yaxis_title='Price')
    
    return fig

def loss_plot(loss_history):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss'), row=1, col=1)
    delta_loss = np.diff(loss_history)
    fig.add_trace(go.Scatter(x=list(range(len(delta_loss))), y=delta_loss, mode='lines', name='Delta Loss'), row=2, col=1)
    fig.update_layout(title='Loss', xaxis_title='Epoch', yaxis_title='Loss')

    return fig

if __name__ == "__main__":
    data = fetch_data("BTC-USDT", 1, "1min", 0, kucoin=True)
    
    main_data, _ = prepare_data_classifier(data, lagged_length=5, train_split=True, pct_threshold=0.1)
    print(main_data)

