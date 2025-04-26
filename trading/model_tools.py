from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from datetime import datetime, timedelta
import requests
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import numpy as np
import hashlib
import os
import tempfile
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rich import print
import techincal_analysis as ta

def fetch_data(ticker, chunks, interval, age_days, kucoin: bool = True, use_cache: bool = True, cache_expiry_hours: int = 24):
    print("[yellow]FETCHING DATA[/yellow]")
    
    cache_key = f"{ticker}_{chunks}_{interval}_{age_days}_{kucoin}"
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    cache_file = os.path.join(tempfile.gettempdir(), f"market_data_{cache_hash}.parquet")
    
    if use_cache and os.path.exists(cache_file):
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        file_age_hours = (datetime.now() - file_modified_time).total_seconds() / 3600
        
        if file_age_hours < cache_expiry_hours:
            try:
                cached_data = pd.read_parquet(cache_file)
                print(f"[blue]USING CACHED DATA FROM {file_modified_time}[/blue]")
                
                with open(f"{cache_file}.json", "w") as f:
                    json.dump({
                        "ticker": ticker,
                        "chunks": chunks,
                        "interval": interval,
                        "age_days": age_days,
                        "kucoin": kucoin,
                        "cached_time": str(file_modified_time),
                        "rows": len(cached_data),
                        "accessed_time": str(datetime.now())
                    }, f, indent=2)
                
                return cached_data
            except Exception as e:
                print(f"[yellow]Cache read error: {e}. Fetching fresh data...[/yellow]")
        else:
            print(f"[yellow]Cache expired ({file_age_hours:.1f} hours old). Fetching fresh data...[/yellow]")
    
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
    
    if use_cache:
        try:
            data.to_parquet(cache_file)
            print(f"[blue]Data cached to {cache_file}[/blue]")
            
            with open(f"{cache_file}.json", "w") as f:
                json.dump({
                    "ticker": ticker,
                    "chunks": chunks,
                    "interval": interval,
                    "age_days": age_days,
                    "kucoin": kucoin,
                    "cached_time": str(datetime.now()),
                    "rows": len(data)
                }, f, indent=2)
                
        except Exception as e:
            print(f"[yellow]Failed to cache data: {e}[/yellow]")
        
    return data

def prepare_data(data, lagged_length=5, train_split=True, scale_y=True):
    scalers = {
        'price': MinMaxScaler(feature_range=(0, 1)),
        'volume': QuantileTransformer(output_distribution='normal'),
        'technical': StandardScaler()
    }

    df = data.copy()

    df['Log_Return'] = ta.log_returns(df['Close'])
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    df['MACD'], df['MACD_Signal'] = ta.macd(df['Close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.bbands(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    
    df['STOCH_K'], df['STOCH_D'] = ta.stoch(df['High'], df['Low'], df['Close'])
    
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    df['ROC'] = ta.roc(df['Close'])
    
    df['WillR'] = ta.willr(df['High'], df['Low'], df['Close'])
    
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
    
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = ta.adx(df['High'], df['Low'], df['Close'])
    
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    
    lagged_features = []
    for col in df.columns:
        for i in range(1, lagged_length):
            lagged_features.append(pd.DataFrame({
                f'Prev{i}_{col}': df[col].shift(i)
            }))
    
    if lagged_features:
        df = pd.concat([df] + lagged_features, axis=1)
    
    df['Close_ZScore'] = (df['Close'] - ta.sma(df['Close'], timeperiod=100)) / ta.stddev(df['Close'], timeperiod=100)
    df['MA10'] = ta.sma(df['Close'], timeperiod=10) / df['Close']
    df['MA20'] = ta.sma(df['Close'], timeperiod=20) / df['Close']
    df['MA50'] = ta.sma(df['Close'], timeperiod=50) / df['Close']
    df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
    df['RSI'] = ta.rsi(df['Close'], timeperiod=14)
    
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

def prepare_data_classifier(data, lagged_length=5):
    scalers = {
        'price': MinMaxScaler(feature_range=(0, 1)),
        'volume': QuantileTransformer(output_distribution='normal'),
        'technical': StandardScaler()
    }

    df = data.copy()
    
    if 'Datetime' in df.columns:
        df.drop(columns=['Datetime'], inplace=True)

    indicators = {}
    
    indicators['Log_Return'] = ta.log_returns(df['Close'])
    indicators['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    macd, signal = ta.macd(df['Close'])
    indicators['MACD'] = macd
    indicators['MACD_Signal'] = signal
    indicators['MACD_Hist'] = macd - signal

    upper, middle, lower = ta.bbands(df['Close'])
    indicators['BB_Upper'] = upper
    indicators['BB_Middle'] = middle
    indicators['BB_Lower'] = lower
    indicators['BB_Width'] = (upper - lower) / middle
    
    indicators['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    k, d = ta.stoch(df['High'], df['Low'], df['Close'])
    indicators['STOCH_K'] = k
    indicators['STOCH_D'] = d
    indicators['OBV'] = ta.obv(df['Close'], df['Volume'])
    indicators['ROC'] = ta.roc(df['Close'])
    indicators['WillR'] = ta.willr(df['High'], df['Low'], df['Close'])
    indicators['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
    adx, plus_di, minus_di = ta.adx(df['High'], df['Low'], df['Close'])
    indicators['ADX'] = adx
    indicators['PLUS_DI'] = plus_di
    indicators['MINUS_DI'] = minus_di
    indicators['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    
    patterns = ['Doji', 'Hammer', 'Shooting_Star', 'Engulfing', 'Harami', 
                         'Morning_Star', 'Evening_Star', 'Three_White_Soldiers',
                         'Three_Black_Crows', 'Dark_Cloud_Cover', 'Piercing_Line']
    pattern_df = ta.get_candlestick_patterns(df, patterns)
    for pattern in patterns:
        indicators[pattern] = pattern_df[pattern]
    
    indicators['Close_ZScore'] = (df['Close'] - ta.sma(df['Close'], timeperiod=100)) / ta.stddev(df['Close'], timeperiod=100)
    indicators['MA10'] = ta.sma(df['Close'], timeperiod=10) / df['Close']
    indicators['MA20'] = ta.sma(df['Close'], timeperiod=20) / df['Close']
    indicators['MA50'] = ta.sma(df['Close'], timeperiod=50) / df['Close']
    indicators['MA10_MA20_Cross'] = indicators['MA10'] - indicators['MA20']
    indicators['RSI'] = ta.rsi(df['Close'], timeperiod=14)
    
    lagged_features = {}
    for col in df.columns:
        for i in range(1, lagged_length):
            lagged_features[f'Prev{i}_{col}'] = df[col].shift(i)
    
    df = pd.concat([df, pd.DataFrame(indicators), pd.DataFrame(lagged_features)], axis=1)
    
    df = df.bfill().ffill()
    df.dropna(inplace=True)

    price_features = ['Open', 'High', 'Low', 'Close'] + [f'Prev{i}_{col}' for i in range(1, lagged_length) for col in ['Open', 'High', 'Low', 'Close']]
    volume_features = ['Volume', 'OBV'] + [f'Prev{i}_Volume' for i in range(1, lagged_length)]
    bounded_features = ['RSI', 'STOCH_K', 'STOCH_D', 'MFI', 'WillR']
    pattern_features = patterns
    
    technical_features = [col for col in df.columns 
                        if col not in (price_features + volume_features + bounded_features + 
                                        pattern_features + ['Datetime'])]
    
    # df[price_features] = scalers['price'].fit_transform(df[price_features])
    
    df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
    df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
    df[volume_features] = scalers['volume'].fit_transform(df[volume_features])
    
    if technical_features:
        df[technical_features] = scalers['technical'].fit_transform(df[technical_features])

    pct_change = df['Close'].pct_change()
    y = pd.Series(0, index=df.index)
    y[pct_change > 0] = 1
    y[pct_change < 0] = 0

    X = df[:-1]
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

