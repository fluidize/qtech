import time
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
import technical_analysis as ta
import smc_analysis as smc

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
                print(f"[blue]USING CACHED DATA FROM {file_modified_time}[/blue] ({os.path.getsize(cache_file)/(1024**2):.2f} MB)")
                
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
        
        progress_bar = tqdm(total=chunks, desc="KUCOIN PROGRESS", ascii="#>")
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
                raise Exception(f"Error fetching {ticker} from Kucoin. Check request parameters. {request}")
            
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
            print(f"[blue]Data cached to {cache_file}[/blue] ({os.path.getsize(cache_file)/(1024**2):.2f} MB)")
            
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

def prepare_data_classifier(data, lagged_length=5, extra_features=False, elapsed_time=False):
    start_time = time.time()
    df = data.copy()
    section_times = {}
    if 'Datetime' in df.columns:
        df.drop(columns=['Datetime'], inplace=True)
    indicators = {}
    
    # ===== PRICE ACTION INDICATORS =====
    section_start = time.time()
    # Basic price relationships
    indicators['Log_Return'] = ta.log_return(df['Close'])
    indicators['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    indicators['Close_Open_Range'] = (df['Close'] - df['Open']) / df['Open']
    section_times['Price Action'] = time.time() - section_start
    
    # ===== TREND INDICATORS =====
    section_start = time.time()
    # Simple Moving Averages
    # indicators['SMA5'] = ta.sma(df['Close'], timeperiod=5) / df['Close']
    # indicators['SMA10'] = ta.sma(df['Close'], timeperiod=10) / df['Close'] 
    # indicators['SMA20'] = ta.sma(df['Close'], timeperiod=20) / df['Close']
    # indicators['SMA50'] = ta.sma(df['Close'], timeperiod=50) / df['Close']
    # indicators['SMA100'] = ta.sma(df['Close'], timeperiod=100) / df['Close']
    
    # # Exponential Moving Averages
    # indicators['EMA5'] = ta.ema(df['Close'], timeperiod=5) / df['Close']
    # indicators['EMA10'] = ta.ema(df['Close'], timeperiod=10) / df['Close']
    # indicators['EMA20'] = ta.ema(df['Close'], timeperiod=20) / df['Close']
    # indicators['EMA50'] = ta.ema(df['Close'], timeperiod=50) / df['Close']
    
    # # Advanced Moving Averages
    # indicators['DEMA10'] = ta.dema(df['Close'], timeperiod=10) / df['Close']
    # indicators['DEMA20'] = ta.dema(df['Close'], timeperiod=20) / df['Close']
    # indicators['TEMA10'] = ta.tema(df['Close'], timeperiod=10) / df['Close']
    # indicators['TEMA20'] = ta.tema(df['Close'], timeperiod=20) / df['Close']
    # indicators['HMA10'] = ta.hma(df['Close'], timeperiod=10) / df['Close']
    # indicators['WMA10'] = ta.wma(df['Close'], timeperiod=10) / df['Close']
    # indicators['KAMA10'] = ta.kama(df['Close']) / df['Close']
    
    # # Moving Average Crossovers
    # indicators['SMA5_SMA10'] = indicators['SMA5'] - indicators['SMA10']
    # indicators['SMA10_SMA20'] = indicators['SMA10'] - indicators['SMA20']
    # indicators['SMA20_SMA50'] = indicators['SMA20'] - indicators['SMA50']
    # indicators['EMA10_EMA20'] = indicators['EMA10'] - indicators['EMA20']
    section_times['Moving Averages'] = time.time() - section_start
    
    section_start = time.time()
    indicators['MACD'], indicators['MACD_Signal'] = ta.macd(df['Close'])
    indicators['MACD_Hist'] = indicators['MACD'] - indicators['MACD_Signal']
    
    indicators['PPO'], indicators['PPO_Signal'], indicators['PPO_Hist'] = ta.ppo(df['Close'])
    
    indicators['ADX'], indicators['PLUS_DI'], indicators['MINUS_DI'] = ta.adx(df['High'], df['Low'], df['Close'])
    indicators['DI_Diff'] = indicators['PLUS_DI'] - indicators['MINUS_DI']
    
    indicators['AROON_UP'], indicators['AROON_DOWN'] = ta.aroon(df['High'], df['Low'])
    indicators['AROON_OSC'] = indicators['AROON_UP'] - indicators['AROON_DOWN']
    
    indicators['AO'] = ta.awesome_oscillator(df['High'], df['Low'])
    indicators['DPO'] = ta.dpo(df['Close'], timeperiod=20) / df['Close']
    section_times['Trend Indicators'] = time.time() - section_start
    
    # ===== MOMENTUM INDICATORS =====
    section_start = time.time()
    indicators['MOM5'] = ta.mom(df['Close'], timeperiod=5) / df['Close']
    indicators['MOM10'] = ta.mom(df['Close'], timeperiod=10) / df['Close']
    
    indicators['ROC5'] = ta.roc(df['Close'], timeperiod=5)
    indicators['ROC10'] = ta.roc(df['Close'], timeperiod=10)
    
    indicators['RSI7'] = ta.rsi(df['Close'], timeperiod=7)
    indicators['RSI14'] = ta.rsi(df['Close'], timeperiod=14)
    indicators['RSI21'] = ta.rsi(df['Close'], timeperiod=21)
    
    indicators['STOCH_K'], indicators['STOCH_D'] = ta.stoch(df['High'], df['Low'], df['Close'])
    indicators['STOCH_K_D'] = indicators['STOCH_K'] - indicators['STOCH_D']
    
    indicators['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
    indicators['WillR'] = ta.willr(df['High'], df['Low'], df['Close'])
    indicators['TSI'], indicators['TSI_Signal'] = ta.tsi(df['Close'])
    indicators['RVI'] = ta.rvi(df['Open'], df['High'], df['Low'], df['Close'])
    section_times['Momentum Indicators'] = time.time() - section_start
    
    # ===== VOLATILITY INDICATORS =====
    section_start = time.time()
    indicators['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    indicators['ATR_Pct'] = indicators['ATR'] / df['Close'] * 100
    
    indicators['BB_Upper'], indicators['BB_Middle'], indicators['BB_Lower'] = ta.bbands(df['Close'])
    indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / indicators['BB_Middle']
    indicators['BB_Pos'] = (df['Close'] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
    
    indicators['KC_Upper'], indicators['KC_Middle'], indicators['KC_Lower'] = ta.keltner_channels(df['High'], df['Low'], df['Close'])
    indicators['KC_Width'] = (indicators['KC_Upper'] - indicators['KC_Lower']) / indicators['KC_Middle']
    indicators['KC_Pos'] = (df['Close'] - indicators['KC_Lower']) / (indicators['KC_Upper'] - indicators['KC_Lower'])
    
    indicators['CHOP'] = ta.choppiness_index(df['High'], df['Low'], df['Close'])
    indicators['HIST_VOL'] = ta.historical_volatility(df['Close'])
    indicators['Volatility_Ratio'] = ta.volatility_ratio(df['High'], df['Low'], df['Close'])
    section_times['Volatility Indicators'] = time.time() - section_start
    
    # ===== VOLUME INDICATORS =====
    section_start = time.time()
    if 'Volume' in df.columns:
        indicators['OBV'] = ta.obv(df['Close'], df['Volume'])
        # indicators['OBV_ROC'] = ta.roc(indicators['OBV'], timeperiod=10)
        
        indicators['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['PVT'] = ta.pvt(df['Close'], df['Volume'])
        indicators['VZO'] = ta.volume_zone_oscillator(df['Close'], df['Volume'])
        
        indicators['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume']) / df['Close']
        indicators['VWAP_Upper'], _, indicators['VWAP_Lower'] = ta.vwap_bands(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['VWAP_Upper'] = indicators['VWAP_Upper'] / df['Close']
        indicators['VWAP_Lower'] = indicators['VWAP_Lower'] / df['Close']
    section_times['Volume Indicators'] = time.time() - section_start
    
    # ===== SUPPORT & RESISTANCE =====
    section_start = time.time()
    indicators['DC_Upper'], indicators['DC_Middle'], indicators['DC_Lower'] = ta.donchian_channel(df['High'], df['Low'])
    indicators['DC_Width'] = (indicators['DC_Upper'] - indicators['DC_Lower']) / indicators['DC_Middle']
    
    indicators['SuperTrend'], indicators['SuperTrend_Line'] = ta.supertrend(df['High'], df['Low'], df['Close'])
    indicators['SuperTrend_Diff'] = (df['Close'] - indicators['SuperTrend_Line']) / df['Close']
    
    try:
        indicators['PSAR'] = ta.psar(df['High'].values, df['Low'].values)
        indicators['PSAR_Diff'] = (df['Close'] - indicators['PSAR']) / df['Close']
    except:
        pass
    section_times['Support & Resistance'] = time.time() - section_start
    
    # ===== PRICE PATTERNS =====
    section_start = time.time()
    indicators['Ichimoku_Tenkan'], indicators['Ichimoku_Kijun'], indicators['Ichimoku_Senkou_A'], indicators['Ichimoku_Senkou_B'], _ = ta.ichimoku(df['High'], df['Low'], df['Close'])
    indicators['Cloud_Diff'] = indicators['Ichimoku_Senkou_A'] - indicators['Ichimoku_Senkou_B']
    
    indicators['Bull_Power'], indicators['Bear_Power'] = ta.elder_ray(df['High'], df['Low'], df['Close'])
    
    if extra_features:
        try:
            indicators['Fractal_Up'], indicators['Fractal_Down'] = ta.fractal_indicator(df['High'], df['Low'])
        except:
            pass
    section_times['Price Patterns'] = time.time() - section_start
    
    # ===== STATISTICAL INDICATORS =====
    section_start = time.time()
    indicators['Z_Score10'] = ta.z_score(df['Close'], timeperiod=10)
    indicators['Z_Score20'] = ta.z_score(df['Close'], timeperiod=20)
    
    # Fisher Transform
    indicators['Fisher10'] = ta.fisher_transform(df['Close'], timeperiod=10)
    
    # ===== CYCLE INDICATORS =====
    try:
        indicators['Price_Cycle20'] = ta.price_cycle(df['Close'], cycle_period=20)
    except:
        pass
    
    indicators['Mass_Index'] = ta.mass_index(df['High'], df['Low'])

    if extra_features:
        try:
            indicators['Hurst'] = ta.hurst_exponent(df['Close'])
        except:
            pass
        try:
            indicators['Percent_Rank'] = ta.percent_rank(df['Close'])
        except:
            pass
    section_times['Statistical & Cycle'] = time.time() - section_start
    
    section_start = time.time()
    lagged_features = {}
    df.drop(columns=['Volume'], inplace=True)
    for col in df.columns:
        for i in range(1, lagged_length):
            lagged_features[f'Prev{i}_{col}'] = df[col].shift(i)
    section_times['Lagged Features'] = time.time() - section_start
    
    section_start = time.time()
    df = pd.concat([df, pd.DataFrame(indicators), pd.DataFrame(lagged_features)], axis=1)
    df.dropna(inplace=True)
    section_times['Data Cleaning'] = time.time() - section_start

    section_start = time.time()
    pivots = smc.pivot_points(data['Open'], data['High'], data['Low'], data['Close'], window=10)
    y = pd.Series(1, index=df.index)  # Default to hold
    for idx, row in pivots.iterrows():
        if row['Type'] == 'Swing_Low' and idx in y.index:
            y.loc[idx] = 2  # Buy
        elif row['Type'] == 'Swing_High' and idx in y.index:
            y.loc[idx] = 0  # Sell
    X = df
    y = y.loc[X.index]  # Align y to X
    section_times['Target Creation'] = time.time() - section_start
    
    end_time = time.time()
    total_time = end_time - start_time
    if elapsed_time:
        print(f"Data preparation done. ({len(X)} rows, {X.shape[1]} features) {total_time:.2f} seconds")
    
    # Sort and print section times
    # sorted_times = sorted(section_times.items(), key=lambda x: x[1], reverse=True)
    # print("\nSection execution times (slowest to fastest):")
    # for section, exec_time in sorted_times:
    #     percentage = (exec_time / total_time) * 100
    #     print(f"{section}: {exec_time:.4f} seconds ({percentage:.1f}%)")
    
    return X, y

def prepare_data_reinforcement(data, lagged_length=5, extra_features=False, elapsed_time=False):
    df = data.copy()
    start_time = time.time()
    if 'Datetime' in df.columns:
        df.drop(columns=['Datetime'], inplace=True)
    
    indicators = {}
    
    indicators['Log_Return'] = ta.log_return(df['Close'])
    indicators['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    indicators['Close_Open_Range'] = (df['Close'] - df['Open']) / df['Open']
    
    indicators['MACD'], indicators['MACD_Signal'] = ta.macd(df['Close'])
    indicators['MACD_Hist'] = indicators['MACD'] - indicators['MACD_Signal']
    
    indicators['PPO'], indicators['PPO_Signal'], indicators['PPO_Hist'] = ta.ppo(df['Close'])
    
    indicators['ADX'], indicators['PLUS_DI'], indicators['MINUS_DI'] = ta.adx(df['High'], df['Low'], df['Close'])
    indicators['DI_Diff'] = indicators['PLUS_DI'] - indicators['MINUS_DI']
    
    indicators['AROON_UP'], indicators['AROON_DOWN'] = ta.aroon(df['High'], df['Low'])
    indicators['AROON_OSC'] = indicators['AROON_UP'] - indicators['AROON_DOWN']
    
    indicators['AO'] = ta.awesome_oscillator(df['High'], df['Low'])
    indicators['DPO'] = ta.dpo(df['Close'], timeperiod=20) / df['Close']
    
    indicators['MOM5'] = ta.mom(df['Close'], timeperiod=5) / df['Close']
    indicators['MOM10'] = ta.mom(df['Close'], timeperiod=10) / df['Close']
    
    indicators['ROC5'] = ta.roc(df['Close'], timeperiod=5)
    indicators['ROC10'] = ta.roc(df['Close'], timeperiod=10)
    
    indicators['RSI7'] = ta.rsi(df['Close'], timeperiod=7)
    indicators['RSI14'] = ta.rsi(df['Close'], timeperiod=14)
    indicators['RSI21'] = ta.rsi(df['Close'], timeperiod=21)
    
    indicators['STOCH_K'], indicators['STOCH_D'] = ta.stoch(df['High'], df['Low'], df['Close'])
    indicators['STOCH_K_D'] = indicators['STOCH_K'] - indicators['STOCH_D']
    
    indicators['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
    indicators['WillR'] = ta.willr(df['High'], df['Low'], df['Close'])
    indicators['TSI'], indicators['TSI_Signal'] = ta.tsi(df['Close'])
    indicators['RVI'] = ta.rvi(df['Open'], df['High'], df['Low'], df['Close'])
    
    indicators['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    indicators['ATR_Pct'] = indicators['ATR'] / df['Close'] * 100
    
    indicators['BB_Upper'], indicators['BB_Middle'], indicators['BB_Lower'] = ta.bbands(df['Close'])
    indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / indicators['BB_Middle']
    indicators['BB_Pos'] = (df['Close'] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
    
    indicators['KC_Upper'], indicators['KC_Middle'], indicators['KC_Lower'] = ta.keltner_channels(df['High'], df['Low'], df['Close'])
    indicators['KC_Width'] = (indicators['KC_Upper'] - indicators['KC_Lower']) / indicators['KC_Middle']
    indicators['KC_Pos'] = (df['Close'] - indicators['KC_Lower']) / (indicators['KC_Upper'] - indicators['KC_Lower'])
    
    indicators['CHOP'] = ta.choppiness_index(df['High'], df['Low'], df['Close'])
    indicators['HIST_VOL'] = ta.historical_volatility(df['Close'])
    indicators['Volatility_Ratio'] = ta.volatility_ratio(df['High'], df['Low'], df['Close'])
    
    if 'Volume' in df.columns:
        indicators['OBV'] = ta.obv(df['Close'], df['Volume'])
        indicators['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['PVT'] = ta.pvt(df['Close'], df['Volume'])
        indicators['VZO'] = ta.volume_zone_oscillator(df['Close'], df['Volume'])
        
        indicators['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume']) / df['Close']
        indicators['VWAP_Upper'], _, indicators['VWAP_Lower'] = ta.vwap_bands(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['VWAP_Upper'] = indicators['VWAP_Upper'] / df['Close']
        indicators['VWAP_Lower'] = indicators['VWAP_Lower'] / df['Close']
    
    indicators['DC_Upper'], indicators['DC_Middle'], indicators['DC_Lower'] = ta.donchian_channel(df['High'], df['Low'])
    indicators['DC_Width'] = (indicators['DC_Upper'] - indicators['DC_Lower']) / indicators['DC_Middle']
    
    indicators['SuperTrend'], indicators['SuperTrend_Line'] = ta.supertrend(df['High'], df['Low'], df['Close'])
    indicators['SuperTrend_Diff'] = (df['Close'] - indicators['SuperTrend_Line']) / df['Close']
    
    try:
        indicators['PSAR'] = ta.psar(df['High'].values, df['Low'].values)
        indicators['PSAR_Diff'] = (df['Close'] - indicators['PSAR']) / df['Close']
    except:
        pass
    
    indicators['Ichimoku_Tenkan'], indicators['Ichimoku_Kijun'], indicators['Ichimoku_Senkou_A'], indicators['Ichimoku_Senkou_B'], _ = ta.ichimoku(df['High'], df['Low'], df['Close'])
    indicators['Cloud_Diff'] = indicators['Ichimoku_Senkou_A'] - indicators['Ichimoku_Senkou_B']
    
    indicators['Bull_Power'], indicators['Bear_Power'] = ta.elder_ray(df['High'], df['Low'], df['Close'])
    
    if extra_features:
        try:
            indicators['Fractal_Up'], indicators['Fractal_Down'] = ta.fractal_indicator(df['High'], df['Low'])
        except:
            pass
    
    indicators['Z_Score10'] = ta.z_score(df['Close'], timeperiod=10)
    indicators['Z_Score20'] = ta.z_score(df['Close'], timeperiod=20)
    
    indicators['Fisher10'] = ta.fisher_transform(df['Close'], timeperiod=10)
    
    try:
        indicators['Price_Cycle20'] = ta.price_cycle(df['Close'], cycle_period=20)
    except:
        pass
    
    indicators['Mass_Index'] = ta.mass_index(df['High'], df['Low'])

    if extra_features:
        try:
            indicators['Hurst'] = ta.hurst_exponent(df['Close'])
        except:
            pass
        try:
            indicators['Percent_Rank'] = ta.percent_rank(df['Close'])
        except:
            pass
    
    lagged_features = {}
    for col in df.columns:
        for i in range(1, lagged_length):
            lagged_features[f'Prev{i}_{col}'] = df[col].shift(i)
    
    indicator_state = pd.concat([df, pd.DataFrame(indicators), pd.DataFrame(lagged_features)], axis=1)
    indicator_state.dropna(inplace=True)

    end_time = time.time()
    total_time = end_time - start_time
    if elapsed_time:
        print(f"Data preparation done. ({len(indicator_state)} rows, {indicator_state.shape[1]} features) {total_time:.2f} seconds")

    return indicator_state

def bad_data_check(df):
    infinite_cols = []
    for col in df.columns:
        if np.isinf(df[col]).sum() > 0:
            infinite_cols.append(col)
    nan_cols = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            nan_cols.append(col)
    print(f"Infinite columns: {infinite_cols}")
    print(f"NaN columns: {nan_cols}")
    print(f"Number of NaN rows: {df.isna().sum().sum()}")
    return infinite_cols, nan_cols

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
    out_df = prepare_data_reinforcement(data, lagged_length=20)
    bad_data_check(out_df)
