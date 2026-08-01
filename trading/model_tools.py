from math import ceil
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import os
import time
from pathlib import Path
import asyncio
import aiohttp

from rich import print
from sklearn.preprocessing import MinMaxScaler

import trading.technical_analysis as ta


def ta_transform(data: pd.DataFrame, add_ticker: str):
    data = ta.heikin_ashi_transform(data)
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
    stoch_k, stoch_d = ta.stoch(high, low, close)

    # Check if add_data columns exist and are not blank
    add_data_cols = [f"add_{add_ticker}_Open", f"add_{add_ticker}_High", f"add_{add_ticker}_Low", f"add_{add_ticker}_Close", f"add_{add_ticker}_Volume"]
    has_add_data = all(col in data.columns for col in add_data_cols) and not data[add_data_cols].isna().all().all()

    if has_add_data:
        add_data = data[add_data_cols].copy()
        add_data.columns = ["Open", "High", "Low", "Close", "Volume"]
        add_data = ta.heikin_ashi_transform(add_data)
        
        add_close = add_data["Close"]
        add_high = add_data["High"]
        add_low = add_data["Low"]
        add_volume = add_data["Volume"]

        add_aroon_up, add_aroon_down = ta.aroon(add_high, add_low, timeperiod=14)
        add_stoch_k, add_stoch_d = ta.stoch(add_high, add_low, add_close)

        result_dict = {
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "log_ret": ta.log_return(close),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "macd_hist": ta.macd_hist(close),
            "aroon_osc": aroon_up - aroon_down,
            "rsi": ta.rsi(close, timeperiod=14),
            "vol_ratio": ta.vol_ratio(volume),
            "atr": ta.atr(high, low, close, timeperiod=14),

            "add_ret_5": add_close.pct_change(5),
            "add_ret_20": add_close.pct_change(20),
            "add_log_ret": ta.log_return(add_close),
            "add_stoch_k": add_stoch_k,
            "add_stoch_d": add_stoch_d,
            "add_macd_hist": ta.macd_hist(add_close),
            "add_aroon_osc": add_aroon_up - add_aroon_down,
            "add_rsi": ta.rsi(add_close, timeperiod=14),
            "add_vol_ratio": ta.vol_ratio(add_volume),
            "add_atr": ta.atr(add_high, add_low, add_close, timeperiod=14),
        }
    else:
        result_dict = {
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "log_ret": ta.log_return(close),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "macd_hist": ta.macd_hist(close),
            "aroon_osc": aroon_up - aroon_down,
            "rsi": ta.rsi(close, timeperiod=14),
            "vol_ratio": ta.vol_ratio(volume),
            "atr": ta.atr(high, low, close, timeperiod=14),
        }

    result = pd.DataFrame(result_dict).dropna(axis=0)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(result)
    result = pd.DataFrame(
        scaled_values,
        columns=result.columns,
        index=result.index,
    )

    # def shannon_entropy(series: pd.Series, bins: int = 50) -> float:
    #     values = series.dropna().to_numpy()
    #     counts, _ = np.histogram(values, bins=bins)
    #     probs = counts[counts > 0] / counts.sum()
    #     return -np.sum(probs * np.log2(probs))

    # def difference_transform(x: pd.Series, y: pd.Series) -> np.ndarray:
    #     return x.sub(y).to_numpy()

    # def acceleration_transform(x: pd.Series) -> np.ndarray:
    #     return x.diff(1).to_numpy()

    # def log_transform(x: pd.Series) -> np.ndarray:
    #     return np.log(x.to_numpy())

    # transforms = [difference_transform, acceleration_transform, log_transform]

    # columns = result.columns

    # for col in columns:
    #     max_entropy = 0
    #     max_entropy_series = None
    #     for transform in transforms:
    #         if transform == difference_transform:
    #             for col2 in columns:
    #                 if col2 != col:
    #                     result[col] = transform(result[col], result[col2])
    #                     if shannon_entropy(result[col]) > max_entropy:
    #                         max_entropy = shannon_entropy(result[col])
    #                         max_entropy_series = result[col]
    #         else:
    #             result[col] = transform(result[col])
    #             if shannon_entropy(result[col]) > max_entropy:
    #                 max_entropy = shannon_entropy(result[col])
    #                 max_entropy_series = result[col]
    #     result[col] = max_entropy_series

    return result


def _fetch_yfinance(symbol, days, interval, age_days):
    end_date = datetime.now() - timedelta(days=age_days)
    start_date = end_date - timedelta(days=days)

    data = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )

    data.sort_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Datetime"}, inplace=True)
    data.rename(columns={"Date": "Datetime"}, inplace=True)
    return pd.DataFrame(data)


def _fetch_birdeye(symbol, days, interval, age_days, retry_limit, proxies, end_time):
    chunks = ceil(days * 1.44)
    headers = {
        "accept": "application/json",
        "x-chain": "solana",
        "X-API-KEY": os.getenv("BIRDEYE_API_KEY"),
    }

    chunk_results = []
    progress_bar = tqdm(total=chunks, desc="BIRDEYE PROGRESS")

    for chunk_index in range(chunks):
        retries = 0
        chunk_end_time = end_time - timedelta(minutes=1000 * chunk_index)
        start_time = chunk_end_time - timedelta(minutes=1000)

        start_ts = int(start_time.timestamp())
        end_ts = int(chunk_end_time.timestamp())
        url = (
            f"https://public-api.birdeye.so/defi/ohlcv?address={symbol}&type={interval}"
            f"&currency=usd&time_from={start_ts}&time_to={end_ts}&ui_amount_mode=raw"
        )

        while retries <= retry_limit:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxies if proxies else None,
                    timeout=30,
                )
                response.raise_for_status()
                request_data = response.json()
                break
            except Exception as e:
                if retries < retry_limit:
                    retries += 1
                    time.sleep(0.5)
                else:
                    print(
                        f"[red]Error fetching {symbol} chunk {chunk_index} after {retry_limit} retries: {e}[/red]"
                    )
                    progress_bar.update(1)
                    continue

        if retries > retry_limit:
            continue

        records = [
            {
                "Datetime": kline["unixTime"],
                "Open": float(kline["o"]),
                "High": float(kline["h"]),
                "Low": float(kline["l"]),
                "Close": float(kline["c"]),
                "Volume": float(kline["v"]),
            }
            for kline in request_data["data"]["items"]
        ]

        chunk_results.append((chunk_index, pd.DataFrame(records)))
        progress_bar.update(1)
        time.sleep(0.1)

    progress_bar.close()
    data = (
        pd.concat(
            [df for _, df in sorted(chunk_results, key=lambda x: x[0])],
            ignore_index=True,
        )
        if chunk_results
        else pd.DataFrame(
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )
    )
    if not data.empty:
        data["Datetime"] = pd.to_datetime(data["Datetime"], unit="s")
        data.sort_values("Datetime", inplace=True)
        data = data.drop_duplicates(subset=["Datetime"], keep="first")
        data.reset_index(drop=True, inplace=True)

        data["HL2"] = (data["High"] + data["Low"]) / 2
        data["OHLC4"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
        data["HLC3"] = (data["High"] + data["Low"] + data["Close"]) / 3
    else:
        print(f"[red]No data retrieved for {symbol}[/red]")

    return data


def _fetch_binance(symbol, days, interval, age_days, retry_limit, proxies, end_time):
    if "min" in interval:
        interval = interval.replace("min", "m")
    elif "hour" in interval:
        interval = interval.replace("hour", "h")
    elif "day" in interval:
        interval = interval.replace("day", "d")
    elif "week" in interval:
        interval = interval.replace("week", "w")
    elif "month" in interval:
        interval = interval.replace("month", "M")

    binance_symbol = symbol.replace("-", "").upper()
    chunks = ceil(days * 1.44)

    async def download_chunk(session, chunk_index, semaphore):
        async with semaphore:
            retries = 0
            chunk_end_time = (
                end_time
                - timedelta(minutes=1000 * chunk_index)
            )
            start_time = chunk_end_time - timedelta(minutes=1000)
            params = {
                "symbol": binance_symbol,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(chunk_end_time.timestamp() * 1000),
                "limit": 1000,
            }
            url = "https://api.binance.com/api/v3/klines"

            while retries <= retry_limit:
                try:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        request_data = await response.json()
                        break
                except Exception as e:
                    if retries < retry_limit:
                        retries += 1
                        await asyncio.sleep(0.5)
                    else:
                        raise Exception(
                            f"Error fetching {binance_symbol} chunk {chunk_index} after {retry_limit} retries: {e}"
                        )

            records = [
                {
                    "Datetime": int(kline[0]) / 1000,
                    "Open": float(kline[1]),
                    "High": float(kline[2]),
                    "Low": float(kline[3]),
                    "Close": float(kline[4]),
                    "Volume": float(kline[5]),
                }
                for kline in request_data
            ]

            return chunk_index, pd.DataFrame(records)

    async def download_all_chunks():
        chunk_results = []
        max_concurrent = min(25, chunks)
        semaphore = asyncio.Semaphore(max_concurrent)

        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)

        progress_bar = tqdm(total=chunks, desc="BINANCE PROGRESS")
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            tasks = [download_chunk(session, x, semaphore) for x in range(chunks)]
            for coro in asyncio.as_completed(tasks):
                try:
                    chunk_index, temp_data = await coro
                    chunk_results.append((chunk_index, temp_data))
                    progress_bar.update(1)
                except Exception as e:
                    print(f"[red]Error downloading chunk: {e}[/red]")
                    progress_bar.update(1)

        progress_bar.close()
        return chunk_results

    try:
        chunk_results = asyncio.run(download_all_chunks())
    except RuntimeError:
        # jupyter notebooks
        import concurrent.futures
        
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(download_all_chunks())
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            chunk_results = future.result()
    data = (
        pd.concat(
            [df for _, df in sorted(chunk_results, key=lambda x: x[0])],
            ignore_index=True,
        )
        if chunk_results
        else pd.DataFrame(
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )
    )
    if not data.empty:
        data["Datetime"] = pd.to_datetime(data["Datetime"], unit="s")
        data.sort_values("Datetime", inplace=True)
        data.reset_index(drop=True, inplace=True)

        data["HL2"] = (data["High"] + data["Low"]) / 2
        data["OHLC4"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
        data["HLC3"] = (data["High"] + data["Low"] + data["Close"]) / 3
    else:
        print(f"[red]No data retrieved for {binance_symbol}[/red]")

    return data


def fetch_data(
    symbols: list[str],  # first symbol is treated as main symbol
    days: int,
    interval: str,
    age_days: int,
    data_source: str,
    cache_expiry_hours: int = 24,
    retry_limit: int = 3,
    verbose: bool = True,
    proxies: dict = {},
) -> pd.DataFrame:
    """ """
    (
        print(
            f"[yellow]FETCHING DATASET(S) {', '.join(symbols)}; {days}d of {interval}[/yellow]"
        )
        if verbose
        else None
    )

    appdata = os.getenv("APPDATA")
    if appdata:
        temp_dir = Path(appdata) / "market_data"
    else:
        temp_dir = Path.home() / ".local" / "share" / "market_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    cache_key = f"{'_'.join(symbols)}_{days}_{interval}_{age_days}_{data_source}"
    cache_file = temp_dir / f"{cache_key}.csv"

    if os.path.exists(cache_file):
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        file_age_hours = (datetime.now() - file_modified_time).total_seconds() / 3600

        if (file_age_hours < cache_expiry_hours) or (cache_expiry_hours == -1):
            try:
                cached_data = pd.read_csv(cache_file, parse_dates=["Datetime"])
                (
                    print(
                        f"[blue]USING CACHE {cache_file}[/blue] ({os.path.getsize(cache_file)/(1024**2):.2f} MB {cached_data.shape[0]} bars)"
                    )
                    if verbose
                    else None
                )
                return cached_data
            except Exception as e:
                print(f"[yellow]Cache read error: {e}. Fetching fresh data...[/yellow]")
        else:
            print(
                f"[yellow]Cache expired ({file_age_hours:.1f} hours old). Fetching fresh data...[/yellow]"
            )

    anchored_end_time = datetime.now(timezone.utc) - timedelta(days=age_days) ### important! ensures no NANs from subtle datetime mismatches
    print("[green]DOWNLOADING DATA[/green]")
    main_symbol = symbols[0]
    if data_source == "yfinance":
        data = pd.DataFrame()
        for symbol in symbols:
            if symbol == main_symbol:
                temp_df = _fetch_yfinance(symbol, days, interval, age_days)
            else:
                temp_df = _fetch_yfinance(symbol, days, interval, age_days)
                temp_df = temp_df.drop(columns=["Datetime"])
                temp_df.rename(
                    columns={col: f"add_{symbol}_{col}" for col in temp_df.columns},
                    inplace=True,
                )
            data = pd.concat([data, temp_df], axis=1)

    elif data_source == "birdeye":
        data = pd.DataFrame()
        for symbol in symbols:
            if symbol == main_symbol:
                temp_df = _fetch_birdeye(
                    symbol,
                    days,
                    interval,
                    age_days,
                    retry_limit,
                    proxies,
                    anchored_end_time,
                )
            else:
                temp_df = _fetch_birdeye(
                    symbol,
                    days,
                    interval,
                    age_days,
                    retry_limit,
                    proxies,
                    anchored_end_time,
                )
                temp_df = temp_df.drop(columns=["Datetime"])
                temp_df.rename(
                    columns={col: f"add_{symbol}_{col}" for col in temp_df.columns},
                    inplace=True,
                )
            data = pd.concat([data, temp_df], axis=1)
    elif data_source == "binance":
        data = pd.DataFrame()
        for symbol in symbols:
            if symbol == main_symbol:
                temp_df = _fetch_binance(
                    symbol,
                    days,
                    interval,
                    age_days,
                    retry_limit,
                    proxies,
                    anchored_end_time,
                )
            else:
                temp_df = _fetch_binance(
                    symbol,
                    days,
                    interval,
                    age_days,
                    retry_limit,
                    proxies,
                    anchored_end_time,
                )
                temp_df = temp_df.drop(columns=["Datetime"])
                temp_df.rename(
                    columns={col: f"add_{symbol}_{col}" for col in temp_df.columns},
                    inplace=True,
                )
            data = pd.concat([data, temp_df], axis=1)
    else:
        raise ValueError(
            f"Unknown data_source: {data_source}. Choose from 'binance', 'kucoin', 'yfinance'."
        )

    if not data.empty and len(data) > 0:
        try:
            data.to_csv(cache_file, index=False)
            print(
                f"[blue]Data cached to {cache_file}[/blue] ({os.path.getsize(cache_file)/(1024**2):.2f} MB {len(data)} bars)"
            )
        except Exception as e:
            print(f"[yellow]Failed to cache data: {e}[/yellow]")
    else:
        print(f"[yellow]Skipping cache - data is empty[/yellow]")

    return data
