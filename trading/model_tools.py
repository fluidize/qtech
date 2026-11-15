from math import ceil
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import os
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


def _fetch_binance(symbol, days, interval, age_days, retry_limit, end_time, fail_fast=False):
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
                    if fail_fast:
                        raise Exception(
                            f"Error fetching {binance_symbol} chunk {chunk_index}: {e}"
                        )
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
                    if fail_fast:
                        progress_bar.close()
                        raise
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


def _fetch_jupiter(symbol, days, interval, age_days, retry_limit, end_time, fail_fast=False):
    if "minute" in interval:
        interval = interval.replace("minute", "m")
    elif "min" in interval:
        interval = interval.replace("min", "m")
    elif "hour" in interval:
        interval = interval.replace("hour", "h")
    elif "day" in interval:
        interval = interval.replace("day", "d")
    elif "week" in interval:
        interval = interval.replace("week", "w")
    elif "month" in interval:
        interval = interval.replace("month", "M")

    jupiter_intervals = {
        "1m": ("1_MINUTE", 60),
        "5m": ("5_MINUTE", 300),
        "15m": ("15_MINUTE", 900),
        "30m": ("30_MINUTE", 1800),
        "1h": ("1_HOUR", 3600),
        "4h": ("4_HOUR", 14400),
        "1d": ("1_DAY", 86400),
        "1w": ("1_WEEK", 604800),
        "1wk": ("1_WEEK", 604800),
        "1M": ("1_MONTH", 2592000),
        "1mo": ("1_MONTH", 2592000),
    }

    if interval not in jupiter_intervals:
        raise ValueError(
            f"Unknown interval for jupiter: {interval}. Choose from {list(jupiter_intervals)}."
        )

    jupiter_interval, candle_seconds = jupiter_intervals[interval]
    candles_per_chunk = 5000
    target_candles = ceil(days * 86400 / candle_seconds)
    chunks = max(1, ceil(target_candles / candles_per_chunk))
    url = f"https://datapi.jup.ag/v2/charts/{symbol}"

    async def download_chunk(session, chunk_index, semaphore):
        async with semaphore:
            retries = 0
            chunk_end_time = end_time - timedelta(
                seconds=candle_seconds * candles_per_chunk * chunk_index
            )
            chunk_candles = min(
                candles_per_chunk, target_candles - chunk_index * candles_per_chunk
            )
            params = {
                "interval": jupiter_interval,
                "to": chunk_end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "candles": chunk_candles,
                "type": "price",
                "quote": "usd",
            }

            while retries <= retry_limit:
                try:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        request_data = (await response.json()).get("candles", [])
                        break
                except Exception as e:
                    if fail_fast:
                        raise Exception(
                            f"Error fetching {symbol} chunk {chunk_index}: {e}"
                        )
                    if retries < retry_limit:
                        retries += 1
                        await asyncio.sleep(0.5)
                    else:
                        raise Exception(
                            f"Error fetching {symbol} chunk {chunk_index} after {retry_limit} retries: {e}"
                        )

            records = [
                {
                    "Datetime": int(candle["time"]),
                    "Open": float(candle["open"]),
                    "High": float(candle["high"]),
                    "Low": float(candle["low"]),
                    "Close": float(candle["close"]),
                    "Volume": float(candle["volume"]),
                }
                for candle in request_data
            ]

            return chunk_index, pd.DataFrame(records)

    async def download_all_chunks():
        chunk_results = []
        max_concurrent = min(5, chunks)
        semaphore = asyncio.Semaphore(max_concurrent)

        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)

        progress_bar = tqdm(total=chunks, desc="JUPITER PROGRESS")
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
                    if fail_fast:
                        progress_bar.close()
                        raise
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
        data.drop_duplicates(subset="Datetime", inplace=True)
        data.sort_values("Datetime", inplace=True)
        data.reset_index(drop=True, inplace=True)

        data["HL2"] = (data["High"] + data["Low"]) / 2
        data["OHLC4"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
        data["HLC3"] = (data["High"] + data["Low"] + data["Close"]) / 3
    else:
        print(f"[red]No data retrieved for {symbol}[/red]")

    return data


def fetch_data(
    symbol: str,  # main symbol
    days: int,
    interval: str,
    age_days: int,
    data_source: str,
    add_symbols: list[str] | None = None,
    cache_expiry_hours: int = 24,
    retry_limit: int = 3,
    verbose: bool = True,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """ """
    if isinstance(symbol, (list, tuple, set)):
        raise TypeError(
            "fetch_data() takes a single symbol; pass additional series with add_symbols=[...]"
        )
    add_symbols = list(add_symbols) if add_symbols else []
    all_symbols = [symbol] + add_symbols

    (
        print(
            f"[yellow]FETCHING DATASET(S) {', '.join(all_symbols)}; {days}d of {interval}[/yellow]"
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

    cache_key = f"{'_'.join(all_symbols)}_{days}_{interval}_{age_days}_{data_source}"
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
    if data_source == "yfinance":

        def fetch_one(name):
            return _fetch_yfinance(name, days, interval, age_days)

    elif data_source == "binance":

        def fetch_one(name):
            return _fetch_binance(
                name,
                days,
                interval,
                age_days,
                retry_limit,
                anchored_end_time,
                fail_fast,
            )

    elif data_source == "jupiter":

        def fetch_one(name):
            return _fetch_jupiter(
                name,
                days,
                interval,
                age_days,
                retry_limit,
                anchored_end_time,
                fail_fast,
            )

    else:
        raise ValueError(
            f"Unknown data_source: {data_source}. Choose from 'binance', 'jupiter', 'yfinance'."
        )

    data = fetch_one(symbol)
    for add_symbol in add_symbols:
        temp_df = fetch_one(add_symbol)
        if "Datetime" in temp_df.columns:
            temp_df = temp_df.drop(columns=["Datetime"])
        temp_df.rename(
            columns={col: f"add_{add_symbol}_{col}" for col in temp_df.columns},
            inplace=True,
        )
        data = pd.concat([data, temp_df], axis=1)

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
