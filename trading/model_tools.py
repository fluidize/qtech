from math import ceil
from datetime import datetime, timedelta, timezone
import requests
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import os
import json
import time
from pathlib import Path
import asyncio
import aiohttp

from rich import print

import sys
sys.path.append("trading")


def fetch_data(symbol, days, interval, age_days, data_source: str = "binance", cache_expiry_hours: int = 24, retry_limit: int = 3, verbose: bool = True, proxies: dict = {}):
    print(f"[yellow]FETCHING DATA {symbol} {interval}[/yellow]") if verbose else None
    
    appdata = os.getenv("APPDATA")
    if appdata:
        temp_dir = Path(appdata) / "market_data"
    else:
        temp_dir = Path.home() / ".local" / "share" / "market_data"

    temp_dir.mkdir(parents=True, exist_ok=True)

    cache_key = f"{symbol}_{days}_{interval}_{age_days}_{data_source}"
    cache_file = temp_dir / f"{cache_key}.parquet"

    if cache_expiry_hours > 0 and os.path.exists(cache_file):
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        file_age_hours = (datetime.now() - file_modified_time).total_seconds() / 3600

        if file_age_hours < cache_expiry_hours:
            try:
                cached_data = pd.read_parquet(cache_file)
                print(f"[blue]USING CACHE {cache_file}[/blue] ({os.path.getsize(cache_file)/(1024**2):.2f} MB, {cached_data.shape[0]} bars)") if verbose else None

                with open(f"{cache_file}.json", "w") as f:
                    json.dump({
                        "symbol": symbol,
                        "days": days,
                        "interval": interval,
                        "age_days": age_days,
                        "data_source": data_source,
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
    if data_source == "yfinance":
        end_date = datetime.now() - timedelta(days=age_days)
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False, auto_adjust=True, threads=False)

        data.sort_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
        data = pd.DataFrame(data)

    elif data_source == "birdeye":
        chunks = ceil(days * 1.44) # adjust due to birdeye 1k bar limit

        proxy_url = proxies.get('https') or proxies.get('http') if proxies else None
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": os.getenv("BIRDEYE_API_KEY")
        }

        chunk_results = []
        times = []
        progress_bar = tqdm(total=chunks, desc="BIRDEYE PROGRESS", ascii="#>")
        
        for chunk_index in range(chunks):
            retries = 0
            end_time = datetime.now(timezone.utc) - timedelta(minutes=1000*chunk_index) - timedelta(days=age_days)
            start_time = end_time - timedelta(minutes=1000)

            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            url = f"https://public-api.birdeye.so/defi/ohlcv?address={symbol}&type={interval}&currency=usd&time_from={start_ts}&time_to={end_ts}&ui_amount_mode=raw"
            
            while retries <= retry_limit:
                try:
                    response = requests.get(url, headers=headers, proxies=proxies if proxies else None, timeout=30)
                    response.raise_for_status()
                    request_data = response.json()
                    break
                except Exception as e:
                    if retries < retry_limit:
                        retries += 1
                        time.sleep(0.5)
                    else:
                        print(f"[red]Error fetching {symbol} chunk {chunk_index} after {retry_limit} retries: {e}[/red]")
                        progress_bar.update(1)
                        continue
            
            if retries > retry_limit:
                continue
            
            records = []
            for kline in request_data["data"]["items"]:
                records.append({
                    "Datetime": kline["unixTime"],
                    "Open": float(kline["o"]),
                    "High": float(kline["h"]),
                    "Low": float(kline["l"]),
                    "Close": float(kline["c"]),
                    "Volume": float(kline["v"])
                })

            temp_data = pd.DataFrame(records)
            chunk_results.append((chunk_index, temp_data))
            times.append(start_time)
            times.append(end_time)
            progress_bar.update(1)
            
            time.sleep(0.1)
        
        progress_bar.close()

        chunk_results.sort(key=lambda x: x[0])
        
        data_list = [df for _, df in chunk_results if not df.empty]
        if data_list:
            data = pd.concat(data_list, ignore_index=True)
        else:
            data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

        if not data.empty:
            data["Datetime"] = pd.to_datetime(data['Datetime'], unit='s')
            data.sort_values('Datetime', inplace=True)
            data = data.drop_duplicates(subset=['Datetime'], keep='first')
            data.reset_index(drop=True, inplace=True)

            data['HL2'] = (data['High'] + data['Low']) / 2
            data['OHLC4'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
            data['HLC3'] = (data['High'] + data['Low'] + data['Close']) / 3
        else:
            print(f"[red]No data retrieved for {symbol}[/red]")

    elif data_source == "binance":
        # Parse interval format for Binance
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

        # Format symbol for Binance (remove hyphen if present)
        binance_symbol = symbol.replace('-', '').upper()

        chunks = ceil(days * 1.44) # adjust due to binance 1k bar limit

        proxy_url = proxies.get('https') or proxies.get('http') if proxies else None

        async def download_chunk(session, chunk_index, semaphore):
            async with semaphore:
                retries = 0
                end_time = datetime.now() - timedelta(minutes=1000*chunk_index) - timedelta(days=age_days)
                start_time = end_time - timedelta(minutes=1000)

                params = {
                    "symbol": binance_symbol,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000),
                    "limit": 1000
                }

                url = "https://api.binance.com/api/v3/klines"
                
                while retries <= retry_limit:
                    try:
                        async with session.get(url, params=params, proxy=proxy_url) as response:
                            response.raise_for_status()
                            request_data = await response.json()
                            break
                    except Exception as e:
                        if retries < retry_limit:
                            retries += 1
                            await asyncio.sleep(0.5)
                        else:
                            raise Exception(f"Error fetching {binance_symbol} chunk {chunk_index} after {retry_limit} retries: {e}")

                records = []
                for kline in request_data:
                    records.append({
                        "Datetime": int(kline[0]) / 1000,  # Convert from milliseconds to seconds
                        "Open": float(kline[1]),
                        "High": float(kline[2]),
                        "Low": float(kline[3]),
                        "Close": float(kline[4]),
                        "Volume": float(kline[5])
                    })

                temp_data = pd.DataFrame(records)
                return chunk_index, temp_data

        async def download_all_chunks():
            chunk_results = []
            max_concurrent = min(25, chunks)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            connector = aiohttp.TCPConnector(limit=max_concurrent)
            timeout = aiohttp.ClientTimeout(total=30)
            
            progress_bar = tqdm(total=chunks, desc="BINANCE PROGRESS", ascii="#>")
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
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

        chunk_results = asyncio.run(download_all_chunks())

        chunk_results.sort(key=lambda x: x[0])
        
        data_list = [df for _, df in chunk_results if not df.empty]
        if data_list:
            data = pd.concat(data_list, ignore_index=True)
        else:
            data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

        if not data.empty:
            data["Datetime"] = pd.to_datetime(data['Datetime'], unit='s')
            data.sort_values('Datetime', inplace=True)
            data.reset_index(drop=True, inplace=True)

            data['HL2'] = (data['High'] + data['Low']) / 2
            data['OHLC4'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
            data['HLC3'] = (data['High'] + data['Low'] + data['Close']) / 3
        else:
            print(f"[red]No data retrieved for {binance_symbol}[/red]")

    else:
        raise ValueError(f"Unknown data_source: {data_source}. Choose from 'binance', 'kucoin', 'yfinance'.")

    if cache_expiry_hours > 0 and not data.empty and len(data) > 0:
        try:
            data.to_parquet(cache_file)
            print(f"[blue]Data cached to {cache_file}[/blue] ({os.path.getsize(cache_file)/(1024**2):.2f} MB {len(data)} bars)")

            with open(f"{cache_file}.json", "w") as f:
                json.dump({
                    "symbol": symbol,
                    "days": days,
                    "interval": interval,
                    "age_days": age_days,
                    "data_source": data_source,
                    "cached_time": str(datetime.now()),
                    "rows": len(data)
                }, f, indent=2)

        except Exception as e:
            print(f"[yellow]Failed to cache data: {e}[/yellow]")
    elif cache_expiry_hours > 0 and (data.empty or len(data) == 0):
        print(f"[yellow]Skipping cache - data is empty[/yellow]")

    return data