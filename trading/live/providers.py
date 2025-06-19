from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import time
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BaseDataProvider(ABC):
    """Base class for data providers (Binance, KuCoin, etc.)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ws_url = None
        self.ws_token = None
        
    @abstractmethod
    async def setup_connection(self) -> None:
        """Setup WebSocket connection parameters."""
        pass
    
    @abstractmethod
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for the specific exchange."""
        pass
    
    @abstractmethod
    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process WebSocket message and return normalized data."""
        pass
    
    @abstractmethod
    def get_historical_data(self, limit: int = 60) -> List[Dict]:
        """Fetch historical candle data."""
        pass
    
    @abstractmethod
    def get_subscription_message(self) -> Optional[Dict]:
        """Get WebSocket subscription message if needed."""
        pass
    
    @abstractmethod
    def get_ping_message(self) -> Optional[Dict]:
        """Get WebSocket ping message if needed."""
        pass
    
    def normalize_candle_data(self, raw_data: Dict) -> Dict:
        """Normalize candle data to standard format."""
        return {
            'Datetime': raw_data.get('timestamp', datetime.now()),
            'Open': float(raw_data.get('open', 0)),
            'High': float(raw_data.get('high', 0)),
            'Low': float(raw_data.get('low', 0)),
            'Close': float(raw_data.get('close', 0)),
            'Volume': float(raw_data.get('volume', 0))
        }
    
class BinanceProvider(BaseDataProvider):
    """Binance data provider for WebSocket and REST API."""
    
    def __init__(self, symbol: str, interval: str = '1m'):
        super().__init__(symbol)
        self.base_url = "https://api.binance.com"
        self.ws_base_url = "wss://stream.binance.com:9443/ws"
        self.interval = interval
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for Binance (remove hyphens, uppercase)."""
        return symbol.replace('-', '').upper()
    
    async def setup_connection(self) -> None:
        """Setup Binance WebSocket connection."""
        # Format symbol for Binance
        formatted_symbol = self.format_symbol(self.symbol).lower()
        self.ws_url = f"{self.ws_base_url}/{formatted_symbol}@kline_{self.interval}"
        
        logger.info(f"Binance WebSocket URL set: {self.ws_url}")
        logger.info(f"Using symbol: {formatted_symbol} interval: {self.interval}")
    
    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process Binance WebSocket message."""
        try:
            if 'k' in message:  # Kline data
                kline = message['k']
                logger.debug(f"Processing Binance kline: {kline}")
                
                return {
                    'timestamp': datetime.fromtimestamp(int(kline['t']) / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['q']), #Volume in quote asset
                    'is_closed': kline['x'],  # Whether this kline is closed
                    'provider': 'binance'
                }
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
            logger.error(f"Message was: {message}")
        return None
    
    def get_historical_data(self, limit: int = 60) -> List[Dict]:
        """Fetch recent 1-minute candles from Binance REST API."""
        formatted_symbol = self.format_symbol(self.symbol)
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': formatted_symbol,
            'interval': self.interval,
            'limit': limit
        }
        
        logger.info(f"Fetching Binance data for symbol: {formatted_symbol} interval: {self.interval}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        candles = response.json()
        
        logger.info(f"Received {len(candles)} candles from Binance")
        
        historical_data = []
        for candle_data in candles:
            timestamp = datetime.fromtimestamp(int(candle_data[0]) / 1000)
            candle = {
                'timestamp': timestamp,
                'open': float(candle_data[1]),
                'high': float(candle_data[2]),
                'low': float(candle_data[3]),
                'close': float(candle_data[4]),
                'volume': float(candle_data[5]),
                'provider': 'binance'
            }
            historical_data.append(self.normalize_candle_data(candle))
        
        return historical_data
    
    def get_subscription_message(self) -> Optional[Dict]:
        """Binance doesn't require explicit subscription."""
        return None
    
    def get_ping_message(self) -> Optional[Dict]:
        """Binance doesn't require explicit ping."""
        return None
    
class KuCoinProvider(BaseDataProvider):
    """KuCoin data provider for WebSocket and REST API."""
    
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.base_url = "https://api.kucoin.com"
        self.ws_token = None
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for KuCoin (keep hyphens, uppercase)."""
        return symbol.upper()
    
    async def setup_connection(self) -> None:
        """Setup KuCoin WebSocket connection."""
        try:
            # Get WebSocket token from KuCoin
            url = f"{self.base_url}/api/v1/bullet-public"
            response = requests.post(url, headers={}, data={})
            response.raise_for_status()
            
            token_data = response.json()
            if token_data['code'] == '200000':
                self.ws_token = token_data['data']['token']
                endpoint = token_data['data']['instanceServers'][0]['endpoint']
                connect_id = int(time.time() * 1000)
                self.ws_url = f"{endpoint}?token={self.ws_token}&connectId={connect_id}"
                
                logger.info(f"KuCoin WebSocket token obtained successfully")
                logger.info(f"WebSocket URL: {self.ws_url}")
            else:
                raise Exception(f"Failed to get KuCoin token: {token_data}")
                
        except Exception as e:
            logger.error(f"Failed to setup KuCoin connection: {e}")
            raise
    
    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process KuCoin WebSocket message."""
        try:
            logger.debug(f"Received KuCoin message: {message}")
            
            if message.get('type') == 'message' and message.get('topic', '').startswith('/market/ticker:'):
                data = message.get('data', {})
                logger.debug(f"Processing ticker data: {data}")
                
                return {
                    'timestamp': datetime.fromtimestamp(int(data.get('time', 0)) / 1000),
                    'price': float(data.get('price', 0)),
                    'volume': float(data.get('vol', 0)),
                    'provider': 'kucoin'
                }
            elif message.get('type') == 'welcome':
                logger.info("KuCoin WebSocket connection established")
            elif message.get('type') == 'ack':
                logger.info(f"KuCoin subscription acknowledged: {message}")
            else:
                logger.debug(f"Unhandled KuCoin message type: {message.get('type')}")
                
        except Exception as e:
            logger.error(f"Error processing KuCoin message: {e}")
            logger.error(f"Message was: {message}")
        return None
    
    def get_historical_data(self, limit: int = 60) -> List[Dict]:
        """Fetch recent 1-minute candles from KuCoin REST API."""
        formatted_symbol = self.format_symbol(self.symbol)
        url = f"{self.base_url}/api/v1/market/candles"
        params = {
            'symbol': formatted_symbol,
            'type': '1min',
            'startAt': int((datetime.now() - timedelta(hours=1)).timestamp()),
            'endAt': int(datetime.now().timestamp())
        }
        
        logger.info(f"Fetching KuCoin data for symbol: {formatted_symbol}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        historical_data = []
        if data.get('code') == '200000':
            candles = data['data']
            logger.info(f"Received {len(candles)} candles from KuCoin")
            
            # KuCoin returns newest first, so reverse it
            for candle_data in reversed(candles):
                timestamp = datetime.fromtimestamp(int(candle_data[0]))
                candle = {
                    'timestamp': timestamp,
                    'open': float(candle_data[1]),
                    'close': float(candle_data[2]),
                    'high': float(candle_data[3]),
                    'low': float(candle_data[4]),
                    'volume': float(candle_data[5]),
                    'provider': 'kucoin'
                }
                historical_data.append(self.normalize_candle_data(candle))
        
        return historical_data
    
    def get_subscription_message(self) -> Optional[Dict]:
        """Get KuCoin subscription message."""
        formatted_symbol = self.format_symbol(self.symbol)
        return {
            "id": int(time.time() * 1000),
            "type": "subscribe",
            "topic": f"/market/ticker:{formatted_symbol}",
            "privateChannel": False,
            "response": True
        }
    
    def get_ping_message(self) -> Optional[Dict]:
        """Get KuCoin ping message."""
        return {
            "id": int(time.time() * 1000),
            "type": "ping"
        }
    
# --- TESTING BINANCE WS STREAM ---

from rich import print

async def main():
    import websockets
    import json
    symbol = "btcusdt"
    interval = "1m"
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    print(f"Connecting to {ws_url}")
    async with websockets.connect(ws_url) as ws:
        print("Connected. Listening for messages...")
        for _ in range(1):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Received: {data}")

if __name__ == "__main__":
    asyncio.run(main())