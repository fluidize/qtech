import asyncio
import websockets
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Callable, Optional, Any
from collections import deque
import logging
from rich import print
from rich.console import Console
from rich.table import Table
from rich.live import Live
import sys
import os

sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")

from backtesting import Strategy
import technical_analysis as ta
from data_provider_factory import DataProviderFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

class LiveTradingSystem:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        data_source: str = "binance",
        buffer_size: int = 500,
        strategy_func: Callable = None,
        strategy_params: Dict = None,
        signal_callback: Callable = None
    ):
        """
        Initialize the live trading system.
        
        Args:
            symbol: Trading pair symbol
            data_source: Data provider ('kucoin', 'binance')
            buffer_size: Number of candles to keep in memory
            strategy_func: Strategy function from Strategy class
            strategy_params: Parameters for the strategy
            signal_callback: Function to call when new signal is generated
        """
        self.symbol = symbol
        self.interval = interval
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.strategy_func = strategy_func or Strategy.ema_cross_strategy
        self.strategy_params = strategy_params or {}
        self.signal_callback = signal_callback
        
        # Data storage
        self.data_buffer = deque(maxlen=buffer_size)
        self.current_data = pd.DataFrame()
        
        # Data provider
        self.provider = DataProviderFactory.create_provider(data_source, symbol=symbol, interval=interval)
        
        # WebSocket connection
        self.connection = None
        
        # State management
        self.running = False
        self.last_signal = 2  # Start with flat position
        self.signal_history = deque(maxlen=100)
        
        # Console for rich output
        self.console = Console()
        
        # Data provider will be initialized async in start() method
        self._initialized = False
    
    async def _initialize_provider(self):
        """Initialize the data provider and bootstrap with historical data."""
        try:
            # Setup provider connection
            await self.provider.setup_connection()
            logger.info("Provider connection setup completed")
            
            # Bootstrap with historical data
            historical_data = self.provider.get_historical_data(limit=self.buffer_size)
            for candle in historical_data:
                self.data_buffer.append(candle)
            
            logger.info(f"Bootstrapped with {len(self.data_buffer)} historical candles from {self.data_source}")
            
            # Update dataframe after bootstrapping
            self._update_dataframe()
            
        except Exception as e:
            logger.warning(f"Failed to initialize provider: {e}")
    
    def _create_ohlcv_from_tick(self, price: float, volume: float = 0, timestamp: datetime = None) -> Dict:
        """Create OHLCV candle from tick data."""
        if timestamp is None:
            timestamp = datetime.now()
            
        return {
            'Datetime': timestamp,
            'Open': price,
            'High': price,
            'Low': price,
            'Close': price,
            'Volume': volume
        }
    
    def _update_current_candle(self, market_data: Dict):
        """Update the current minute candle with new market data."""
        logger.debug(f"Updating candle with data: {market_data}")
        
        # Handle complete OHLC data (like Binance klines)
        if all(key in market_data for key in ['open', 'high', 'low', 'close']):
            candle = {
                'Datetime': market_data['timestamp'],
                'Open': market_data['open'],
                'High': market_data['high'],
                'Low': market_data['low'],
                'Close': market_data['close'],
                'Volume': market_data['volume']
            }
            
            if len(self.data_buffer) == 0:
                self.data_buffer.append(candle)
                logger.info(f"Created first candle from OHLC: {candle}")
            else:
                current_candle = self.data_buffer[-1]
                current_minute = current_candle['Datetime'].replace(second=0, microsecond=0)
                new_minute = market_data['timestamp'].replace(second=0, microsecond=0)
                
                if current_minute == new_minute:
                    self.data_buffer[-1] = candle
                    logger.debug(f"Updated existing candle: {current_candle['Close']} -> {candle['Close']}")
                else:
                    self.data_buffer.append(candle)
                    logger.info(f"Added new candle: {candle}")
                    logger.info(f"Buffer now has {len(self.data_buffer)} candles")
        
        # Handle tick data (like KuCoin price updates)
        elif 'price' in market_data:
            if len(self.data_buffer) == 0:
                candle = self._create_ohlcv_from_tick(
                    market_data['price'], 
                    market_data.get('volume', 0),
                    market_data['timestamp']
                )
                self.data_buffer.append(candle)
                logger.info(f"Created first candle from tick: {candle}")
            else:
                current_candle = self.data_buffer[-1]
                current_minute = current_candle['Datetime'].replace(second=0, microsecond=0)
                tick_minute = market_data['timestamp'].replace(second=0, microsecond=0)
                
                if tick_minute == current_minute:
                    old_close = current_candle['Close']
                    current_candle['High'] = max(current_candle['High'], market_data['price'])
                    current_candle['Low'] = min(current_candle['Low'], market_data['price'])
                    current_candle['Close'] = market_data['price']
                    current_candle['Volume'] += market_data.get('volume', 0)
                    logger.debug(f"Updated candle with tick: {old_close} -> {current_candle['Close']}")
                else:
                    candle = self._create_ohlcv_from_tick(
                        market_data['price'],
                        market_data.get('volume', 0),
                        market_data['timestamp']
                    )
                    self.data_buffer.append(candle)
                    logger.info(f"Created new candle from tick: {candle}")
                    logger.info(f"Buffer now has {len(self.data_buffer)} candles")
        else:
            logger.warning(f"Unhandled market data format: {market_data}")
    
    def _update_dataframe(self):
        """Convert buffer to DataFrame for strategy processing."""
        if len(self.data_buffer) < 10:  # Need minimum data for indicators
            return
            
        self.current_data = pd.DataFrame(list(self.data_buffer))
        self.current_data.set_index('Datetime', inplace=True)
        self.current_data = self.current_data.sort_index()
    
    def _run_strategy(self) -> int:
        """Run the trading strategy on current data."""
        if self.current_data.empty or len(self.current_data) < 20:
            return self.last_signal
        
        try:
            # Run strategy
            signals = self.strategy_func(self.current_data, **self.strategy_params)
            
            # Get the latest signal
            latest_signal = signals.iloc[-1] if len(signals) > 0 else 2
            
            # Convert to int if it's not already
            if isinstance(latest_signal, (pd.Series, np.ndarray)):
                latest_signal = int(latest_signal.iloc[-1] if hasattr(latest_signal, 'iloc') else latest_signal[0])
            else:
                latest_signal = int(latest_signal)
            
            return latest_signal
            
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return self.last_signal
    
    def _handle_signal_change(self, new_signal: int):
        """Handle when strategy generates a new signal."""
        if new_signal != self.last_signal:
            # Get current price from buffer or dataframe
            current_price = 0
            if not self.current_data.empty:
                current_price = self.current_data['Close'].iloc[-1]
            elif self.data_buffer:
                current_price = self.data_buffer[-1]['Close']
            
            signal_info = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'previous_signal': self.last_signal,
                'new_signal': new_signal,
                'current_price': current_price,
                'action': self._signal_to_action(self.last_signal, new_signal)
            }
            
            self.signal_history.append(signal_info)
            self.last_signal = new_signal
            
            # Call user callback if provided
            if self.signal_callback:
                self.signal_callback(signal_info)
            
            # Log the signal
            self._log_signal(signal_info)
    
    def _signal_to_action(self, old_signal: int, new_signal: int) -> str:
        """Convert signal change to human-readable action."""
        signal_names = {0: 'HOLD', 1: 'SHORT', 2: 'FLAT', 3: 'LONG'}
        
        if old_signal == new_signal:
            return f"MAINTAIN {signal_names.get(new_signal, 'UNKNOWN')}"
        
        old_name = signal_names.get(old_signal, 'UNKNOWN')
        new_name = signal_names.get(new_signal, 'UNKNOWN')
        
        if new_signal == 2:  # Going to flat
            return f"CLOSE {old_name} → FLAT"
        elif old_signal == 2:  # Opening from flat
            return f"OPEN {new_name}"
        else:  # Direct switch
            return f"SWITCH {old_name} → {new_name}"
    
    def _log_signal(self, signal_info: Dict):
        """Log the trading signal."""
        timestamp = signal_info['timestamp'].strftime('%H:%M:%S')
        action = signal_info['action']
        price = signal_info['current_price']
        
        # Color coding for different actions
        if 'LONG' in action or 'OPEN' in action:
            color = 'green'
        elif 'SHORT' in action:
            color = 'red'
        elif 'CLOSE' in action or 'FLAT' in action:
            color = 'yellow'
        else:
            color = 'blue'
        
        self.console.print(f"[{color}]{timestamp} | {self.symbol} {self.interval} | {action} @ ${price:.4f}[/{color}]")
    
    async def _websocket_handler(self):
        """Handle WebSocket connection and message processing."""
        try:
            logger.info(f"Attempting to connect to WebSocket: {self.provider.ws_url}")
            
            async with websockets.connect(self.provider.ws_url) as ws:
                self.connection = ws
                logger.info(f"Connected to {self.data_source} WebSocket")
                
                # Subscribe to symbol if needed
                subscribe_msg = self.provider.get_subscription_message()
                if subscribe_msg:
                    logger.info(f"Sending subscription message: {subscribe_msg}")
                    await ws.send(json.dumps(subscribe_msg))
                else:
                    logger.info("No subscription message needed for this provider")
                
                logger.info("🔄 Starting message processing loop...")   
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(message)
                        
                        # Process message using provider
                        market_data = self.provider.process_message(data)
                        
                        if market_data:
                            self._update_current_candle(market_data)
                            self._update_dataframe()
                            
                            # Run strategy and check for signals
                            new_signal = self._run_strategy()
                            self._handle_signal_change(new_signal)
                    
                    except asyncio.TimeoutError:
                        logger.debug("WebSocket timeout - pinging...")
                        ping_msg = self.provider.get_ping_message()
                        if ping_msg:
                            await ws.send(json.dumps(ping_msg))
                    
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed, attempting to reconnect...")
                        break
                    
                    except Exception as e:
                        logger.error(f"Error in WebSocket handler: {e}")
                        await asyncio.sleep(1)
        
        except websockets.exceptions.InvalidURI as e:
            logger.error(f"Invalid WebSocket URI: {self.provider.ws_url}")
            logger.error(f"URI Error: {e}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket connection error: {e}")
            logger.error(f"WebSocket Error type: {type(e).__name__}")
        except Exception as e:
            logger.error(f"Fatal error in WebSocket handler: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
        if self.running:
            logger.info("Attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)
    
    async def start(self):
        """Start the live trading system."""
        self.running = True
        self.console.print(f"[green]Starting Live Trading System for {self.symbol}[/green]")
        self.console.print(f"[blue]Data Source: {self.data_source}[/blue]")
        self.console.print(f"[blue]Interval: {self.interval}[/blue]")
        self.console.print(f"[blue]Strategy: {self.strategy_func.__name__}[/blue]")
        self.console.print(f"[blue]Parameters: {self.strategy_params}[/blue]")
        
        # Initialize provider if not already done
        if not self._initialized:
            self.console.print("[yellow]Initializing data provider...[/yellow]")
            await self._initialize_provider()
            self._initialized = True
            self.console.print("[green]Data provider initialized successfully[/green]")
        
        self.console.print("[yellow]Starting WebSocket connection...[/yellow]")
        
        connection_attempts = 0
        max_attempts = 5
        
        while self.running and connection_attempts < max_attempts:
            try:
                connection_attempts += 1
                logger.info(f"WebSocket connection attempt {connection_attempts}/{max_attempts}")
                
                await self._websocket_handler()
                
                # If we get here, the connection was closed but we should retry
                if self.running:
                    logger.warning(f"WebSocket disconnected, retrying in 10 seconds... (attempt {connection_attempts})")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Fatal error in trading system (attempt {connection_attempts}): {e}")
                logger.error(f"Error type: {type(e).__name__}")
                
                if connection_attempts >= max_attempts:
                    logger.error(f"Max connection attempts ({max_attempts}) reached. Stopping.")
                    self.running = False
                    break
                    
                if self.running:
                    wait_time = min(30, connection_attempts * 5)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
        
        if connection_attempts >= max_attempts:
            self.console.print("[red]Failed to establish stable connection after maximum attempts[/red]")
        else:
            self.console.print("[yellow]Trading system stopped[/yellow]")
    
    def stop(self):
        """Stop the live trading system."""
        self.running = False
        self.console.print("[red]Stopping Live Trading System[/red]")
    
    def get_current_status(self) -> Dict:
        """Get current system status."""
        # Get last price from buffer or dataframe
        last_price = 0
        if not self.current_data.empty:
            last_price = self.current_data['Close'].iloc[-1]
        elif self.data_buffer:
            last_price = self.data_buffer[-1]['Close']
            
        return {
            'running': self.running,
            'symbol': self.symbol,
            'data_source': self.data_source,
            'buffer_size': len(self.data_buffer),
            'current_signal': self.last_signal,
            'last_price': last_price,
            'signal_history_count': len(self.signal_history)
        }
    
    def get_recent_signals(self, count: int = 10) -> List[Dict]:
        """Get recent trading signals."""
        return list(self.signal_history)[-count:]

class LiveTradingMonitor:
    """Monitor and display live trading system status."""
    
    def __init__(self, trading_system: LiveTradingSystem):
        self.system = trading_system
        self.console = Console()
    
    def create_status_table(self) -> Table:
        """Create a status table for display."""
        table = Table(title="Live Trading System Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        status = self.system.get_current_status()
        
        table.add_row("Symbol", status['symbol'])
        table.add_row("Data Source", status['data_source'])
        table.add_row("Status", "Running" if status['running'] else "Stopped")
        table.add_row("Buffer Size", str(status['buffer_size']))
        table.add_row("Current Signal", {0: 'HOLD', 1: 'SHORT', 2: 'FLAT', 3: 'LONG'}.get(status['current_signal'], 'UNKNOWN'))
        table.add_row("Last Price", f"${status['last_price']:.4f}")
        table.add_row("Total Signals", str(status['signal_history_count']))
        
        return table
    
    def display_recent_signals(self, count: int = 5):
        """Display recent trading signals."""
        signals = self.system.get_recent_signals(count)
        
        if not signals:
            self.console.print("[yellow]No recent signals[/yellow]")
            return
        
        table = Table(title=f"Recent Signals (Last {len(signals)})")
        table.add_column("Time", style="cyan")
        table.add_column("Action", style="green")
        table.add_column("Price", style="yellow")
        
        for signal in signals:
            timestamp = signal['timestamp'].strftime('%H:%M:%S')
            table.add_row(timestamp, signal['action'], f"${signal['current_price']:.4f}")
        
        self.console.print(table)

# Example usage functions
def example_signal_callback(signal_info: Dict):
    """Example callback function for handling trading signals."""
    print(f"🚨 TRADING SIGNAL: {signal_info['action']} at ${signal_info['current_price']:.4f}")
    
    # - Send notifications (email, SMS, Discord, etc.)
    # - Execute trades through a broker API
    # - Log to database
    # - Update a dashboard

async def run_simple_system():
    """Simple version without status display - just signal logging."""
    
    system = LiveTradingSystem(
        symbol="BTCUSDT",
        interval="1m",
        data_source="binance",
        buffer_size=200,
        strategy_func=Strategy.macd_strategy,
        strategy_params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        signal_callback=example_signal_callback
    )
    
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop()

if __name__ == "__main__":
    # Choose one:
    # asyncio.run(run_live_system())        # With status display
    asyncio.run(run_simple_system())    # Simple version
    # asyncio.run(test_binance_bootstrap()) # Test data bootstrap only