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

import plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")

import strategy
import technical_analysis as ta
from data_provider_factory import DataProviderFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

class LiveTradingSystem:
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        interval: str = "1m",
        data_source: str = "binance",
        buffer_size: int = 500,
        strategy_func: Callable = None,
        strategy_params: Dict = None,
        signal_callback: Callable = None,
        always_call_callback: bool = True,
        debug_plot: bool = False,
        debug_plot_window: int = 100
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
            always_call_callback: If True, callback is called on every candle close regardless of signal changes
            debug_plot: Enable debug plotting on each candle close
            debug_plot_window: Number of recent candles to show in debug plot
        """
        self.symbol = symbol
        self.interval = interval
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.strategy_func = strategy_func
        self.strategy_params = strategy_params or {}
        self.signal_callback = signal_callback
        self.always_call_callback = always_call_callback
        self.debug_plot = debug_plot
        self.debug_plot_window = debug_plot_window
        
        # Data storage - buffer only contains closed candles
        self.data_buffer = deque(maxlen=buffer_size)
        self.current_data = pd.DataFrame()  # Will contain exactly what's in buffer
        
        # Data provider
        self.provider = DataProviderFactory.create_provider(data_source, symbol=symbol, interval=interval)
        
        # WebSocket connection
        self.connection = None
        
        # State management
        self.running = False
        self.last_signal = 0  # Start with HOLD position. Last signal is either 1, 2, or 3, never 0 following init.
        self.signal_history = deque(maxlen=100)
        self.last_data_is_closed = False  # Track if last candle was closed
        
        self.console = Console()
        
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
    
    def _update_current_candle(self, market_data: Dict) -> tuple:
        """Update the current minute candle with new market data.
        Returns (candle_closed, current_candle) where:
        - candle_closed: True if a closed candle was processed, False otherwise
        - current_candle: The current candle (either open or closed) for price display
        """
        logger.debug(f"Updating candle with data: {market_data}")
        
        # Handle complete OHLC data (like Binance klines)
        if all(key in market_data for key in ['Open', 'High', 'Low', 'Close']):
            candle = {
                'Datetime': market_data['Timestamp'],
                'Open': market_data['Open'],
                'High': market_data['High'],
                'Low': market_data['Low'],
                'Close': market_data['Close'],
                'Volume': market_data['Volume'],
                'Is_Closed': market_data['Is_Closed']
            }
            
            # Only add to buffer if candle is closed
            if candle['Is_Closed']:
                if len(self.data_buffer) == 0:
                    self.data_buffer.append(candle)
                    logger.info(f"Added first closed candle to buffer: {candle}")
                else:
                    current_buffer_candle = self.data_buffer[-1]
                    current_minute = current_buffer_candle['Datetime'].replace(second=0, microsecond=0)
                    new_minute = market_data['Timestamp'].replace(second=0, microsecond=0)
                    
                    if current_minute == new_minute:
                        # Update existing candle in buffer
                        self.data_buffer[-1] = candle
                        logger.debug(f"Updated existing closed candle in buffer: {current_buffer_candle['Close']} -> {candle['Close']}")
                    else:
                        # Add new closed candle to buffer
                        self.data_buffer.append(candle)
                        logger.debug(f"Added new closed candle to buffer: {candle}")
                
                # Detect transition from open to closed candle
                candle_just_closed = not self.last_data_is_closed
                self.last_data_is_closed = True
                return candle_just_closed, candle
            else:
                # Candle is still open, just update tracking
                self.last_data_is_closed = False
                return False, candle
        
        # Handle tick data (like KuCoin price updates)
        elif 'Price' in market_data:
            if len(self.data_buffer) == 0:
                # Create first candle
                candle = self._create_ohlcv_from_tick(
                    market_data['Price'], 
                    market_data.get('Volume', 0),
                    market_data['Timestamp']
                )
                # For tick data, we'll treat the first candle as closed to bootstrap
                candle['Is_Closed'] = True
                self.data_buffer.append(candle)
                logger.info(f"Created first candle from tick: {candle}")
                return False, candle
            else:
                # Get the last candle from buffer
                last_candle = self.data_buffer[-1]
                current_minute = last_candle['Datetime'].replace(second=0, microsecond=0)
                tick_minute = market_data['Timestamp'].replace(second=0, microsecond=0)
                
                if tick_minute == current_minute:
                    # Update current candle in buffer (treating it as still forming)
                    old_close = last_candle['Close']
                    last_candle['High'] = max(last_candle['High'], market_data['Price'])
                    last_candle['Low'] = min(last_candle['Low'], market_data['Price'])
                    last_candle['Close'] = market_data['Price']
                    last_candle['Volume'] += market_data.get('Volume', 0)
                    logger.debug(f"Updated candle with tick: {old_close} -> {last_candle['Close']}")
                    return False, last_candle
                else:
                    # Create new candle for new minute
                    new_candle = self._create_ohlcv_from_tick(
                        market_data['Price'],
                        market_data.get('Volume', 0),
                        market_data['Timestamp']
                    )
                    new_candle['Is_Closed'] = True  # Mark as closed since minute changed
                    self.data_buffer.append(new_candle)
                    logger.info(f"Created new candle from tick: {new_candle}")
                    return True, new_candle  # New candle created
        else:
            logger.warning(f"Unhandled market data format: {market_data}")
            return False, None
    
    def _update_dataframe(self):
        """Convert all candles from buffer to DataFrame for strategy processing.
        Since buffer only contains closed candles, this ensures memory efficiency."""
        if len(self.data_buffer) == 0:
            self.current_data = pd.DataFrame()
            logger.debug("No closed candles in buffer, DataFrame is empty")
            return
        
        # Convert all candles in buffer to DataFrame (all are closed)
        self.current_data = pd.DataFrame(list(self.data_buffer))
        self.current_data.set_index('Datetime', inplace=True)
        self.current_data = self.current_data.sort_index()
        logger.debug(f"Updated DataFrame with {len(self.data_buffer)} closed candles from buffer")
    
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
        """Handle when strategy generates a new signal. Adds signal info to instance log and prints to console."""
        # Get current price from current_data, buffer, or dataframe
        current_price = 0
        if not self.current_data.empty:
            current_price = self.current_data['Close'].iloc[-1]
        elif self.data_buffer:
            current_price = self.data_buffer[-1]['Close']
        
        # Always create signal info for callback
        signal_info = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'previous_signal': self.last_signal,
            'new_signal': new_signal,
            'current_price': current_price,
            'action': self._signal_to_action(self.last_signal, new_signal)
        }
        
        # Call user callback if provided (either always or only on signal changes)
        if self.signal_callback and (self.always_call_callback or new_signal != 0):
            self.signal_callback(signal_info)
        
        # Only log and track signal changes
        if (new_signal != self.last_signal) & (new_signal != 0):
            self.signal_history.append(signal_info)
            self.last_signal = new_signal
            
            # Log the signal
            self._log_signal(signal_info)
    
    def _signal_to_action(self, old_signal: int, new_signal: int) -> str:
        """Convert signal change to human-readable action."""
        signal_names = {0: 'HOLD', 1: 'SHORT', 2: 'FLAT', 3: 'LONG'}
        
        if (old_signal == new_signal) or (new_signal == 0):
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
        
        self.console.print(f"[{color}]{timestamp} | {self.symbol} {self.interval} | {action} @ ${price:.2f}[/{color}]")
    
    async def _websocket_signal_handler(self):
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
                
                logger.info("Starting message processing loop...")
                self.timeprint("Websocket connection established", color="green")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(message)
                        
                        market_data = self.provider.process_message(data)
                        
                        if market_data:
                            # Update buffer with new candle data
                            candle_closed, candle = self._update_current_candle(market_data)
                            
                            # Show live price updates when candle is open (not closed)
                            if not candle_closed and candle and 'Close' in candle:
                                current_price = candle['Close']
                                self.timeprint(f"Live Price | {self.symbol} | ${current_price:.2f}", color="cyan", end="\r")
                                self.console.file.flush()  # Ensure output is displayed
                            
                            # Only update DataFrame and run strategy on closed candles
                            if candle_closed:
                                self._update_dataframe()
                                # Use the closed candle price for display
                                closed_price = candle['Close']
                                self.timeprint(f"Candle Closed | {self.symbol} | ${closed_price:.2f}", color="blue")
                                new_signal = self._run_strategy()
                                self._handle_signal_change(new_signal)
                                
                                # Create debug plot if enabled
                                if self.debug_plot:
                                    self._create_debug_plot()
                    
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
        self.timeprint(f"Starting Live Trading System for {self.symbol}", color="green")
        self.timeprint(f"Data Source: {self.data_source}", color="blue")
        self.timeprint(f"Interval: {self.interval}", color="blue")
        self.timeprint(f"Strategy: {self.strategy_func.__name__}", color="blue")
        self.timeprint(f"Parameters: {self.strategy_params}", color="blue")
        
        # Initialize provider if not already done
        if not self._initialized:
            self.timeprint("Initializing data provider...", color="yellow")
            await self._initialize_provider()
            self._initialized = True
            self.timeprint("Data provider initialized successfully", color="green")
        
        self.timeprint("Starting WebSocket connection...", color="yellow")
        
        connection_attempts = 0
        max_attempts = 5
        
        while self.running and connection_attempts < max_attempts:
            try:
                connection_attempts += 1
                logger.info(f"WebSocket connection attempt {connection_attempts}/{max_attempts}")
                
                await self._websocket_signal_handler() #where all the signal gen and callbacks happen
                
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
            self.timeprint("Failed to establish stable connection after maximum attempts", color="red")
        else:
            self.timeprint("Trading system stopped", color="yellow")
    
    def stop(self):
        """Stop the live trading system."""
        self.running = False
        self.timeprint("Stopping Live Trading System", color="red")
    
    def get_current_status(self) -> Dict:
        """Get current system status."""
        # Get last price from current_data, buffer, or dataframe
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

    def timeprint(self, text, color="white", end="\n"):
        now = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[{color}]{now} | {text}[/{color}]", end=end)
    
    def _create_debug_plot(self):
        """Create debug plot showing recent candles and signals similar to backtesting.py"""
        if not self.debug_plot or self.current_data.empty or len(self.current_data) < 10:
            return
        
        try:
            # Get recent data for plotting
            plot_data = self.current_data.tail(self.debug_plot_window).copy()
            
            # Run strategy on current data to get all signals
            if self.strategy_func:
                signals = self.strategy_func(self.current_data, **self.strategy_params)
                plot_signals = signals.tail(self.debug_plot_window)
                
                # Convert signals to position for visualization
                position = plot_signals.copy()
                position[plot_signals == 0] = np.nan
                position = position.ffill().fillna(2).astype(int)
                plot_data['Position'] = position
                plot_data['Signal'] = plot_signals
            else:
                plot_data['Position'] = 2
                plot_data['Signal'] = 0
            
            # Create plotly figure
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    increasing_fillcolor='#26FF00',
                    increasing_line_color='#26FF00',
                    decreasing_fillcolor='#ff073a',
                    decreasing_line_color='#ff073a',
                    name='Price'
                )
            )
            
            # Add signal markers
            position_changes = np.diff(plot_data['Position'].values)
            long_entries = []
            short_entries = []
            flats = []
            long_entry_prices = []
            short_entry_prices = []
            flat_prices = []
            
            for i in range(len(position_changes)):
                if position_changes[i] != 0:
                    prev_pos = plot_data['Position'].iloc[i]
                    current_pos = plot_data['Position'].iloc[i+1]
                    if i < len(plot_data) - 1:
                        price_at_signal = plot_data['Close'].iloc[i+1]
                        timestamp = plot_data.index[i+1]
                        
                        if current_pos == 3 and prev_pos != 3:  # Long entry
                            long_entries.append(timestamp)
                            long_entry_prices.append(price_at_signal)
                        elif current_pos == 1 and prev_pos != 1:  # Short entry
                            short_entries.append(timestamp)
                            short_entry_prices.append(price_at_signal)
                        elif current_pos == 2 and prev_pos != 2:  # Exit to flat
                            flats.append(timestamp)
                            flat_prices.append(price_at_signal)
            
            # Add signal markers
            if long_entries:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries,
                        y=long_entry_prices,
                        mode='markers',
                        name='Long Entry',
                        marker=dict(color="#26FF00", size=10, symbol='triangle-up'),
                    )
                )
            
            if short_entries:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries,
                        y=short_entry_prices,
                        mode='markers',
                        name='Short Entry',
                        marker=dict(color='#ff073a', size=10, symbol='triangle-down'),
                    )
                )
            
            if flats:
                fig.add_trace(
                    go.Scatter(
                        x=flats,
                        y=flat_prices,
                        mode='markers',
                        name='Exit to Flat',
                        marker=dict(color='yellow', size=8, symbol='circle'),
                    )
                )
            
            # Update layout
            current_time = datetime.now().strftime("%H:%M:%S")
            latest_signal = plot_data['Signal'].iloc[-1] if len(plot_data) > 0 else 0
            latest_position = plot_data['Position'].iloc[-1] if len(plot_data) > 0 else 2
            latest_price = plot_data['Close'].iloc[-1] if len(plot_data) > 0 else 0
            
            signal_names = {0: 'HOLD', 1: 'SHORT', 2: 'FLAT', 3: 'LONG'}
            
            fig.update_layout(
                title=f'Live Debug Plot - {self.symbol} {self.interval} | {current_time} | Signal: {signal_names.get(latest_signal, "UNKNOWN")} | Position: {signal_names.get(latest_position, "UNKNOWN")} | Price: ${latest_price:.4f}',
                xaxis=dict(
                    title='Time',
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    title="Price ($)",
                ),
                showlegend=True,
                template="plotly_dark",
                height=600,
                width=1200
            )
            
            # Show plot
            fig.show()
            
            # Log debug info
            self.timeprint(f"Debug Plot Updated | Latest Signal: {signal_names.get(latest_signal, 'UNKNOWN')} | Position: {signal_names.get(latest_position, 'UNKNOWN')} | Data Points: {len(plot_data)}", color="magenta")
            
            # Print data quality info
            self._print_data_quality_info(plot_data)
            
        except Exception as e:
            logger.error(f"Error creating debug plot: {e}")
            self.timeprint(f"Debug plot error: {e}", color="red")
    
    def _print_data_quality_info(self, data: pd.DataFrame):
        """Print data quality information for debugging"""
        try:
            if data.empty:
                self.timeprint("Data Quality: EMPTY DATASET", color="red")
                return
            
            # Check for missing values
            missing_data = data.isnull().sum()
            if missing_data.any():
                self.timeprint(f"Data Quality: Missing values found - {missing_data[missing_data > 0].to_dict()}", color="yellow")
            
            # Check for price anomalies
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        self.timeprint(f"Data Quality: Zero/negative prices in {col}", color="red")
                    
                    # Check for extreme price movements
                    pct_change = data[col].pct_change().abs()
                    extreme_moves = pct_change > 0.5  # 50% price change
                    if extreme_moves.any():
                        self.timeprint(f"Data Quality: Extreme price movements detected in {col}", color="yellow")
            
            # Check OHLC logic
            if all(col in data.columns for col in price_cols):
                invalid_ohlc = (data['High'] < data['Low']) | (data['High'] < data['Open']) | (data['High'] < data['Close']) | (data['Low'] > data['Open']) | (data['Low'] > data['Close'])
                if invalid_ohlc.any():
                    self.timeprint(f"Data Quality: Invalid OHLC logic detected in {invalid_ohlc.sum()} rows", color="red")
            
            # Check time consistency
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                if len(time_diffs.unique()) > 3:  # Allow some variation
                    self.timeprint("Data Quality: Inconsistent time intervals detected", color="yellow")
            
            # Overall data summary
            latest_candle = data.iloc[-1]
            self.timeprint(f"Data Quality: {len(data)} candles, Latest: O:{latest_candle.get('Open', 'N/A'):.4f} H:{latest_candle.get('High', 'N/A'):.4f} L:{latest_candle.get('Low', 'N/A'):.4f} C:{latest_candle.get('Close', 'N/A'):.4f}", color="cyan")
            
        except Exception as e:
            self.timeprint(f"Error checking data quality: {e}", color="red")

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

class SimulatedPortfolio:
    """Simulated portfolio for backtesting algorithms in real-time."""
    
    def __init__(self, initial_balance: float = 10000.0, slippage_rate: float = 0.001):
        """
        Initialize simulated portfolio.
        
        Args:
            initial_balance: Starting portfolio value in USD
            slippage_rate: Slippage rate per trade (0.001 = 0.1% price impact)
        """
        self.initial_balance = initial_balance
        self.slippage_rate = slippage_rate
        
        # Portfolio state
        self.cash = initial_balance
        self.position = 0.0  # Current position size (positive = long, negative = short)
        self.entry_price = 0.0
        self.entry_time = None
        
        # Tracking
        self.trade_history = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_slippage_cost = 0.0
        
        # Current state
        self.current_price = 0.0
        self.last_update_time = None
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value including unrealized P&L."""
        if self.position == 0:
            return self.cash
        
        # For long positions: value = cash + (position * current_price)
        # For short positions: value = cash + (position * (2 * entry_price - current_price))
        if self.position > 0:  # Long
            position_value = self.position * current_price
        else:  # Short
            # Short position value = initial_cash_received + (entry_price - current_price) * position_size
            position_value = self.position * current_price  # This is negative for shorts
        
        return self.cash + position_value
    
    def get_position_value(self, current_price: float) -> float:
        """Calculate current position value at market price."""
        return self.position * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        
        if self.position > 0:  # Long position
            return (current_price - self.entry_price) * self.position
        else:  # Short position
            return (self.entry_price - current_price) * abs(self.position)
    
    def _calculate_execution_price(self, market_price: float, is_buy: bool, slippage_rate: Optional[float] = None) -> float:
        """
        Calculate execution price with slippage.
        
        Args:
            market_price: Current market price
            is_buy: True if buying, False if selling
            slippage_rate: Optional slippage rate to override class slippage_rate
            
        Returns:
            Execution price after slippage
        """
        # Use provided slippage_rate or fallback to class slippage_rate
        rate = slippage_rate if slippage_rate is not None else self.slippage_rate
        
        if is_buy:
            # When buying, slippage increases the price (worse execution)
            return market_price * (1 + rate)
        else:
            # When selling, slippage decreases the price (worse execution)
            return market_price * (1 - rate)
    
    def execute_trade(self, signal: int, current_price: float, timestamp: datetime, slippage_bps: Optional[float] = None) -> Dict:
        """
        Execute a simulated trade based on signal.
        
        Args:
            signal: 0=HOLD, 1=SHORT, 2=FLAT, 3=LONG
            current_price: Current market price
            timestamp: Trade timestamp
            slippage_bps: Optional slippage in basis points (1 bps = 0.01%). If None, uses class slippage_rate
            
        Returns:
            Dict with trade details
        """
        self.current_price = current_price
        self.last_update_time = timestamp
        
        # Use provided slippage_bps or fallback to class slippage_rate
        slippage_rate = slippage_bps / 10000 if slippage_bps is not None else self.slippage_rate
        
        # Calculate current portfolio value before trade
        pre_trade_value = self.get_portfolio_value(current_price)
        
        trade_info = {
            'timestamp': timestamp,
            'signal': signal,
            'market_price': current_price,
            'execution_price': current_price,
            'pre_trade_value': pre_trade_value,
            'action': 'HOLD',
            'position_change': 0.0,
            'slippage': 0.0,
            'slippage_cost': 0.0,
            'post_trade_value': pre_trade_value,
            'unrealized_pnl': 0.0,
            'slippage_bps_used': slippage_bps
        }
        
        # Determine action based on signal
        if signal == 0:  # HOLD
            # Do nothing, maintain current position
            trade_info['action'] = 'HOLD'
            
        elif signal == 3:  # LONG
            if self.position <= 0:  # Need to buy
                trade_info['action'] = 'BUY'
                if self.position < 0:  # Close short first
                    trade_info['action'] = 'COVER_AND_BUY'
                    self._close_position(current_price, timestamp, trade_info, slippage_rate)
                
                # Use all available cash for position
                available_cash = self.cash
                
                # Calculate execution price with slippage (buying = higher price)
                execution_price = self._calculate_execution_price(current_price, is_buy=True, slippage_rate=slippage_rate)
                position_size = available_cash / execution_price
                
                # Total cost including slippage
                total_cost = position_size * execution_price
                slippage_cost = position_size * (execution_price - current_price)
                
                if total_cost <= available_cash:
                    self.position = position_size
                    self.cash -= total_cost
                    self.entry_price = execution_price
                    self.entry_time = timestamp
                    self.total_slippage_cost += slippage_cost
                    
                    trade_info['position_change'] = position_size
                    trade_info['execution_price'] = execution_price
                    trade_info['slippage'] = execution_price - current_price
                    trade_info['slippage_cost'] = slippage_cost
                
        elif signal == 1:  # SHORT
            if self.position >= 0:  # Need to sell short
                trade_info['action'] = 'SELL_SHORT'
                if self.position > 0:  # Close long first
                    trade_info['action'] = 'SELL_AND_SHORT'
                    self._close_position(current_price, timestamp, trade_info, slippage_rate)
                
                # Use all available cash as collateral for short position
                available_cash = self.cash
                
                # Calculate execution price with slippage (selling = lower price)
                execution_price = self._calculate_execution_price(current_price, is_buy=False, slippage_rate=slippage_rate)
                position_size = available_cash / current_price  # Size based on market price
                
                # Cash received from short sale (less slippage)
                cash_received = position_size * execution_price
                slippage_cost = position_size * (current_price - execution_price)
                
                if cash_received > 0:
                    self.position = -position_size  # Negative for short
                    self.cash += cash_received
                    self.entry_price = execution_price
                    self.entry_time = timestamp
                    self.total_slippage_cost += slippage_cost
                    
                    trade_info['position_change'] = -position_size
                    trade_info['execution_price'] = execution_price
                    trade_info['slippage'] = current_price - execution_price
                    trade_info['slippage_cost'] = slippage_cost
                
        elif signal == 2:  # FLAT
            if self.position != 0:  # Need to close position
                trade_info['action'] = 'CLOSE_POSITION'
                self._close_position(current_price, timestamp, trade_info, slippage_rate)
        
        # Calculate post-trade values
        trade_info['post_trade_value'] = self.get_portfolio_value(current_price)
        trade_info['unrealized_pnl'] = self.get_unrealized_pnl(current_price)
        
        # Update tracking
        self.portfolio_values.append({
            'timestamp': timestamp,
            'value': trade_info['post_trade_value'],
            'cash': self.cash,
            'position': self.position,
            'position_value': self.get_position_value(current_price),
            'unrealized_pnl': trade_info['unrealized_pnl']
        })
        
        # Log trade if there was an action
        if trade_info['action'] != 'HOLD':
            self.trade_history.append(trade_info)
            self.total_trades += 1
            
            # Track realized P&L when position is closed
            if trade_info['action'] in ['CLOSE_POSITION', 'COVER_AND_BUY', 'SELL_AND_SHORT']:
                realized_pnl = trade_info.get('realized_pnl', 0)
                self.total_pnl += realized_pnl
                if realized_pnl > 0:
                    self.winning_trades += 1
        
        return trade_info
    
    def _close_position(self, current_price: float, timestamp: datetime, trade_info: Dict, slippage_rate: Optional[float] = None):
        """Close current position and update trade info."""
        if self.position == 0:
            return
        
        # Calculate execution price with slippage
        is_buy = self.position < 0  # If short position, we need to buy to cover
        execution_price = self._calculate_execution_price(current_price, is_buy=is_buy, slippage_rate=slippage_rate)
        
        # Calculate slippage cost
        slippage_cost = abs(self.position) * abs(execution_price - current_price)
        self.total_slippage_cost += slippage_cost
        
        # Calculate realized P&L and update cash
        if self.position > 0:  # Closing long position (selling)
            # Sell at execution price (with slippage)
            cash_received = self.position * execution_price
            self.cash += cash_received
            realized_pnl = (execution_price - self.entry_price) * self.position
        else:  # Closing short position (buying to cover)
            # Buy to cover at execution price (with slippage)
            cash_spent = abs(self.position) * execution_price
            self.cash -= cash_spent
            realized_pnl = (self.entry_price - execution_price) * abs(self.position)
        
        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        # Update trade info
        trade_info['slippage_cost'] = trade_info.get('slippage_cost', 0) + slippage_cost
        trade_info['realized_pnl'] = realized_pnl
        trade_info['execution_price'] = execution_price
    
    def get_statistics(self) -> Dict:
        """Get portfolio statistics."""
        if not self.portfolio_values:
            current_value = self.initial_balance
        else:
            current_value = self.portfolio_values[-1]['value']
        
        total_return = ((current_value - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_value': current_value,
            'total_return_pct': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate_pct': win_rate,
            'total_pnl': self.total_pnl,
            'total_slippage_cost': self.total_slippage_cost,
            'current_position': self.position,
            'current_cash': self.cash
        }
    
    def log_portfolio_status(self, console, timestamp: datetime, current_price: float):
        """Log current portfolio status."""
        stats = self.get_statistics()
        portfolio_value = self.get_portfolio_value(current_price)
        unrealized_pnl = self.get_unrealized_pnl(current_price)
        
        # Color based on performance
        if stats.get('total_return_pct', 0) > 0:
            color = "green"
        elif stats.get('total_return_pct', 0) < 0:
            color = "red"
        else:
            color = "yellow"
        
        # Position status
        if self.position > 0:
            position_status = f"LONG {self.position:.4f}"
        elif self.position < 0:
            position_status = f"SHORT {abs(self.position):.4f}"
        else:
            position_status = "FLAT"
        
        console.print(f"[{color}]{timestamp.strftime('%H:%M:%S')} | Portfolio: ${portfolio_value:.2f} | {position_status} | Unrealized P&L: ${unrealized_pnl:.2f} | Total Return: {stats.get('total_return_pct', 0):.2f}% | Slippage: ${stats.get('total_slippage_cost', 0):.2f}[/{color}]")