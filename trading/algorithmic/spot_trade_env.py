import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from rich import print
from tqdm import tqdm
import requests
import plotly.graph_objects as go


import scipy.stats as stats
# from time_series.close_predictor.close_only import ModelTesting
# f2a184bce09576aff1042c190719fa663b5c3e0f06be78608e5097ee70762292

def _calculate_adx(context, period=14):
    high = context['High']
    low = context['Low']
    close = context['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.Series([max(a, b, c) for a, b, c in zip(tr1, tr2, tr3)])
    
    # Calculate +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(0.0, index=context.index)
    minus_dm = pd.Series(0.0, index=context.index)
    
    plus_dm[up_move > down_move] = up_move[up_move > down_move]
    minus_dm[down_move > up_move] = down_move[down_move > up_move]
    
    # Calculate smoothed TR, +DM, and -DM
    smoothed_tr = tr.rolling(window=period).mean()
    smoothed_plus_dm = plus_dm.rolling(window=period).mean()
    smoothed_minus_dm = minus_dm.rolling(window=period).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX (smoothed DX)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def _calculate_rsi(context, period=14):
    delta_p = context['Close'].diff()
    gain = delta_p.where(delta_p > 0, 0)
    loss = -delta_p.where(delta_p < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi

def _calculate_macd(context, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = context['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = context['Close'].ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def _calculate_atr(context, period=14):
    high = context['High']
    low = context['Low']
    close = context['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.Series([max(a, b, c) for a, b, c in zip(tr1, tr2, tr3)])

    atr = tr.rolling(window=period).mean()
    return atr

def _calculate_supertrend(context, period=14):
    atr = _calculate_atr(context, period=period)
    multiplier = 3.0
    
    hl2 = (context['High'] + context['Low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()
    supertrend = pd.Series(index=context.index, dtype=float)
    
    supertrend.iloc[0] = final_upperband.iloc[0]
    
    close = context['Close']
    prev_upperband = final_upperband.shift(1)
    prev_lowerband = final_lowerband.shift(1)
    prev_supertrend = supertrend.shift(1)
    
    mask_upper = (basic_upperband < prev_upperband) | (close.shift(1) > prev_upperband)
    final_upperband.loc[mask_upper] = basic_upperband.loc[mask_upper]
    final_upperband.loc[~mask_upper] = prev_upperband.loc[~mask_upper]
    
    mask_lower = (basic_lowerband > prev_lowerband) | (close.shift(1) < prev_lowerband)
    final_lowerband.loc[mask_lower] = basic_lowerband.loc[mask_lower]
    final_lowerband.loc[~mask_lower] = prev_lowerband.loc[~mask_lower]
    
    mask_supertrend = close > final_upperband
    supertrend.loc[mask_supertrend] = final_lowerband.loc[mask_supertrend]
    supertrend.loc[~mask_supertrend] = final_upperband.loc[~mask_supertrend]
    return supertrend

def _calculate_std(context, period=20):
    raw_std = context['Close'].rolling(window=period).std()
    std = (raw_std - raw_std.min())/(raw_std.max() - raw_std.min()) #normalize std or else different symbols will have different thresholds
    return std

@dataclass
class Position:
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def profit_loss(self) -> float:
        return self.quantity * (self.current_price - self.average_price)
    
    @property
    def profit_loss_pct(self) -> float:
        return ((self.current_price - self.average_price) / self.average_price) * 100

class Portfolio:
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[float] = []
        self.pnl_history: List[float] = []
        self.pct_pnl_history: List[float] = []
        self.enable_fee = False
        self.fee_percentage = 0.001 #maker and taker fee

    def buy(self, symbol: str, quantity: float, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        cost = round(quantity * price, 4) #roundoff buy errors
        quantity = quantity * (1-self.fee_percentage) if self.enable_fee else quantity

        if cost > self.cash:
            if verbose:
                print(f"[red]Insufficient funds for buy order. Required: ${cost}, Available: ${self.cash}[/red]")
            return False
            
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            total_cost = (pos.quantity * pos.average_price) + (quantity * price)
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                average_price=total_cost / total_quantity,
                current_price=price
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price
            )
            
        self.cash -= cost
        self.trade_history.append({
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'cash_remaining': self.cash,
            'timestamp': timestamp
        })
        if verbose:
            print(f"[green]Bought {quantity} shares of {symbol} at {price:.2f} for {cost:.2f}[/green]")
        return True
    
    def sell(self, symbol: str, quantity: float, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        if symbol not in self.positions:
            if verbose:
                print(f"[red]No position found for {symbol}[/red]")
            return False
            
        pos = self.positions[symbol]
        if quantity > pos.quantity:
            if verbose:
                print(f"[red]Insufficient shares for sell order. Required: {quantity}, Available: {pos.quantity}[/red]")
            return False
            
        proceeds = round(quantity * price * (1-self.fee_percentage), 4) if self.enable_fee else round(quantity * price, 4)
        self.cash += proceeds
        
        if quantity == pos.quantity:
            self.trade_history.append({
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'cash_remaining': self.cash,
            'timestamp': timestamp,
            'PnL': self.positions[symbol].profit_loss
            })
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=pos.quantity - quantity,
                average_price=pos.average_price,
                current_price=price
            )
        if verbose:
            print(f"[red]Sold {quantity} shares of {symbol} at {price:.2f} for {proceeds:.2f}[/red]")
        return True
    
    def update_positions(self, current_prices: Dict[str, pd.Series]):
        for symbol, price_data in current_prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                # Extract Close price from the OHLCV data
                current_price = float(price_data['Close'])
                self.positions[symbol] = Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    average_price=pos.average_price,
                    current_price=current_price
                )
    
    @property
    def total_value(self) -> float:
        position_value = sum(pos.value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def total_profit_loss(self) -> float:
        return self.total_value - self.initial_capital
    
    @property
    def total_profit_loss_pct(self) -> float:
        return (self.total_profit_loss / self.initial_capital) * 100
    
    def get_position_summary(self) -> str:
        if not self.positions:
            return "No open positions"
            
        summary = "Current Positions:\n"
        for symbol, pos in self.positions.items():
            summary += f"{symbol}: {pos.quantity:.4f} shares @ ${pos.average_price:.2f} "
            summary += f"(Current: ${pos.current_price:.2f}, P/L: ${pos.profit_loss:.2f} [{pos.profit_loss_pct:.2f}%])\n"
        return summary

    def buy_max(self, symbol: str, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        if self.cash <= 0:
            return False
        
        quantity = self.cash / price
        return self.buy(symbol, quantity, price, timestamp, verbose)

    def sell_max(self, symbol: str, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        if symbol not in self.positions:
            return False
        
        quantity = self.positions[symbol].quantity
        return self.sell(symbol, quantity, price, timestamp, verbose)

class TradingEnvironment:
    def __init__(self, symbols: List[str], instance_name: str = 'default', initial_capital: float = 10000.0, chunks: int = 5, interval: str = '5m', age_days: int = 10):
        self.instance_name = instance_name
        self.symbols = symbols
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.portfolio = Portfolio(initial_capital)
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}  # Full historical data for each symbol
        self.context_length = 200  # Number of historical points needed for indicators
        
        # Trading state
        self.current_index = 0  # Current position in the data
        self.is_initialized = False  # Flag to track if context is properly initialized
        
        # Context storage
        self.context: Dict[str, pd.DataFrame] = {}  # Current context window for each symbol
        self.extended_context = False  # Whether to keep growing context or maintain fixed size

    def fetch_data(self, yfinance: bool = False):
        print("[green]DOWNLOADING DATA[/green]")
        if yfinance:
            for symbol in self.symbols:
                data = pd.DataFrame()
                times = []
                for x in range(self.chunks):
                    chunksize = 1
                    start_date = datetime.now() - timedelta(days=chunksize) - timedelta(days=chunksize*x) - timedelta(days=self.age_days)
                    end_date = datetime.now() - timedelta(days=chunksize*x) - timedelta(days=self.age_days)
                    temp_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=self.interval, progress=False)
                    data = pd.concat([data, temp_data])
                    times.append(start_date)
                    times.append(end_date)
                
                earliest = min(times)
                latest = max(times)
                difference = latest - earliest
                print(f"{symbol} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")

                data.sort_index(inplace=True)
                data.columns = data.columns.droplevel(1)
                data.reset_index(inplace=True)
                data.rename(columns={'index': 'Datetime'}, inplace=True)
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
                self.data[symbol] = pd.DataFrame(data)  
                self.context[symbol] = self.data[symbol].iloc[:self.context_length].copy()

            min_length = min(len(df) for df in self.data.values())
            for symbol in self.symbols:
                self.data[symbol] = self.data[symbol].iloc[-min_length:]
        else:
            #KUCOIN API | https://www.kucoin.com/docs/rest/spot-trading/market-data/get-klines
            #MAX 1500 BAR PER REQ 1m GRANULARITY
            for symbol in self.symbols:
                data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
                times = []
                
                for x in range(self.chunks):
                    chunksize = 1440  # 1d of 1m data
                    end_time = datetime.now() - timedelta(minutes=chunksize*x)
                    start_time = end_time - timedelta(minutes=chunksize)
                    
                    params = {
                        "symbol": "SOL-USDT",
                        "type": "1min",
                        "startAt": str(int(start_time.timestamp())),
                        "endAt": str(int(end_time.timestamp()))
                    }
                    
                    request = requests.get("https://api.kucoin.com/api/v1/market/candles", params=params).json()
                    request_data = request["data"]  # list of lists
                    
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
                    data = pd.concat([data, temp_data])
                    times.append(start_time)
                    times.append(end_time)
                
                earliest = min(times)
                latest = max(times)
                difference = latest - earliest
                print(f"{symbol} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")
                
                data["Datetime"] = pd.to_datetime(pd.to_numeric(data['Datetime']), unit='s')
                data.sort_values('Datetime', inplace=True)
                data.reset_index(drop=True, inplace=True)
                
                self.data[symbol] = data
            
        print(f"Fetched {len(data)} data points for {symbol}")

    def initialize_context(self):
        """Initialize the context windows for all symbols."""
        if not self.data:
            raise ValueError("No data available. Call fetch_data() first.")
            
        for symbol in self.symbols:
            if len(self.data[symbol]) < self.context_length:
                raise ValueError(f"Insufficient data for {symbol}. Need at least {self.context_length} points.")
            
            # Initialize context with first context_length rows
            self.context[symbol] = self.data[symbol].iloc[:self.context_length].copy()
        
        self.current_index = self.context_length
        self.is_initialized = True
        print("Context initialized successfully")

    def update_context(self):
        """Update the context windows with new data."""
        if not self.is_initialized:
            raise ValueError("Context not initialized. Call initialize_context() first.")
            
        for symbol in self.symbols:
            if self.current_index >= len(self.data[symbol]):
                continue
                
            current_data = self.data[symbol].iloc[self.current_index]
            
            if self.extended_context:
                # Growing context mode - append new data
                self.context[symbol] = pd.concat([
                    self.context[symbol],
                    pd.DataFrame([current_data])
                ]).reset_index(drop=True)
            else:
                # Fixed-size context mode - slide window
                self.context[symbol] = pd.concat([
                    self.context[symbol].iloc[1:],
                    pd.DataFrame([current_data])
                ]).reset_index(drop=True)

    def step(self) -> bool:
        """Advance the environment by one step."""
        if not self.is_initialized:
            raise ValueError("Context not initialized. Call initialize_context() first.")
            
        # Check if we've reached the end of the data
        if self.current_index >= len(self.data[self.symbols[0]]) - 1:
            return False
            
        # Update context with new data
        self.update_context()
        
        # Update portfolio with current prices
        current_prices = self.get_current_prices()
        self.portfolio.update_positions(current_prices)
        self.portfolio.pnl_history.append(self.portfolio.total_profit_loss)
        self.portfolio.pct_pnl_history.append(self.portfolio.total_profit_loss_pct)
        
        # Advance to next index
        self.current_index += 1
        return True
    
    def reset(self):
        """Reset the environment to its initial state."""
        self.initialize_context()
        self.portfolio = Portfolio(self.portfolio.initial_capital)
        print("Environment reset successfully")
    
    def get_current_prices(self) -> Dict[str, pd.Series]:
        """Get current prices for all symbols."""
        return {
            symbol: self.data[symbol].iloc[self.current_index]
            for symbol in self.symbols
        }
    
    def get_current_timestamp(self) -> datetime:
        """Get the current timestamp."""
        return self.data[self.symbols[0]].iloc[self.current_index]['Datetime']
    
    def get_state(self) -> Dict:
        """Get the current state of the environment."""
        return {
            'timestamp': self.get_current_timestamp(),
            'prices': self.get_current_prices(),
            'context': self.context,
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.positions,
            'trade_history': self.portfolio.trade_history
        }

    def get_summary(self, verbose: bool = False) -> str:
        max_drawdown = (pd.Series(self.portfolio.pct_pnl_history) - pd.Series(self.portfolio.pct_pnl_history).cummax()).min()

        closed_trades = list(filter(lambda x: x['action'] == 'SELL', self.portfolio.trade_history))
        gains = list(filter(lambda x: x > 0, [trade['PnL'] for trade in closed_trades]))
        losses = list(filter(lambda x: x < 0, [trade['PnL'] for trade in closed_trades]))
        
        if len(gains) == 0 or len(losses) == 0:
            profit_factor = np.nan
            RR_ratio = np.nan
            win_rate = np.nan
            optimal_wr = np.nan
        else:
            profit_factor = sum(gains) / abs(sum(losses))
            RR_ratio = (sum(gains) / len(gains)) / abs(sum(losses) / len(losses))
            win_rate = len(gains) / (len(gains) + len(losses))
            optimal_wr = 1 / (RR_ratio + 1)

        prices = np.array([self.portfolio.trade_history[i]['price'] for i in range(len(self.portfolio.trade_history))])
        quantities = np.array([self.portfolio.trade_history[i]['quantity'] for i in range(len(self.portfolio.trade_history))])
        total_volume = sum(prices * quantities)
        total_trades = len(self.portfolio.trade_history)
        PT_ratio = self.portfolio.total_profit_loss_pct / total_trades #profit% to trade ratio

        if verbose:
            print(f"{self.symbols[0]} {self.interval} | PnL: {self.portfolio.total_profit_loss_pct:.2f}% | Max DD: {max_drawdown:.2f}% | PF: {profit_factor:.2f} | RR Ratio:{RR_ratio:.2f} | P%/T Ratio: {PT_ratio:.2f}% | WR: {win_rate:.2f} | Optimal WR: {optimal_wr:.2f} | Total Volume: {total_volume:.2f} | Total Trades: {total_trades} | Gains: {len(gains)} | Losses: {len(losses)}")
        output_dict = {
            'PnL': self.portfolio.total_profit_loss_pct,
            'Max DD': max_drawdown,
            'PF': profit_factor,
            'RR Ratio': RR_ratio,
            'PT Ratio': PT_ratio,
            'WR': win_rate,
            'Optimal WR': optimal_wr,
            'Total Volume': total_volume,
            'Total Trades': total_trades,
            'Gains': len(gains),
            'Losses': len(losses)
        }
        return output_dict
    
    def execute_dict(self, signals: Dict[str, bool]) -> None:
        """
        Execute trades based on a dictionary of buy/sell signals.
        
        Args:
            signals (Dict[str, bool]): Dictionary containing 'buy' and 'sell' boolean signals
        """
        current_prices = self.get_current_prices()
        current_symbol = self.get_current_symbol()
        current_price = current_prices[current_symbol]['Close']
        
        if signals.get('buy', False):
            self.portfolio.buy_max(current_symbol, current_price, self.get_current_timestamp())
        elif signals.get('sell', False):
            self.portfolio.sell_max(current_symbol, current_price, self.get_current_timestamp())

    def create_performance_plot(self, show_graph: bool = False):
        fig = go.Figure()
        
        # Calculate strategy performance line
        # Start from context_length since that's when we begin trading
        strategy_data = self.data[self.symbols[0]].iloc[self.context_length:]
        strategy_data = strategy_data.reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=strategy_data['Datetime'], 
            y=self.portfolio.pct_pnl_history, 
            mode='lines', 
            name='Strategy %'
        ))
        
        # Calculate price percentage change line
        price_data = self.data[self.symbols[0]]
        start_price = price_data.iloc[0]['Close']
        start_pct_change = ((price_data['Close'] - start_price) / start_price) * 100
        fig.add_trace(go.Scatter(x=price_data['Datetime'], y=start_pct_change, mode='lines', name='Price %', line=dict(color='orange')))
        
        # Add buy/sell markers
        buy_x = []
        buy_y = []
        sell_x = []
        sell_y = []
        
        for trade in self.portfolio.trade_history:
            trade_time = pd.to_datetime(trade['timestamp'])
            
            idx = price_data['Datetime'].searchsorted(trade_time)
                
            trade_price = price_data.iloc[idx]['Close']
            trade_pct_change = ((trade_price - start_price) / start_price) * 100
                
            if trade['action'] == 'BUY':
                buy_x.append(trade_time)
                buy_y.append(trade_pct_change)
            else:  # SELL
                sell_x.append(trade_time)
                sell_y.append(trade_pct_change)
        
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='Buy', 
                               marker=dict(color='green', size=5, symbol='circle')))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='Sell', 
                               marker=dict(color='red', size=5, symbol='circle')))
        
        summary = self.get_summary()
        formatted_summary = f"PnL: {summary['PnL']:.2f}% | Max DD: {summary['Max DD']:.2f}% | PF: {summary['PF']:.2f} | RR Ratio:{summary['RR Ratio']:.2f} | P%/T Ratio: {summary['PT Ratio']:.2f}% | WR: {summary['WR']:.2f} | Optimal WR: {summary['Optimal WR']:.2f} | Total Volume: {summary['Total Volume']:.2f} | Total Trades: {summary['Total Trades']} | Gains: {summary['Gains']} | Losses: {summary['Losses']}"

        # Update layout with proper margins and height
        fig.update_layout(
            title=f"Portfolio Performance {self.instance_name} | {formatted_summary}",
            xaxis_title='Time',
            yaxis_title='Profit/Loss (%)',
            height=800,  # Set a fixed height
            margin=dict(l=50, r=50, t=100, b=50)  # Add margins
        )
        
        if show_graph:
            fig.show()
        return fig

class BacktestEnvironment:
    def __init__(self):
        self.environments = {}
        self.current_symbol = None
    
    def _add_environment(self, environment: TradingEnvironment):
        self.environments[environment.instance_name] = environment
        self.current_symbol = environment.symbols[0]
    
    def add_strategy_environments(self, strategies: List):
        for strategy in strategies:
            default_env = TradingEnvironment(symbols=['SOL-USD'],
                                             instance_name=strategy.__name__, 
                                             initial_capital=40, 
                                             chunks=29, 
                                             interval='1m', 
                                             age_days=0
                                             ) #set env defaults here
            self._add_environment(default_env)
    
    def Perfect(self, env, context, current_ohlcv):
        #perfect strategy
        index = env.current_index
        try:
            if context['Close'].iloc[-1] < env.data[env.symbols[0]]['Close'].iloc[index+1]:
                env.portfolio.buy_max(self.current_symbol, current_ohlcv['Close'], env.get_current_timestamp())
            elif context['Close'].iloc[-1] > env.data[env.symbols[0]]['Close'].iloc[index+1]:
                env.portfolio.sell_max(self.current_symbol, current_ohlcv['Close'], env.get_current_timestamp())
        except:
            pass

    def Reversion(self, env: TradingEnvironment, context, current_ohlcv):
        ma50 = context['Close'].rolling(window=50).mean().iloc[-1]
        ma100 = context['Close'].rolling(window=100).mean().iloc[-1]
        
        current_close = current_ohlcv['Close']
        macd_line, signal_line, histogram = _calculate_macd(context)

        current_macd_line = macd_line.iloc[-1]
        current_signal_line = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]

        buy_conditions = [ #entry bearish
            ma50 < ma100,
            current_macd_line > current_signal_line
        ]
        sell_conditions = [
            ma50 > ma100,
            current_macd_line < current_signal_line
        ]

        if all(buy_conditions):
            env.portfolio.buy_max(self.current_symbol, current_close, env.get_current_timestamp())
        elif all(sell_conditions):
            env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())

    def RSI(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']

        rsi = _calculate_rsi(context, 28)
        current_rsi = rsi.iloc[-1]

        if current_rsi < 30:
            env.portfolio.buy(self.current_symbol, 0.1, current_close, env.get_current_timestamp())
        elif current_rsi > 70:
            env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())

    def MACD(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']

        macd_line, signal_line, histogram = _calculate_macd(context)
        current_macd_line = macd_line.iloc[-1]
        current_signal_line = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        if current_macd_line > current_signal_line:
            env.portfolio.buy_max(self.current_symbol, current_close, env.get_current_timestamp())
        elif current_macd_line < current_signal_line:
            env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())

    def SuperTrend(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']
        supertrend = _calculate_supertrend(context, 7)
        current_supertrend = supertrend.iloc[-1]
        prev_close = context['Close'].iloc[-2]
        prev_supertrend = supertrend.iloc[-2]
        
        if current_close > current_supertrend and prev_close <= prev_supertrend:
            env.portfolio.buy_max(self.current_symbol, current_close, env.get_current_timestamp())

        elif current_close < current_supertrend and prev_close >= prev_supertrend:
            env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())

    def Simple_Scalper(self, env: TradingEnvironment, context, current_ohlcv):
        """Basic strategy that buys when current close is higher that prev close. Performs well in 1m."""
        current_close = context['Close'].iloc[-1]
        prev_close = context['Close'].iloc[-2]
        
        buy_conditions = [current_close > prev_close]
        sell_conditions = [current_close < prev_close]

        if all(buy_conditions):
            env.portfolio.buy_max(self.current_symbol, current_close, env.get_current_timestamp())
        elif all(sell_conditions):
            env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())

    def Custom_Scalper(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = context['Close'].iloc[-1]
        prev_close = context['Close'].iloc[-2]
        std = _calculate_std(context, 5)
        current_std = std.iloc[-1]
        price_change_pct = ((current_close - prev_close) / prev_close) * 100

        sma_short = context['Close'].rolling(window=5).mean().iloc[-1]
        sma_long = context['Close'].rolling(window=20).mean().iloc[-1]
        bullish = sma_short > sma_long

        std_threshold = 0.3
        pct_threshold = 0.05
        
        buy_conditions = [
            price_change_pct > pct_threshold,
            current_close > sma_long,
            current_std < std_threshold,
            bullish
        ]
        
        sell_conditions = [
            price_change_pct < -pct_threshold,
            not bullish
        ]

        if all(buy_conditions):
            env.portfolio.buy_max(self.current_symbol, current_close, env.get_current_timestamp())
        elif all(sell_conditions):
            env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())
    def NN(self, env: TradingEnvironment, context, current_ohlcv):
        ...

    def run(self, strategies: List, verbose: bool = False, show_graph: bool = False):
        self.add_strategy_environments(strategies)
        for env in self.environments.values():
            env.fetch_data(yfinance=True)
            # env.create_ohlcv_chart()
            env.initialize_context()
        print("Starting Backtest")
        total_steps = 0
        for env in self.environments.values():
            total_steps += len(env.data[env.symbols[0]]) - env.context_length - 1

        progress_bar = tqdm(total=total_steps)
        while all(env.step() for env in self.environments.values()):
            for strategy, env in zip(strategies, self.environments.values()):
                current_state = env.get_state()
                context = current_state['context'][self.current_symbol]
                current_ohlcv = current_state['prices'][self.current_symbol]
                strategy(env, context, current_ohlcv)
                progress_bar.update(1)
        progress_bar.close()

        output_dict = {}
        for env in self.environments.values():
            if verbose:
                print("\nFinal Portfolio State:")
                print(f"Cash: {env.portfolio.cash:.2f}")
                print(f"Total Value: {env.portfolio.total_value:.2f}")
                print(env.get_summary(verbose=True))
            if show_graph:
                env.create_performance_plot(show_graph=True)
            output_dict[env.instance_name] = env.get_summary()

        return output_dict

backtest = BacktestEnvironment()
backtest.run([backtest.Custom_Scalper], show_graph=True)
