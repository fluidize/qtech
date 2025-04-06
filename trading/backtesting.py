import time
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from rich import print

from model_tools import *
from indicators import *

class VectorizedBacktesting:
    def __init__(
        self,
        symbol: str,
        instance_name: str = "default",
        initial_capital: float = 10000.0,
        chunks: int = 5,
        interval: str = "5m",
        age_days: int = 10,
    ):
        self.instance_name = instance_name
        self.symbol = symbol
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.initial_capital = initial_capital

        # Data storage
        self.data: pd.DataFrame = None

    def fetch_data(self, kucoin: bool = True):
        self.data = fetch_data(self.symbol, self.chunks, self.interval, self.age_days, kucoin=kucoin)

    def run_strategy(self, strategy_func, **kwargs):
        start_time = time.time()

        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA100'] = self.data['Close'].rolling(window=100).mean()
        self.data['RSI'] = calculate_rsi(self.data, 14)
        
        macd_line, signal_line, histogram = calculate_macd(self.data)
        self.data['MACD_Line'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Hist'] = histogram
        
        self.data['SuperTrend'] = calculate_supertrend(self.data, 7)
        
        self.data['STD'] = self.data['Close'].rolling(window=5).std()
        
        signals = strategy_func(self.data, **kwargs)
        
        self.data['Position'] = signals
        self.data['Position'] = self.data['Position'].replace(-1, 0)  # Set sell signals to 0 (close position)
        
        self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Returns']
        
        self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
        self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Returns']
        
        self.data['Peak'] = self.data['Portfolio_Value'].cummax()
        self.data['Drawdown'] = (self.data['Portfolio_Value'] - self.data['Peak']) / self.data['Peak']
        
        end_time = time.time()
        print(f"ELAPSED: {end_time - start_time:.2f} seconds")

        return self.data

    def get_performance_metrics(self):
        if self.data is None or 'Strategy_Returns' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        trading_days = 365

        total_return = self.data['Cumulative_Returns'].iloc[-1] - 1
        max_drawdown = self.data['Drawdown'].min()
        
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
        excess_returns = self.data['Strategy_Returns'] - daily_rf
        sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
        
        downside_returns = self.data['Strategy_Returns'][self.data['Strategy_Returns'] < 0]
        sortino_ratio = np.sqrt(trading_days) * (self.data['Strategy_Returns'].mean() - daily_rf) / downside_returns.std()

        # Initialize lists to track PnL and entry prices
        pnl_list = []
        entry_prices = []
        
        # Calculate PnL for each position opened
        position_changes = self.data['Position'].diff()
        for idx in range(len(self.data)):
            if position_changes[idx] == 1:  # Entry point
                entry_prices.append(self.data['Close'].iloc[idx])
            elif position_changes[idx] == -1 and entry_prices:  # Exit point
                entry_price = entry_prices.pop(0)  # Get the last entry price
                exit_price = self.data['Close'].iloc[idx]
                pnl = exit_price - entry_price  # Calculate PnL
                pnl_list.append(pnl)

        # Calculate winning trades and other metrics
        winning_trades = sum(1 for pnl in pnl_list if pnl > 0)
        total_trades = len(pnl_list)
        
        average_wins = np.mean([pnl for pnl in pnl_list if pnl > 0]) if winning_trades > 0 else 0
        average_losses = np.mean([pnl for pnl in pnl_list if pnl < 0]) if total_trades > 0 else 0
        RR_ratio = (average_wins / abs(average_losses)) if average_losses < 0 else 0

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        optimal_win_rate = 1 / (RR_ratio + 1) if RR_ratio > 0 else 0

        PT_ratio = (self.data['Strategy_Returns'].sum() / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = (self.data['Strategy_Returns'] > 0).sum() / (self.data['Strategy_Returns'] < 0).sum() if (self.data['Strategy_Returns'] < 0).sum() > 0 else 0

        return {
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Optimal Win Rate': optimal_win_rate,
            'RR Ratio': RR_ratio,
            'PT Ratio': PT_ratio,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Total Trades': total_trades,
        }

    def plot_performance(self, show_graph: bool = True):
        if self.data is None or 'Portfolio_Value' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        fig = go.Figure()
        
        # Add portfolio value
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Portfolio_Value'],
            mode='lines',
            name='Strategy Portfolio Value'
        ))
        
        # Add asset value (normalized to initial capital)
        asset_value = self.initial_capital * (1 + self.data['Returns'].cumsum())
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=asset_value,
            mode='lines',
            name='Asset Value',
            line=dict(dash='dash')
        ))
        
        # Get actual trade signals (changes in position)
        position_changes = self.data['Position'].diff()
        # Buy signals: Any positive change in position
        buy_signals = self.data[position_changes > 0]

        sell_signals = self.data[position_changes < 0]
        
        # Print trade statistics
        print("\nTrade Statistics:")
        print(f"Total Buy Signals: {len(buy_signals)}")
        print(f"Total Sell Signals: {len(sell_signals)}")
        print(f"Total Trades: {len(buy_signals) + len(sell_signals)}")
        
        offset = self.initial_capital * 0.0005
        buy_asset_values = asset_value[buy_signals.index] - offset
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_asset_values,
            mode='markers',
            name='Buy',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ))
        
        sell_asset_values = asset_value[sell_signals.index] + offset
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_asset_values,
            mode='markers',
            name='Sell',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ))
        
        summary = self.get_performance_metrics()
        fig.update_layout(
            title=f'{self.symbol} {self.interval} | TR: {summary["Total Return"]*100:.3f}% | Max DD: {summary["Max Drawdown"]*100:.3f}% | RR: {summary["RR Ratio"]:.3f} | WR: {summary["Win Rate"]*100:.3f}% | Optimal WR: {summary["Optimal Win Rate"]*100:.3f}% | PT: {summary["PT Ratio"]*100:.3f}% | PF: {summary["Profit Factor"]:.3f} | Sharpe: {summary["Sharpe Ratio"]:.3f} | Sortino: {summary["Sortino Ratio"]:.3f} | Trades: {summary["Total Trades"]}',
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            )
        )
        
        if show_graph:
            fig.show()
        
        return fig

    def perfect_strategy(self, data: pd.DataFrame) -> pd.Series:
        future_returns = data['Close'].shift(-1) / data['Close'] - 1
        return (future_returns > 0).astype(int)

    def reversion_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        
        ma_cross = data['MA50'] < data['MA100']
        macd_cross = data['MACD_Line'] > data['MACD_Signal']
        
        position[ma_cross & macd_cross] = 1
        position[~ma_cross & ~macd_cross] = -1
        
        return position

    def rsi_strategy(self, data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
        position = pd.Series(0, index=data.index)
        
        position[data['RSI'] < oversold] = 1
        position[data['RSI'] > overbought] = -1
        
        return position

    def macd_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        
        position[data['MACD_Line'] > data['MACD_Signal']] = 1
        position[data['MACD_Line'] < data['MACD_Signal']] = -1
        
        return position

    def supertrend_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        
        position[data['Close'] > data['SuperTrend']] = 1
        position[data['Close'] < data['SuperTrend']] = -1
        
        return position

    def scalper_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        
        price_change = data['Close'].pct_change()
        position[price_change > 0] = 1
        position[price_change < 0] = -1
        return position

    def custom_scalper_strategy(self, data: pd.DataFrame, 
                              pct_threshold: float = 0.05,
                              std_threshold: float = 0.3) -> pd.Series:
        position = pd.Series(0, index=data.index)
        
        price_change_pct = data['Close'].pct_change()
        sma_short = data['Close'].rolling(window=5).mean()
        sma_long = data['Close'].rolling(window=20).mean()
        bullish = sma_short > sma_long
        
        buy_conditions = (
            (price_change_pct > pct_threshold) &
            (data['Close'] > sma_long) &
            (data['STD'] < std_threshold) &
            bullish
        )
        
        sell_conditions = (
            (price_change_pct < -pct_threshold) &
            ~bullish
        )
        
        position[buy_conditions] = 1
        position[sell_conditions] = -1
        
        return position

    def nn_strategy(self, data: pd.DataFrame, model_path: str = r"trading\model.pth") -> pd.Series:
        from single_pytorch_model import load_model
        
        model = load_model(model_path)
        predictions = model.predict(model, data[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        # Initialize the position series with the first value set to 0
        position = pd.Series(0, index=data.index)
        
        # Set positions based on predictions, starting from the second element
        buy_mask = (predictions-data['Close'][1:]) > 0
        position[1:][buy_mask] = 1  # Buy signal
        position[1:][~buy_mask] = -1  # Sell signal

        return position

if __name__ == "__main__":
    backtest = VectorizedBacktesting(
        symbol="BTC-USDT",
        initial_capital=40.0,
        chunks=29,
        interval="5min",
        age_days=0
    )
    
    backtest.fetch_data(kucoin=True)
    
    backtest.run_strategy(backtest.nn_strategy)
    # metrics = backtest.get_performance_metrics()
    backtest.plot_performance()