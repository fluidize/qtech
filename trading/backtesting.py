import time
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from rich import print
import sys
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import model_tools as mt
from trading.techincal_indicators import *

import warnings
# warnings.filterwarnings("ignore")

# Buy signals: Any positive change in position
# Sell signals: Any negative change in position
# If .diff() is 0, then no change in position

class VectorizedBacktesting:
    def __init__(
        self,
        symbol: str,
        instance_name: str = "default",
        initial_capital: float = 10000.0,
        chunks: int = 5,
        interval: str = "5min",
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
        self.data = mt.fetch_data(self.symbol, self.chunks, self.interval, self.age_days, kucoin=kucoin)

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
        
        risk_free_rate = 0.00  #ts too volatile gng :broken_heart:
        daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
        excess_returns = self.data['Strategy_Returns'] - daily_rf
        sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
        
        downside_returns = self.data['Strategy_Returns'][self.data['Strategy_Returns'] < 0]
        sortino_ratio = np.sqrt(trading_days) * (self.data['Strategy_Returns'].mean() - daily_rf) / downside_returns.std()

        pnl_list = []
        entry_prices = []
        
        position_changes = self.data['Position'].diff()
        for idx in range(len(self.data)):
            if position_changes[idx] == 1:
                entry_prices.append(self.data['Close'].iloc[idx])
            elif position_changes[idx] == -1 and entry_prices:
                entry_price = entry_prices.pop(0)
                exit_price = self.data['Close'].iloc[idx]
                pnl = exit_price - entry_price
                pnl_list.append(pnl)

        winning_trades = sum(1 for pnl in pnl_list if pnl > 0)
        total_trades = len(pnl_list)
        
        average_wins = np.mean([pnl for pnl in pnl_list if pnl > 0]) if winning_trades > 0 else 0
        average_losses = np.mean([pnl for pnl in pnl_list if pnl < 0]) if total_trades > 0 else 0
        RR_ratio = (average_wins / abs(average_losses)) if average_losses < 0 else 0

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        BE_rate = 1 / (RR_ratio + 1) if RR_ratio > 0 else 0

        PT_ratio = (self.data['Strategy_Returns'].sum() / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = (self.data['Strategy_Returns'] > 0).sum() / (self.data['Strategy_Returns'] < 0).sum() if (self.data['Strategy_Returns'] < 0).sum() > 0 else 0

        return {
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Breakeven Rate': BE_rate,
            'RR Ratio': RR_ratio,
            'PT Ratio': PT_ratio,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Total Trades': total_trades,
        }

    def plot_performance(self, show_graph: bool = True, advanced: bool = False):
        if self.data is None or 'Portfolio_Value' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        summary = self.get_performance_metrics()

        if not advanced:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['Portfolio_Value'],
                mode='lines',
                name='Strategy Portfolio Value',
                line=dict(color='green'),
                fillcolor='rgba(60, 179, 113, 0.3)'
            ))
            
            asset_value = self.initial_capital * (1 + self.data['Returns'].cumsum())
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=asset_value,
                mode='lines',
                name='Asset Value'
            ))
            
            position_changes = self.data['Position'].diff()
            buy_signals = self.data[position_changes > 0]
            sell_signals = self.data[position_changes < 0]
            
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
                title=f'{self.symbol} {self.interval} {self.age_days}d old | TR: {summary["Total Return"]*100:.3f}% | Max DD: {summary["Max Drawdown"]*100:.3f}% | RR: {summary["RR Ratio"]:.3f} | WR: {summary["Win Rate"]*100:.3f}% | BE: {summary["Breakeven Rate"]*100:.3f}% | PT: {summary["PT Ratio"]*100:.3f}% | PF: {summary["Profit Factor"]:.3f} | Sharpe: {summary["Sharpe Ratio"]:.3f} | Sortino: {summary["Sortino Ratio"]:.3f} | Trades: {summary["Total Trades"]}',
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
        else: # Advanced Plotting
            fig = go.Figure().set_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    "Equity Curve", "Drawdown Curve",
                    "Profit and Loss Distribution (%)", "Average Profit per Trade (%)",
                    f"Win Rate | BE: {summary['Breakeven Rate']*100:.2f}%", "Sharpe Ratio",
                    "Position Size Distribution", "Risk/Reward Distribution"
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "histogram"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )

            # 1. Equity Curve
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['Portfolio_Value'],
                    mode='lines',
                    name='Strategy Portfolio Value'
                ),
                row=1, col=1
            )
            
            asset_value = self.initial_capital * (1 + self.data['Returns'].cumsum())
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=asset_value,
                    mode='lines',
                    name='Asset Value',
                    line=dict(dash='dash')
                ),
                row=1, col=1
            )

            # 2. Drawdown Curve
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['Drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red')
                ),
                row=1, col=2
            )

            # 3. Profit and Loss Distribution
            pnl_list = []
            entry_prices = []
            
            position_changes = self.data['Position'].diff()
            for idx in range(len(self.data)):
                if position_changes[idx] == 1:
                    entry_prices.append(self.data['Close'].iloc[idx])
                elif position_changes[idx] == -1 and entry_prices:
                    entry_price = entry_prices.pop(0)
                    exit_price = self.data['Close'].iloc[idx]
                    pnl = (exit_price - entry_price) / entry_price * 100  # Convert to percentage
                    pnl_list.append(pnl)

            fig.add_trace(
                go.Histogram(
                    x=pnl_list,
                    name='Trade Returns',
                    nbinsx=250
                ),
                row=2, col=1
            )

            # 4. Average Profit per Trade
            average_profit_series = []
            cumulative_profit = 0
            total_trades = 0
            entry_prices = []

            for idx in range(len(self.data)):
                if position_changes[idx] == 1:
                    entry_prices.append(self.data['Close'].iloc[idx])
                    average_profit_series.append(cumulative_profit / total_trades if total_trades > 0 else 0)
                elif position_changes[idx] == -1 and entry_prices:
                    entry_price = entry_prices.pop(0)
                    exit_price = self.data['Close'].iloc[idx]
                    pnl = (exit_price - entry_price) / entry_price * 100  # Convert to percentage
                    cumulative_profit += pnl
                    total_trades += 1
                    average_profit_series.append(cumulative_profit / total_trades)
                else:
                    average_profit_series.append(cumulative_profit / total_trades if total_trades > 0 else 0)

            avg_profit_series = pd.Series(average_profit_series, index=self.data.index)

            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=avg_profit_series * 100,
                    mode='lines',
                    name='Avg Profit per Trade (%)',
                ),
                row=2, col=2
            )

            mean_profit = np.mean(pnl_list) * 100 if pnl_list else 0
            fig.add_trace(
                go.Scatter(
                    x=[self.data.index[0], self.data.index[-1]],
                    y=[mean_profit, mean_profit],
                    mode='lines',
                    name='Mean Profit',
                    line=dict(dash='dash', color='red')
                ),
                row=2, col=2
            )

            # 5. Win Rate Over Time
            win_rates = []
            total_trades = 0
            winning_trades = 0
            
            for idx in range(len(self.data)):
                if position_changes[idx] == 1:
                    entry_prices.append(self.data['Close'].iloc[idx])
                elif position_changes[idx] == -1 and entry_prices:
                    entry_price = entry_prices.pop(0)
                    exit_price = self.data['Close'].iloc[idx]
                    pnl = (exit_price - entry_price) / entry_price * 100  # Convert to percentage
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1
                    win_rates.append(winning_trades / total_trades if total_trades > 0 else 0)
                else:
                    win_rates.append(winning_trades / total_trades if total_trades > 0 else 0)
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=[rate * 100 for rate in win_rates],
                    mode='lines',
                    name='Win Rate'
                ),
                row=3, col=1
            )

            # 6. Sharpe Ratio Over Time
            rolling_returns = self.data['Strategy_Returns'].rolling(window=30)
            rolling_sharpe = np.sqrt(365) * rolling_returns.mean() / rolling_returns.std()
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=rolling_sharpe,
                    mode='lines',
                    name='Rolling Sharpe'
                ),
                row=3, col=2
            )
            
            mean_sharpe = rolling_sharpe.mean()
            fig.add_trace(
                go.Scatter(
                    x=[self.data.index[0], self.data.index[-1]],
                    y=[mean_sharpe, mean_sharpe],
                    mode='lines',
                    name='Mean Sharpe',
                    line=dict(dash='dash', color='red')
                ),
                row=3, col=2
            )

            # 7. Position Size Distribution
            position_sizes = self.data['Position'].abs()
            fig.add_trace(
                go.Histogram(
                    x=position_sizes,
                    name='Position Sizes',
                    nbinsx=10
                ),
                row=4, col=1
            )

            # 8. Risk/Reward Distribution
            winning_trades = [pnl for pnl in pnl_list if pnl > 0]
            losing_trades = [pnl for pnl in pnl_list if pnl < 0]
            risk_reward = np.abs(np.mean(winning_trades) / np.mean(losing_trades)) if losing_trades else 0
            fig.add_trace(
                go.Histogram(
                    x=pnl_list,
                    name='Risk/Reward',
                    nbinsx=250
                ),
                row=4, col=2
            )

            # Update layout
            fig.update_layout(
                title_text=f"{self.symbol} {self.interval} Performance Analysis",
                showlegend=True,
                template="plotly_dark"
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_yaxes(title_text="Number of Trades", row=2, col=2)
            fig.update_yaxes(title_text="Win Rate (%)", row=3, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=2)
            fig.update_yaxes(title_text="Frequency", row=4, col=1)
            fig.update_yaxes(title_text="Frequency", row=4, col=2)

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

    def simple_scalper_strategy(self, data: pd.DataFrame) -> pd.Series:
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

    def nn_strategy(self, data: pd.DataFrame, model_path: str = r"trading\best_model.pth", batch_size: int = 64) -> pd.Series:
        sys.path.append(r"trading\brains\time_series\single_predictors")
        from classifier_model import load_model
        
        position = pd.Series(0, index=data.index)
        
        model = load_model(model_path)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        # Precompute static features
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        print("a")
        features, _ = mt.prepare_data_classifier(features, train_split=True)
        print("b")
        # Convert features to tensor
        features_tensor = torch.tensor(features.values, dtype=torch.float32)
        features_tensor = features_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Use DataLoader for batching
        dataset = TensorDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        progress_bar = tqdm(total=len(dataloader), desc="Backtesting")
        all_predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch_features = batch[0]
                logits = model(batch_features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = np.argmax(probabilities.cpu(), axis=1)
                all_predictions.extend(predictions)
                progress_bar.update(1)
        progress_bar.close()

        # Assign positions based on predictions with proper position management
        current_position = 0
        for i, prediction in enumerate(all_predictions):
            if prediction == 2:  # Buy signal
                if current_position <= 0:  # Only buy if not already long
                    position.iloc[i] = 1
                    current_position = 1
                else:
                    position.iloc[i] = 0  # Hold existing position
            elif prediction == 0:  # Sell signal
                if current_position >= 0:  # Only sell if not already short
                    position.iloc[i] = -1
                    current_position = -1
                else:
                    position.iloc[i] = 0  # Hold existing position
            else:  # Hold signal
                position.iloc[i] = 0  # Close any open position
                current_position = 0

        return position

if __name__ == "__main__":
    backtest = VectorizedBacktesting(
        symbol="SOL-USDT",
        initial_capital=39.5,
        chunks=1,
        interval="1min",
        age_days=0
    )
    
    backtest.fetch_data(kucoin=True)
    
    backtest.run_strategy(backtest.nn_strategy)
    print(backtest.get_performance_metrics())
    backtest.plot_performance(advanced=False)