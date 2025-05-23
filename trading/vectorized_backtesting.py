import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from rich import print
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
import os

import model_tools as mt
import technical_analysis as ta
import smc_analysis as smc
import vb_metrics as metrics

import warnings
# warnings.filterwarnings("ignore")
# Buy signals: Any positive change in position
# Sell signals: Any negative change in position
# If .diff() is 0, then no change in position

class VectorizedBacktesting:
    def __init__(
        self,
        instance_name: str = "default",
        initial_capital: float = 10000.0,
    ):
        self.instance_name = instance_name
        self.initial_capital = initial_capital

        self.symbol = None
        self.chunks = None
        self.interval = None
        self.age_days = None
        self.n_days = None
        self.data: pd.DataFrame = pd.DataFrame()

    def fetch_data(self, symbol: str, chunks: int, interval: str, age_days: int, kucoin: bool = True):
        self.symbol = symbol
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.data = mt.fetch_data(symbol, chunks, interval, age_days, kucoin=kucoin)

        oldest = self.data['Datetime'].iloc[0]
        newest = self.data['Datetime'].iloc[-1]
        self.n_days = (newest - oldest).days

    def _signals_to_stateful_position(self, signals: pd.Series) -> pd.Series:
        """Convert raw signals (-1, 0, 1) to a stateful position series (1=long, 0=flat)."""
        position = pd.Series(0, index=signals.index)
        current = 0
        for i, sig in enumerate(signals):
            if sig == 1:
                current = 1
            elif sig == -1:
                current = 0
            elif sig == 0:
                pass  # hold previous
            position.iloc[i] = current
        return position

    def run_strategy(self, strategy_func, **kwargs):
        """Run a trading strategy on the data"""
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")

        raw_signals = strategy_func(self.data, **kwargs)
        position = self._signals_to_stateful_position(raw_signals)

        self.data['Return'] = self.data['Close'].pct_change()
        self.data['Position'] = position
        self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Return']
        self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
        self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Returns']
        self.data['Peak'] = self.data['Portfolio_Value'].cummax()
        self.data['Drawdown'] = (self.data['Portfolio_Value'] - self.data['Peak']) / self.data['Peak']
        return self.data

    def get_performance_metrics(self):
        if self.data is None or 'Position' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        return {
            'Total_Return': metrics.get_total_return(self.data['Position'], self.data['Close']),
            'Alpha': metrics.get_alpha(self.data['Position'], self.data['Close'], n_days=self.n_days),
            'Beta': metrics.get_beta(self.data['Position'], self.data['Close']),
            'Active_Returns': metrics.get_active_returns(self.data['Position'], self.data['Close']),
            'Max_Drawdown': metrics.get_max_drawdown(self.data['Position'], self.data['Close'], self.initial_capital),
            'Sharpe_Ratio': metrics.get_sharpe_ratio(self.data['Position'], self.data['Close']),
            'Sortino_Ratio': metrics.get_sortino_ratio(self.data['Position'], self.data['Close']),
            'Win_Rate': metrics.get_win_rate(self.data['Position'], self.data['Close']),
            'Breakeven_Rate': metrics.get_breakeven_rate(self.data['Position'], self.data['Close']),
            'RR_Ratio': metrics.get_rr_ratio(self.data['Position'], self.data['Close']),
            'PT_Ratio': metrics.get_pt_ratio(self.data['Position'], self.data['Close']),
            'Profit_Factor': metrics.get_profit_factor(self.data['Position'], self.data['Close']),
            'Total_Trades': metrics.get_total_trades(self.data['Position'])
        }

    def plot_performance(self, show_graph: bool = True, advanced: bool = False):
        if self.data is None or 'Portfolio_Value' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        summary = self.get_performance_metrics()

        if not advanced:
            fig = go.Figure()
            
            portfolio_value = self.data['Portfolio_Value'].values
            returns = self.data['Return'].values
            asset_value = self.initial_capital * (1 + self.data['Return']).cumprod()

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=portfolio_value,
                mode='lines',
                name='Portfolio',
            ))
            
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=asset_value,
                mode='lines',
                name='Asset Value'
            ))

            position_changes = np.diff(self.data['Position'].values)
            buy_signals = self.data.index[np.where(position_changes > 0)[0] + 1]
            sell_signals = self.data.index[np.where(position_changes < 0)[0] + 1]

            buy_asset_values = asset_value[np.where(position_changes > 0)[0] + 1]
            sell_asset_values = asset_value[np.where(position_changes < 0)[0] + 1]

            fig.add_trace(go.Scatter(
                x=buy_signals,
                y=buy_asset_values,
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))

            fig.add_trace(go.Scatter(
                x=sell_signals,
                y=sell_asset_values,
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig.update_layout(
                title=f'{self.symbol} {self.chunks} days of {self.interval} | {self.age_days}d old | TR: {summary["Total_Return"]*100:.3f}% | Max DD: {summary["Max_Drawdown"]*100:.3f}% | RR: {summary["RR_Ratio"]:.3f} | WR: {summary["Win_Rate"]*100:.3f}% | BE: {summary["Breakeven_Rate"]*100:.3f}% | PT: {summary["PT_Ratio"]*100:.3f}% | PF: {summary["Profit_Factor"]:.3f} | Sharpe: {summary["Sharpe_Ratio"]:.3f} | Sortino: {summary["Sortino_Ratio"]:.3f} | Trades: {summary["Total_Trades"]}',
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
            start = time.time()
            fig = go.Figure().set_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    f"Equity Curve | TR: {summary['Total_Return']*100:.3f}% | α: {summary['Alpha']*100:.3f}% | β: {summary['Beta']:.3f}", "Drawdown Curve",
                    "Profit and Loss Distribution (%)", "Average Profit per Trade (%)",
                    f"Win Rate | BE: {summary['Breakeven_Rate']*100:.2f}%", f"Sharpe: {summary['Sharpe_Ratio']:.3f} | Sortino: {summary['Sortino_Ratio']:.3f}",
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
            
            asset_value = self.initial_capital * (1 + self.data['Return']).cumprod()
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
            
            position_changes = self.data['Position'].diff().values
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

            end = time.time()
            print(f"[green]Advanced Plotting Done ({end - start:.2f} seconds)[/green]")

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
    
    def hodl_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Hodl strategy implementation. Use for debugging/benchmarking."""
        # Always signal to buy/hold long
        return pd.Series(1, index=data.index)

    def perfect_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Perfect strategy implementation. Use for debugging/benchmarking."""
        future_returns = data['Close'].shift(-1) / data['Close'] - 1
        # 1 for buy/long, -1 for sell/short, 0 for hold
        signals = pd.Series(0, index=data.index)
        signals[future_returns > 0] = 1
        signals[future_returns < 0] = -1
        return signals
    
    def ema_cross_strategy(self, data: pd.DataFrame, fast_period: int = 9, slow_period: int = 26) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        ema_fast = ta.ema(data['Close'], fast_period)
        ema_slow = ta.ema(data['Close'], slow_period)
        signals[ema_fast > ema_slow] = 1
        signals[ema_fast < ema_slow] = -1
        return signals

    def reversion_strategy(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        sma50 = ta.sma(data['Close'], 50)
        sma100 = ta.sma(data['Close'], 100)
        macd, signal = ta.macd(data['Close'])
        sma_cross = sma50 < sma100
        macd_cross = macd > signal
        signals[sma_cross & macd_cross] = 1
        signals[(~sma_cross) & (~macd_cross)] = -1
        return signals

    def rsi_strategy(self, data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
        if oversold >= overbought:
            print(f"Warning: Invalid RSI thresholds - oversold ({oversold}) >= overbought ({overbought})")
            return pd.Series(0, index=data.index)
        signals = pd.Series(0, index=data.index)
        rsi = ta.rsi(data['Close'])
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
        return signals

    def macd_strategy(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        macd, signal = ta.macd(data['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        signals[macd > signal] = 1
        signals[macd < signal] = -1
        return signals

    def supertrend_strategy(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        supertrend, supertrend_line = ta.supertrend(data['High'], data['Low'], data['Close'], period=14, multiplier=3)
        signals[data['Close'] > supertrend_line] = 1
        signals[data['Close'] < supertrend_line] = -1
        return signals
    
    def psar_strategy(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        psar = ta.psar(data['High'], data['Low'], acceleration_start=0.02, acceleration_step=0.02, max_acceleration=0.2)
        signals[data['Close'] > psar] = 1
        signals[data['Close'] < psar] = -1
        return signals
    
    def smc_strategy(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        # ... fill in your SMC logic here, using -1, 0, 1 signals ...
        return signals

    def custom_scalper_strategy(self, data: pd.DataFrame, fast_period: int = 9, slow_period: int = 26, adx_threshold: int = 25, momentum_period: int = 10, momentum_threshold: float = 0.75, wick_threshold: float = 0.5) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        fast_ema = ta.ema(data['Close'], fast_period)
        slow_ema = ta.ema(data['Close'], slow_period)

        adx, plus_di, minus_di = ta.adx(data['High'], data['Low'], data['Close'])
        upper_wick = data['High'] - np.maximum(data['Open'], data['Close'])
        lower_wick = np.minimum(data['Open'], data['Close']) - data['Low']

        momentum = ta.mom(data['Close'], momentum_period)
        
        body_size = np.abs(data['Close'] - data['Open']) + 1e-9
        upper_wick_ratio = upper_wick / body_size
        lower_wick_ratio = lower_wick / body_size

        is_liquidity_sweep_up = (upper_wick_ratio > lower_wick_ratio) & (upper_wick_ratio > wick_threshold)
        is_liquidity_sweep_down = (lower_wick_ratio > upper_wick_ratio) & (lower_wick_ratio > wick_threshold)
        buy_conditions = (fast_ema > slow_ema) & (adx > adx_threshold) & ~is_liquidity_sweep_up | (momentum <= -momentum_threshold)
        sell_conditions = (fast_ema < slow_ema) & (adx > adx_threshold) & ~is_liquidity_sweep_down | (momentum >= momentum_threshold)
        signals[buy_conditions] = 1
        signals[sell_conditions] = -1
        return signals

    def nn_strategy(self, data: pd.DataFrame, batch_size: int = 64, check_consistency: bool = False) -> pd.Series:
        from brains.time_series.single_predictors.classifier_model import load_model
        MODEL_PATH = r"trading\BTC-USDT_1min_5_38features.pth"
        signals = pd.Series(0, index=data.index)
        model = load_model(MODEL_PATH)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        selected_features = model.selected_features
        features_df, _ = mt.prepare_data_classifier(data[['Open', 'High', 'Low', 'Close', 'Volume']], lagged_length=5)
        missing_features = [f for f in selected_features if f not in features_df.columns]
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
            available_features = [f for f in selected_features if f in features_df.columns]
            features_df = features_df[available_features]
        else:
            features_df = features_df[selected_features]
        X = torch.tensor(features_df.values, dtype=torch.float32).to(model.DEVICE)
        X_seq = X.unsqueeze(1)
        dataset = TensorDataset(X_seq)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        position_values = np.zeros(len(features_df))
        with torch.no_grad():
            idx = 0
            for batch in tqdm(dataloader, desc="NN Strategy Batching"):
                batch_x = batch[0]  # (batch, 1, features)
                outputs = model(batch_x)
                probs = F.softmax(outputs, dim=2)  # (batch, 1, classes)
                actions = torch.argmax(probs, dim=2).cpu().numpy().flatten()  # (batch,)
                for i, action in enumerate(actions):
                    if action == 2:
                        position_values[idx + i] = 1
                    elif action == 0:
                        position_values[idx + i] = -1
                    else:
                        position_values[idx + i] = 0
                idx += len(actions)
        signals.iloc[len(data)-len(position_values):] = position_values
        signals = signals.ffill().fillna(0)
        if check_consistency:
            with torch.no_grad():
                single_preds = []
                for i in range(min(32, len(X))):
                    x_single = X[i].unsqueeze(0).unsqueeze(0)  # (1, 1, F)
                    out = model(x_single)
                    prob = F.softmax(out, dim=2)
                    act = torch.argmax(prob, dim=2).item()
                    single_preds.append(act)
                batch_preds = position_values[:len(single_preds)]
                if not np.all(batch_preds == np.array([1 if a==2 else -1 if a==0 else 0 for a in single_preds])):
                    print("[red]WARNING: Batch and single inference results differ![/red]")
        return signals

if __name__ == "__main__":
    backtest = VectorizedBacktesting(
        initial_capital=400
    )
    
    backtest.fetch_data(
        symbol="SOL-USDT",
        chunks=50,
        interval="1min",
        age_days=0
    )
    
    backtest.run_strategy(backtest.custom_scalper_strategy, fast_period=46, slow_period=66, adx_threshold=69.02778758321986, momentum_period=5, momentum_threshold=0.9854203300638632, wick_threshold=0.14776972524351914)
    print(backtest.get_performance_metrics())
    backtest.plot_performance(advanced=False)