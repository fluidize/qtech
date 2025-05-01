import time
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from rich import print
import sys
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import model_tools as mt
import techincal_analysis as ta
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

        self.data: pd.DataFrame = None

    def fetch_data(self, kucoin: bool = True):
        self.data = mt.fetch_data(self.symbol, self.chunks, self.interval, self.age_days, kucoin=kucoin)

    def run_strategy(self, strategy_func, **kwargs):
        """Run a trading strategy on the data"""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        backtesting_start_time = time.time()
        signals = strategy_func(self.data, **kwargs)
        backtesting_end_time = time.time()
        # print(f"[green]BACKTESTING DONE ({backtesting_end_time - backtesting_start_time:.2f} seconds)[/green]")

        self.data['Return'] = self.data['Close'].pct_change()

        self.data['Position'] = signals
        self.data['Position'] = self.data['Position'].replace(-1, 0)  # Set -1 signals to 0 (close position)
        
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
            'Alpha': metrics.get_alpha(self.data['Position'], self.data['Close']),
            'Beta': metrics.get_beta(self.data['Position'], self.data['Close']),
            'Excess_Returns': metrics.get_excess_returns(self.data['Position'], self.data['Close']),
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
        return pd.Series(1, index=data.index)

    def perfect_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Perfect strategy implementation. Use for debugging/benchmarking."""
        future_returns = data['Close'].shift(-1) / data['Close'] - 1
        return (future_returns > 0).astype(int)

    def reversion_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        sma50 = ta.sma(data['Close'], 50)
        sma100 = ta.sma(data['Close'], 100)
        macd, signal = ta.macd(data['Close'])
        
        sma_cross = sma50 < sma100
        macd_cross = macd > signal
        
        position[sma_cross & macd_cross] = 1
        position[~sma_cross & ~macd_cross] = 0
        
        return position

    def rsi_strategy(self, data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
        """RSI strategy implementation.
        
        Args:
            data: DataFrame with price data
            oversold: RSI threshold for oversold conditions (default: 30)
            overbought: RSI threshold for overbought conditions (default: 70)
            
        Returns:
            pd.Series: Position series (1 for long, 0 for flat)
        """
        if oversold >= overbought:
            print(f"Warning: Invalid RSI thresholds - oversold ({oversold}) >= overbought ({overbought})")
            return pd.Series(0, index=data.index)
        
        position = pd.Series(0, index=data.index)
        rsi = ta.rsi(data['Close'])
        
        # Buy when RSI is below oversold threshold
        buy_conditions = (rsi < oversold)
        # Sell when RSI is above overbought threshold
        sell_conditions = (rsi > overbought)
        
        position[buy_conditions] = 1
        position[sell_conditions] = 0
        
        return position

    def macd_strategy(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        position = pd.Series(0, index=data.index)
        macd, signal = ta.macd(data['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        
        position[macd > signal] = 1
        position[macd < signal] = 0
        
        return position

    def supertrend_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        supertrend, supertrend_line = ta.supertrend(data['High'], data['Low'], data['Close'], period=14, multiplier=3)

        position[data['Close'] > supertrend_line] = 1
        position[data['Close'] < supertrend_line] = 0
        
        return position
    
    def psar_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        psar = ta.psar(data['High'], data['Low'], acceleration_start=0.02, acceleration_step=0.02, max_acceleration=0.2)
        
        position[data['Close'] > psar] = 1
        position[data['Close'] < psar] = 0
        
        return position
    
    def smc_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        ...
        
        return position

    def custom_scalper_strategy(self, data: pd.DataFrame) -> pd.Series:
        position = pd.Series(0, index=data.index)
        psar = ta.psar(data['High'], data['Low'], acceleration_start=0.02, acceleration_step=0.02, max_acceleration=0.2)
        supertrend, supertrend_line = ta.supertrend(data['High'], data['Low'], data['Close'], period=14, multiplier=3)
        rsi = ta.rsi(data['Close'], timeperiod=30)

        buy_conditions = (data['Close'] > psar) & (data['Close'] > supertrend_line) & (rsi < 55)
        sell_conditions = (data['Close'] < psar) & (data['Close'] < supertrend_line)
        
        position[buy_conditions] = 1
        position[sell_conditions] = 0
        
        return position

    def nn_strategy(self, data: pd.DataFrame, batch_size: int = 64) -> pd.Series:
        """
        Neural network strategy implementation using one-by-one prediction to avoid any risk of data leakage.
        This approach processes each data point individually, which is slower but guarantees no temporal mixing.
        For a faster batch processing version (where shuffle=False should prevent leakage), see nn_strategy_batch.
        """
        from brains.time_series.single_predictors.classifier_model import load_model
        import model_tools as mt
        
        MODEL_PATH = r"trading\BTC-USDT_15min_20_7features.pth"
        
        position = pd.Series(0, index=data.index)
        
        model = load_model(MODEL_PATH)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        
        features_df, _ = mt.prepare_data_classifier(data[['Open', 'High', 'Low', 'Close', 'Volume']], lagged_length=20)
        
        selected_features = model.selected_features
        
        missing_features = [f for f in selected_features if f not in features_df.columns]
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
            available_features = [f for f in selected_features if f in features_df.columns]
            features_df = features_df[available_features]
        else:
            features_df = features_df[selected_features]
        
        position_values = np.zeros(len(features_df))
        
        with torch.no_grad():
            for i in tqdm(range(len(features_df))):
                x = torch.tensor(features_df.iloc[i].values, dtype=torch.float32).unsqueeze(0).to(model.DEVICE)
                outputs = model(x)
                
                probs = F.softmax(outputs, dim=1)
                
                if probs[0, 1].item() > 0.85:
                    position_values[i] = 1

        position.iloc[len(data)-len(position_values):] = position_values
        
        position = position.ffill().fillna(0)
        
        return position
        
    def nn_strategy_batch(self, data: pd.DataFrame, batch_size: int = 64, confidence_threshold: float = 0.1) -> pd.Series:
        """
        Neural network strategy implementation using batch processing for efficiency.
        
        Note: While this is faster, batching could theoretically introduce subtle data leakage in some cases.
        The shuffle=False setting preserves temporal order in batches, but if you want to be absolutely certain
        there's no leakage, use the nn_strategy method instead which processes one sample at a time.
        """
        from brains.time_series.single_predictors.classifier_model import load_model
        import model_tools as mt
        
        MODEL_PATH = r"trading\BTC-USDT_5min_20_10features.pth"
        
        position = pd.Series(0, index=data.index)
        
        model = load_model(MODEL_PATH)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        selected_features = model.selected_features
        
        features_df, _ = mt.prepare_data_classifier(data[['Open', 'High', 'Low', 'Close', 'Volume']], lagged_length=20)
        
        missing_features = [f for f in selected_features if f not in features_df.columns]
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
            available_features = [f for f in selected_features if f in features_df.columns]
            features_df = features_df[available_features]
        else:
            features_df = features_df[selected_features]
        
        X = torch.tensor(features_df.values, dtype=torch.float32).to(model.DEVICE)
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        probabilities = []
        tqdm_dataloader = tqdm(dataloader, desc="Processing Batches")
        with torch.no_grad():
            for batch in tqdm_dataloader:
                batch_x = batch[0]
                outputs = model(batch_x)
                
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())

        probabilities = np.array(probabilities)
        
        position_values = np.zeros(len(features_df))
        buy_probability = probabilities[:, 1]
        sell_probability = probabilities[:, 0]
        probability_difference = buy_probability - sell_probability #positive = more confident buy | negative = more confident sell
        position_values[probability_difference > confidence_threshold] = 1

        position.iloc[len(data)-len(position_values):] = position_values

        position = position.ffill().fillna(0)
        
        return position

if __name__ == "__main__":
    backtest = VectorizedBacktesting(
        symbol="BTC-USDT",
        initial_capital=39.5,
        chunks=365,
        interval="5min",
        age_days=365
    )
    
    backtest.fetch_data(kucoin=True)
    
    backtest.run_strategy(backtest.nn_strategy_batch)
    print(backtest.get_performance_metrics())
    backtest.plot_performance(advanced=True)