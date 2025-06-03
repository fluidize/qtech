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

sys.path.append("trading")

import model_tools as mt
import technical_analysis as ta
import smc_analysis as smc

import vb_metrics as metrics

class VectorizedBacktesting:
    def __init__(
        self,
        instance_name: str = "default",
        initial_capital: float = 10000.0,
        slippage_pct: float = 0.001,  # 0.1% slippage per trade
        commission_pct: float = 0.001,  # 0.1% commission per trade
        reinvest: bool = True,  # True for compound returns, False for linear returns
    ):
        self.instance_name = instance_name
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.reinvest = reinvest

        self.symbol = None
        self.chunks = None
        self.interval = None
        self.age_days = None
        self.n_days = None
        self.data: pd.DataFrame = pd.DataFrame()

    def fetch_data(self, symbol: str = "None", chunks: int = "None", interval: str = "None", age_days: int = "None", kucoin: bool = True):
        self.symbol = symbol
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        if any([symbol, chunks, interval, age_days]) == "None":
            pass
        else:
            self.data = mt.fetch_data(symbol, chunks, interval, age_days, kucoin=kucoin)
            # self._validate_data_quality()
            self.set_n_days()

            return self.data

    def load_data(self, data: pd.DataFrame):
        self.data = data
        # self._validate_data_quality()
        self.set_n_days()

    def _validate_data_quality(self):
        """Validate data quality and handle common issues."""
        if self.data.empty:
            raise ValueError("Data is empty")
        
        # Check for missing data
        missing_data = self.data.isnull().sum()
        if missing_data.any():
            print(f"[yellow]Warning: Missing data found: {missing_data[missing_data > 0].to_dict()}[/yellow]")
            
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            print(f"[yellow]Warning: {duplicates} duplicate rows found and removed[/yellow]")
            self.data = self.data.drop_duplicates()
            
        # Check for data gaps (if datetime index)
        if 'Datetime' in self.data.columns:
            time_diff = self.data['Datetime'].diff()
            expected_interval = time_diff.mode()[0] if not time_diff.mode().empty else None
            if expected_interval:
                large_gaps = time_diff > expected_interval * 2
                if large_gaps.any():
                    print(f"[yellow]Warning: {large_gaps.sum()} large time gaps detected[/yellow]")

    def _apply_trading_costs(self, returns: pd.Series, position_changes: pd.Series) -> pd.Series:
        """Apply slippage and commissions to returns."""
        if self.slippage_pct == 0 and self.commission_pct == 0:
            return returns
            
        costs = pd.Series(0.0, index=returns.index)
        trade_mask = position_changes != 0
        
        # Apply costs only when position changes (trades occur)
        costs[trade_mask] = self.slippage_pct + self.commission_pct
        
        # Subtract costs from returns
        adjusted_returns = returns - costs
        return adjusted_returns

    def set_n_days(self):
        oldest = self.data['Datetime'].iloc[0]
        newest = self.data['Datetime'].iloc[-1]
        self.n_days = (newest - oldest).days

    def _signals_to_stateful_position(self, signals: pd.Series) -> pd.Series:
        """Convert raw signals to a stateful position series (0=hold, 1=short, 2=flat, 3=long)."""
        position = signals.copy()
        position[signals == 0] = np.nan
        position = position.ffill().fillna(2).astype(int) #forward fill hold signals, default to flat at start
        return position

    def run_strategy(self, strategy_func, verbose: bool = False, **kwargs):
        """
        Run a trading strategy on the data with realistic execution at open prices.
        
        Supports two return calculation modes:
        - Compound (reinvest=True): Returns compound over time. Profits increase position size.
          Example: 10% + 10% = 21% total return (1.1 * 1.1 - 1)
        - Linear (reinvest=False): Each period's return applied to initial capital only.
          Example: 10% + 10% = 20% total return (additive)
          
        Args:
            strategy_func: Strategy function that returns position signals
            verbose: Print execution details
            **kwargs: Parameters passed to strategy function
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")

        start_time = time.time()

        # Generate signals based on available data (no hindsight bias)
        raw_signals = strategy_func(self.data, **kwargs)
        position = self._signals_to_stateful_position(raw_signals)

        # Calculate open-to-open returns (realistic execution)
        # Return[t] = (Open[t+1] - Open[t]) / Open[t]
        # This represents the return from executing at Open[t] to Open[t+1]
        self.data['Return'] = self.data['Open'].shift(-1) / self.data['Open'] - 1
        self.data['Position'] = position
        
        # Calculate position changes for cost application
        position_changes = position.diff().fillna(0)
        
        # REALISTIC TIMING: Signal at close of bar t-1 -> Execute at open of bar t -> Earn return from open t to open t+1
        # So: Strategy_Return[t] = Position[t] * Return[t]
        # Where Return[t] is the return from Open[t] to Open[t+1]
        # And Position[t] is the position taken at Open[t] based on signal at Close[t-1]
        base_returns = metrics.stateful_position_to_multiplier(position) * self.data['Return']
        
        # Apply trading costs (slippage + commissions)
        if self.slippage_pct == 0 and self.commission_pct == 0:
            self.data['Strategy_Returns'] = base_returns
        else:
            self.data['Strategy_Returns'] = self._apply_trading_costs(base_returns, position_changes)
        
        # Calculate cumulative returns and portfolio value based on reinvestment setting
        if self.reinvest:
            # Compound returns (reinvestment) - default behavior
            self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
            self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Returns']
        else:
            # Linear returns (no reinvestment) - each period's return applied to initial capital only
            self.data['Linear_Profit'] = (self.initial_capital * self.data['Strategy_Returns']).cumsum()
            self.data['Portfolio_Value'] = self.initial_capital + self.data['Linear_Profit']
            self.data['Cumulative_Returns'] = self.data['Portfolio_Value'] / self.initial_capital
        
        self.data['Peak'] = self.data['Portfolio_Value'].cummax()
        self.data['Drawdown'] = (self.data['Portfolio_Value'] - self.data['Peak']) / self.data['Peak']

        end_time = time.time()
        if verbose:
            print(f"[green]Strategy execution time: {end_time - start_time:.2f} seconds[/green]")
            if self.slippage_pct > 0 or self.commission_pct > 0:
                print(f"[blue]Applied {self.slippage_pct*100:.3f}% slippage + {self.commission_pct*100:.3f}% commission per trade[/blue]")
            print(f"[cyan]Return calculation mode: {'Compound (Reinvest)' if self.reinvest else 'Linear (No Reinvest)'}[/cyan]")

        return self.data

    def get_performance_metrics(self):
        if self.data is None or 'Position' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        # Use the strategy returns that already include trading costs and proper timing
        position = self.data['Position']
        open_prices = self.data['Open']
        strategy_returns = self.data['Strategy_Returns'].dropna()
        portfolio_value = self.data['Portfolio_Value'].dropna()
        
        # Calculate metrics using the actual portfolio performance (respects reinvest setting)
        total_return = (portfolio_value.iloc[-1] / self.initial_capital) - 1
        
        # For benchmark comparison, use the same reinvest logic
        asset_returns = self.data['Return'].dropna()
        if self.reinvest:
            benchmark_total_return = (1 + asset_returns).prod() - 1
        else:
            benchmark_total_return = asset_returns.sum()
        
        active_return = total_return - benchmark_total_return
        
        # Calculate drawdown from actual portfolio values
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe and Sortino using strategy returns
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std(ddof=1) if strategy_returns.std() != 0 else float('nan')
        
        downside_returns = strategy_returns[strategy_returns < 0]
        sortino_ratio = strategy_returns.mean() / downside_returns.std(ddof=1) if len(downside_returns) > 0 and downside_returns.std() != 0 else float('nan')
        
        # Information ratio using active returns
        if self.reinvest:
            active_returns_series = strategy_returns - asset_returns.reindex_like(strategy_returns).fillna(0)
        else:
            # For linear returns, active return per period is the difference between strategy and benchmark
            benchmark_returns_linear = asset_returns.reindex_like(strategy_returns).fillna(0)
            active_returns_series = strategy_returns - benchmark_returns_linear
        
        info_ratio = active_returns_series.mean() / active_returns_series.std(ddof=1) if active_returns_series.std() != 0 else float('nan')
        
        # Calculate Alpha and Beta using regression (these should work the same for both modes)
        if len(strategy_returns) >= 2 and len(asset_returns) >= 2:
            try:
                import statsmodels.api as sm
                # Align the series
                common_index = strategy_returns.index.intersection(asset_returns.index)
                if len(common_index) >= 2:
                    y = strategy_returns.loc[common_index]
                    x = asset_returns.loc[common_index]
                    X = sm.add_constant(x.values)
                    model = sm.OLS(y.values, X).fit()
                    alpha_raw = model.params[0]
                    beta = model.params[1]
                    # Annualize alpha
                    alpha = alpha_raw * (365 / self.n_days) if self.n_days > 0 else alpha_raw
                else:
                    alpha, beta = float('nan'), float('nan')
            except:
                alpha, beta = float('nan'), float('nan')
        else:
            alpha, beta = float('nan'), float('nan')
        
        # Trade-level metrics (these work the same regardless of reinvest mode)
        trade_pnls = metrics.get_trade_pnls(position, open_prices)
        
        # Profit factor using strategy returns
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        profit_factor = positive_returns.sum() / abs(negative_returns.sum()) if negative_returns.sum() < 0 else float('inf')

        return {
            'Total_Return': total_return,
            'Alpha': alpha,
            'Beta': beta,
            'Active_Return': active_return,
            'Max_Drawdown': max_drawdown,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Information_Ratio': info_ratio,
            'Win_Rate': len([pnl for pnl in trade_pnls if pnl > 0]) / len(trade_pnls) if trade_pnls else 0,
            'Breakeven_Rate': metrics.get_breakeven_rate_from_pnls(trade_pnls),
            'RR_Ratio': metrics.get_rr_ratio_from_pnls(trade_pnls),
            'PT_Ratio': (strategy_returns.sum() / len(trade_pnls)) * 100 if trade_pnls else 0,
            'Profit_Factor': profit_factor,
            'Total_Trades': len(trade_pnls)
        }

    def plot_performance(self, show_graph: bool = True, advanced: bool = False):
        if self.data is None or 'Portfolio_Value' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        summary = self.get_performance_metrics()

        if not advanced:
            fig = go.Figure()
            
            portfolio_value = self.data['Portfolio_Value'].values
            # Calculate asset value using the same reinvestment logic as the strategy
            asset_returns = self.data['Return'].dropna()
            if self.reinvest:
                asset_value = self.initial_capital * (1 + self.data['Return']).cumprod()
            else:
                asset_linear_profit = (self.initial_capital * self.data['Return']).cumsum()
                asset_value = self.initial_capital + asset_linear_profit

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=portfolio_value,
                mode='lines',
                name='Strategy Portfolio',
            ))
            
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=asset_value,
                mode='lines',
                name=f'Buy & Hold ({"Compound" if self.reinvest else "Linear"})'
            ))

            position_changes = np.diff(self.data['Position'].values)
            
            # Separate different types of position changes
            long_entries = []
            short_entries = []
            flats = []
            long_entry_values = []
            short_entry_values = []
            flat_values = []
            
            for i in range(len(position_changes)):
                if position_changes[i] != 0:  # Any position change
                    prev_pos = self.data['Position'].iloc[i]
                    current_pos = self.data['Position'].iloc[i + 1]
                    
                    if current_pos == 3 and prev_pos != 3:  # Entering long from any other position
                        long_entries.append(self.data.index[i + 1])
                        long_entry_values.append(asset_value.iloc[i + 1])
                    elif current_pos == 1 and prev_pos != 1:  # Entering short from any other position
                        short_entries.append(self.data.index[i + 1])
                        short_entry_values.append(asset_value.iloc[i + 1])
                    elif current_pos == 2 and prev_pos != 2:  # Exiting to flat from any position
                        flats.append(self.data.index[i + 1])
                        flat_values.append(asset_value.iloc[i + 1])

            # Add long entry signals (green triangles up)
            if long_entries:
                fig.add_trace(go.Scatter(
                    x=long_entries,
                    y=long_entry_values,
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))

            # Add short entry signals (red triangles down) 
            if short_entries:
                fig.add_trace(go.Scatter(
                    x=short_entries,
                    y=short_entry_values,
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))

            # Add flat signals (yellow circles)
            if flats:
                fig.add_trace(go.Scatter(
                    x=flats,
                    y=flat_values,
                    mode='markers',
                    name='Exit to Flat',
                    marker=dict(color='yellow', size=8, symbol='circle')
                ))
            
            fig.update_layout(
                title=f'{self.symbol} {self.chunks} days of {self.interval} | {self.age_days}d old | {"Compound" if self.reinvest else "Linear"} | TR: {summary["Total_Return"]*100:.3f}% | Max DD: {summary["Max_Drawdown"]*100:.3f}% | RR: {summary["RR_Ratio"]:.3f} | WR: {summary["Win_Rate"]*100:.3f}% | BE: {summary["Breakeven_Rate"]*100:.3f}% | PT: {summary["PT_Ratio"]*100:.3f}% | PF: {summary["Profit_Factor"]:.3f} | Sharpe: {summary["Sharpe_Ratio"]:.3f} | Sortino: {summary["Sortino_Ratio"]:.3f} | Trades: {summary["Total_Trades"]}',
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
                    "Position Distribution", "Cumulative PnL by Trade"
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "scatter"}]
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
            
            # Calculate asset value using the same reinvestment logic as the strategy
            if self.reinvest:
                asset_value = self.initial_capital * (1 + self.data['Return']).cumprod()
            else:
                asset_linear_profit = (self.initial_capital * self.data['Return']).cumsum()
                asset_value = self.initial_capital + asset_linear_profit
                
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=asset_value,
                    mode='lines',
                    name=f'Buy & Hold ({"Compound" if self.reinvest else "Linear"})',
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

            # 3. Profit and Loss Distribution - use metrics function with open prices
            pnl_list = metrics.get_trade_pnls(self.data['Position'], self.data['Open'])
            # Convert PnL to percentage of portfolio value at time of trade
            initial_value = self.initial_capital
            pnl_pct_list = [(pnl / initial_value) * 100 for pnl in pnl_list]
            
            fig.add_trace(
                go.Histogram(
                    x=pnl_pct_list,
                    name='Trade Returns (%)',
                    nbinsx=50
                ),
                row=2, col=1
            )

            # 4. Average Profit per Trade - use strategy returns
            pnl_arr = np.array(pnl_pct_list)  # Already in percentage
            cumulative_pnl = np.cumsum(pnl_arr) if len(pnl_arr) else np.array([0])
            trade_numbers = np.arange(1, len(pnl_arr) + 1) if len(pnl_arr) else np.array([1])
            avg_pnl_per_trade = cumulative_pnl / trade_numbers
            
            fig.add_trace(
                go.Scatter(
                    x=trade_numbers,
                    y=avg_pnl_per_trade,
                    mode='lines',
                    name='Avg PnL per Trade (%)',
                ),
                row=2, col=2
            )

            if len(pnl_arr):
                mean_pnl = np.mean(pnl_arr)
                fig.add_trace(
                    go.Scatter(
                        x=[1, len(pnl_arr)],
                        y=[mean_pnl, mean_pnl],
                        mode='lines',
                        name='Mean PnL',
                        line=dict(dash='dash', color='red')
                    ),
                    row=2, col=2
                )

            # 5. Win Rate Over Time
            if len(pnl_arr):
                cumulative_wins = np.cumsum(pnl_arr > 0)
                win_rates = cumulative_wins / trade_numbers * 100
                fig.add_trace(
                    go.Scatter(
                        x=trade_numbers,
                        y=win_rates,
                        mode='lines',
                        name='Win Rate (%)'
                    ),
                    row=3, col=1
                )

            # 6. Sharpe Ratio Over Time (per-period, not annualized)
            rolling_returns = self.data['Strategy_Returns'].rolling(window=30)
            rolling_sharpe = rolling_returns.mean() / rolling_returns.std()
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=rolling_sharpe,
                    mode='lines',
                    name='Rolling Sharpe (Per-Period)'
                ),
                row=3, col=2
            )
            
            if not rolling_sharpe.isna().all():
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

            # 7. Position Distribution - show actual position states
            position_counts = self.data['Position'].value_counts().sort_index()
            position_labels = {0: 'Hold', 1: 'Short', 2: 'Flat', 3: 'Long'}
            
            fig.add_trace(
                go.Histogram(
                    x=[position_labels.get(pos, f'Unknown({pos})') for pos in self.data['Position']],
                    name='Position States'
                ),
                row=4, col=1
            )

            # 8. Cumulative PnL by Trade
            if pnl_pct_list:
                fig.add_trace(
                    go.Scatter(
                        x=trade_numbers,
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='Cumulative PnL (%)',
                        marker=dict(size=4)
                    ),
                    row=4, col=2
                )

            # Update layout
            fig.update_layout(
                title_text=f"{self.symbol} {self.interval} Performance Analysis ({'Compound' if self.reinvest else 'Linear'} Returns)",
                showlegend=True,
                template="plotly_dark"
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_yaxes(title_text="Avg PnL per Trade (%)", row=2, col=2)
            fig.update_yaxes(title_text="Win Rate (%)", row=3, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=2)
            fig.update_yaxes(title_text="Frequency", row=4, col=1)
            fig.update_yaxes(title_text="Cumulative PnL (%)", row=4, col=2)
            
            # Update x-axis labels
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_xaxes(title_text="PnL (%)", row=2, col=1)
            fig.update_xaxes(title_text="Trade Number", row=2, col=2)
            fig.update_xaxes(title_text="Trade Number", row=3, col=1)
            fig.update_xaxes(title_text="Time", row=3, col=2)
            fig.update_xaxes(title_text="Position State", row=4, col=1)
            fig.update_xaxes(title_text="Trade Number", row=4, col=2)

            end = time.time()
            print(f"[green]Advanced Plotting Done ({end - start:.2f} seconds)[/green]")

        if show_graph:
            fig.show()
        
        return fig

    def debug_strategy(self, n_bars: int = 10):
        """Print debugging information about the strategy execution."""
        if self.data is None or 'Position' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")
            
        print(f"\n[bold blue]STRATEGY DEBUG INFORMATION[/bold blue]")
        print(f"Data shape: {self.data.shape}")
        print(f"Trading costs: {self.slippage_pct*100:.3f}% slippage + {self.commission_pct*100:.3f}% commission")
        print(f"Return mode: {'Compound (Reinvest)' if self.reinvest else 'Linear (No Reinvest)'}")
        
        # Position distribution
        pos_counts = self.data['Position'].value_counts().sort_index()
        print(f"\nPosition distribution:")
        pos_labels = {0: 'Hold', 1: 'Short', 2: 'Flat', 3: 'Long'}
        for pos, count in pos_counts.items():
            print(f"  {pos_labels.get(pos, f'Unknown({pos})')}: {count} bars ({count/len(self.data)*100:.1f}%)")
        
        # Strategy performance summary
        total_return = (self.data['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        max_dd = self.data['Drawdown'].min() * 100
        n_trades = len([1 for i in range(1, len(self.data)) if self.data['Position'].iloc[i] != self.data['Position'].iloc[i-1]])
        
        print(f"\nPerformance Summary:")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Number of Position Changes: {n_trades}")
        
        # Show first N bars
        print(f"\n[bold]FIRST {n_bars} BARS:[/bold]")
        debug_cols = ['Open', 'Close', 'Return', 'Position', 'Strategy_Returns', 'Portfolio_Value']
        available_cols = [col for col in debug_cols if col in self.data.columns]
        
        first_bars = self.data[available_cols].head(n_bars)
        for i, (idx, row) in enumerate(first_bars.iterrows()):
            open_price = row.get('Open', 'N/A')
            close_price = row.get('Close', 'N/A')
            print(f"Bar {i:2d}: Open={open_price:8.2f}, Close={close_price:8.2f}, "
                  f"Return={row['Return']*100:6.2f}%, Pos={row['Position']}, "
                  f"StratRet={row['Strategy_Returns']*100:6.2f}%, Portfolio=${row['Portfolio_Value']:8.2f}")
        
        # Show last N bars  
        print(f"\n[bold]LAST {n_bars} BARS:[/bold]")
        last_bars = self.data[available_cols].tail(n_bars)
        start_idx = len(self.data) - n_bars
        for i, (idx, row) in enumerate(last_bars.iterrows()):
            open_price = row.get('Open', 'N/A')
            close_price = row.get('Close', 'N/A')
            print(f"Bar {start_idx+i:2d}: Open={open_price:8.2f}, Close={close_price:8.2f}, "
                  f"Return={row['Return']*100:6.2f}%, Pos={row['Position']}, "
                  f"StratRet={row['Strategy_Returns']*100:6.2f}%, Portfolio=${row['Portfolio_Value']:8.2f}")
        
        # Check for NaN values
        nan_cols = []
        for col in self.data.columns:
            if self.data[col].isna().any():
                nan_count = self.data[col].isna().sum()
                nan_cols.append(f"{col}: {nan_count}")
        
        if nan_cols:
            print(f"\n[yellow]NaN values found:[/yellow]")
            for col_info in nan_cols:
                print(f"  {col_info}")
        else:
            print(f"\n[green]No NaN values found[/green]")

class Strategy:
    @staticmethod
    def hold_strategy(data: pd.DataFrame, signal: int = 3) -> pd.Series:
        return pd.Series(signal, index=data.index)

    @staticmethod
    def perfect_strategy(data: pd.DataFrame) -> pd.Series:
        """
        Perfect strategy that uses only past information to predict future open-to-open returns.
        
        Uses Close[t-1] to predict return from Open[t] to Open[t+1]
        Signal at Close[t-1] -> Execute at Open[t] -> Earn Open[t] to Open[t+1]
        """
        signals = pd.Series(2, index=data.index)
        
        if len(data) < 2:
            return signals
            
        # For each bar, use only information available at the previous close
        for i in range(1, len(data)):
            # At close of bar i-1, predict return from Open[i] to Open[i+1]
            if i < len(data) - 1:  # Make sure we have the next open
                # This is the return we want to predict and capture
                future_return = (data['Open'].iloc[i+1] - data['Open'].iloc[i]) / data['Open'].iloc[i]
                
                # "Perfect" prediction based on past data (in reality this would be a model)
                if future_return > 0:
                    signals.iloc[i] = 3  # Long
                elif future_return < 0:
                    signals.iloc[i] = 1  # Short
                # else stay flat (2)
        
        return signals

    @staticmethod
    def ema_cross_strategy(data: pd.DataFrame, fast_period: int = 9, slow_period: int = 26) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        ema_fast = ta.ema(data['Close'], fast_period)
        ema_slow = ta.ema(data['Close'], slow_period)
        signals[ema_fast > ema_slow] = 3
        signals[ema_fast < ema_slow] = 1
        return signals

    @staticmethod
    def zscore_reversion_strategy(data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        zscore = ta.zscore(data['Close'])
        signals[zscore < -1] = 3
        signals[zscore > 1] = 1
        return signals

    @staticmethod
    def zscore_momentum_strategy(data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        zscore = ta.zscore(data['Close'])
        signals[zscore < -1] = 1
        signals[zscore > 1] = 3
        return signals

    @staticmethod
    def rsi_strategy(data: pd.DataFrame, oversold: int = 32, overbought: int = 72) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        rsi = ta.rsi(data['Close'])
        signals[rsi < oversold] = 3
        signals[rsi > overbought] = 1
        return signals

    @staticmethod
    def macd_strategy(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        macd, signal = ta.macd(data['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        signals[macd > signal] = 3
        signals[macd < signal] = 1
        return signals

    @staticmethod
    def supertrend_strategy(data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        supertrend, supertrend_line = ta.supertrend(data['High'], data['Low'], data['Close'], period=14, multiplier=3)
        signals[data['Close'] > supertrend_line] = 3
        signals[data['Close'] < supertrend_line] = 1
        return signals

    @staticmethod
    def psar_strategy(data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        psar = ta.psar(data['High'], data['Low'], acceleration_start=0.02, acceleration_step=0.02, max_acceleration=0.2)
        signals[data['Close'] > psar] = 3
        signals[data['Close'] < psar] = 1
        return signals

    @staticmethod
    def smc_strategy(data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        return signals

    @staticmethod
    def scalper_strategy1(data: pd.DataFrame, fast_period: int = 9, slow_period: int = 26, adx_threshold: int = 25, momentum_period: int = 10, momentum_threshold: float = 0.75, wick_threshold: float = 0.5) -> pd.Series:
        signals = pd.Series(2, index=data.index)
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
        signals[buy_conditions] = 3
        signals[sell_conditions] = 1
        return signals

    @staticmethod
    def ETHBTC_trader(data: pd.DataFrame, chunks, interval, age_days, lower_zscore_threshold: float = -1, upper_zscore_threshold: float = 1) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        eth_price = mt.fetch_data('ETH-USDT', chunks=chunks, interval=interval, age_days=age_days, kucoin=True)
        btc_price = mt.fetch_data('BTC-USDT', chunks=chunks, interval=interval, age_days=age_days, kucoin=True)
        ethbtc_ratio = eth_price[['Open', 'High', 'Low', 'Close']] / btc_price[['Open', 'High', 'Low', 'Close']]
        
        zscore = ta.zscore(ethbtc_ratio['Open'])

        #follow ethbtc ratio breakouts
        buy_conditions = zscore > upper_zscore_threshold
        sell_conditions = zscore < lower_zscore_threshold
        signals[buy_conditions] = 3
        signals[sell_conditions] = 1
        return signals

    @staticmethod
    def volatility_breakout_strategy(data: pd.DataFrame, atr_period: int = 14, atr_lookback: int = 30, atr_threshold: float = 1.2, donchian_period: int = 20, ema_fast: int = 20, ema_slow: int = 50, use_rsi_filter: int = 1, rsi_threshold: float = 70.0) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        atr = ta.atr(data['High'], data['Low'], data['Close'], timeperiod=atr_period)
        donchian_high, donchian_middle, donchian_low = ta.donchian_channel(data['High'], data['Low'], timeperiod=donchian_period)
        ema_fast_line = ta.ema(data['Close'], ema_fast)
        ema_slow_line = ta.ema(data['Close'], ema_slow)
        atr_min = atr.rolling(window=atr_lookback).min()
        if bool(use_rsi_filter):
            rsi = ta.rsi(data['Close'])
        for i in range(atr_lookback, len(data)):
            volatility_low = atr.iloc[i] < (atr_min.iloc[i] * atr_threshold)
            breakout_long = data['Close'].iloc[i] > donchian_high.iloc[i-1]
            trend_up = ema_fast_line.iloc[i] > ema_slow_line.iloc[i]
            rsi_ok = True
            if use_rsi_filter:
                rsi_ok = rsi.iloc[i] < rsi_threshold
            if volatility_low and breakout_long and trend_up and rsi_ok:
                signals.iloc[i] = 3
            elif volatility_low and data['Close'].iloc[i] < donchian_low.iloc[i-1] and not trend_up:
                if not use_rsi_filter or rsi.iloc[i] > (100 - rsi_threshold):
                    signals.iloc[i] = 1
        return signals

    @staticmethod
    def high_rr_momentum_strategy(data: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, volume_multiplier: float = 1.5, ema_trend: int = 50, min_consolidation_bars: int = 10) -> pd.Series:
        signals = pd.Series(2, index=data.index)
        bb_upper, bb_middle, bb_lower = ta.bbands(data['Close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle
        ema_trend_line = ta.ema(data['Close'], ema_trend)
        avg_volume = data['Volume'].rolling(window=20).mean()
        bb_width_min = bb_width.rolling(window=30).min()
        for i in range(30, len(data)):
            squeeze = bb_width.iloc[i] < (bb_width_min.iloc[i] * 1.3)
            breakout_up = data['Close'].iloc[i] > bb_upper.iloc[i-1]
            breakout_down = data['Close'].iloc[i] < bb_lower.iloc[i-1]
            uptrend = data['Close'].iloc[i] > ema_trend_line.iloc[i]
            volume_spike = data['Volume'].iloc[i] > (avg_volume.iloc[i] * volume_multiplier)
            if squeeze and breakout_up and uptrend and volume_spike:
                signals.iloc[i] = 3
            elif squeeze and breakout_down and not uptrend and volume_spike:
                signals.iloc[i] = 1
        return signals

    @staticmethod
    def nn_strategy(data: pd.DataFrame, batch_size: int = 64, check_consistency: bool = False) -> pd.Series:
        from brains.time_series.single_predictors.classifier_model import load_model
        MODEL_PATH = r"trading\BTC-USDT_1min_5_38features.pth"
        signals = pd.Series(2, index=data.index)
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
                batch_x = batch[0]
                outputs = model(batch_x)
                probs = F.softmax(outputs, dim=2)
                actions = torch.argmax(probs, dim=2).cpu().numpy().flatten()
                for i, action in enumerate(actions):
                    if action == 3:
                        position_values[idx + i] = 3
                    elif action == 1:
                        position_values[idx + i] = 1
                    else:
                        position_values[idx + i] = 2
                idx += len(actions)
        signals.iloc[len(data)-len(position_values):] = position_values
        signals = signals.ffill().fillna(2)
        if check_consistency:
            with torch.no_grad():
                single_preds = []
                for i in range(min(32, len(X))):
                    x_single = X[i].unsqueeze(0).unsqueeze(0)
                    out = model(x_single)
                    prob = F.softmax(out, dim=2)
                    act = torch.argmax(prob, dim=2).item()
                    single_preds.append(act)
                batch_preds = position_values[:len(single_preds)]
                if not np.all(batch_preds == np.array([3 if a==3 else 1 if a==1 else 2 for a in single_preds])):
                    print("[red]WARNING: Batch and single inference results differ![red]")
        return signals

if __name__ == "__main__":
    # Test compound returns (reinvestment)
    backtest_compound = VectorizedBacktesting(
        initial_capital=400,
        slippage_pct=0.0,
        commission_pct=0.0,
        reinvest=True
    )
    backtest_compound.fetch_data(
        symbol="BTC-USDT",
        chunks=10,
        interval="5min",
        age_days=0
    )
    
    backtest_compound.run_strategy(Strategy.ETHBTC_trader, 
                          verbose=True, 
                          chunks=backtest_compound.chunks, 
                          interval=backtest_compound.interval, 
                          age_days=backtest_compound.age_days,
                          lower_zscore_threshold=-0.33,
                          upper_zscore_threshold=0.101)
    print("[bold green]COMPOUND RETURNS (REINVEST):[/bold green]")
    print(backtest_compound.get_performance_metrics())
    
    # Test linear returns (no reinvestment)
    backtest_linear = VectorizedBacktesting(
        initial_capital=400,
        slippage_pct=0.0,
        commission_pct=0.0,
        reinvest=False
    )
    backtest_linear.fetch_data(
        symbol="BTC-USDT",
        chunks=10,
        interval="5min",
        age_days=0
    )
    
    backtest_linear.run_strategy(Strategy.ETHBTC_trader, 
                          verbose=True, 
                          chunks=backtest_linear.chunks, 
                          interval=backtest_linear.interval, 
                          age_days=backtest_linear.age_days,
                          lower_zscore_threshold=-0.33,
                          upper_zscore_threshold=0.101)
    print("[bold yellow]LINEAR RETURNS (NO REINVEST):[/bold yellow]")
    print(backtest_linear.get_performance_metrics())
    
    # Compare both on same plot would require modification to plotting function
    backtest_compound.plot_performance(advanced=False)
    backtest_linear.plot_performance(advanced=False)