import time
import numpy as np
import pandas as pd
from rich import print

import trading.model_tools as mt
import trading.backtesting.vb_metrics as metrics

class VectorizedBacktesting:
    def __init__(
        self,
        instance_name: str = "default",
        initial_capital: float = 10000.0,
        slippage_pct: float = 0.001,
        commission_fixed: float = 0.0,
        leverage: float = 1.0,
    ):
        self.instance_name = instance_name
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.leverage = leverage

        self.symbol = None
        self.chunks = None
        self.interval = None
        self.age_days = None
        self.n_days = None
        self.data_source = None
        self.data: pd.DataFrame = pd.DataFrame()

        self.strategy_output = None
        #strategy may return a signal and indicators as
        #return signals, (indicator, use_price_scale)

    def fetch_data(self,
        symbol: str = "None",
        days: int = "None",
        interval: str = "None",
        age_days: int = "None",
        data_source: str = "kucoin",
        cache_expiry_hours: int = 24,
        retry_limit: int = 3,
        verbose: bool = True
    ):
        self.symbol = symbol
        self.days = days
        self.interval = interval
        self.age_days = age_days
        self.data_source = data_source
        if any([symbol, days, interval, age_days]) == "None":
            pass
        else:
            self.data = mt.fetch_data(symbol, days, interval, age_days, data_source=data_source, cache_expiry_hours=cache_expiry_hours, retry_limit=retry_limit, verbose=verbose)
            self._set_n_days()

            return self.data

    def load_data(self, data: pd.DataFrame):
        self.data = data
        self._set_n_days()

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

    def _apply_trading_costs(self, base_strategy_returns: pd.Series, positions: pd.Series) -> pd.Series:
        """
        Apply slippage and fixed commission based on continuous position changes.
        
        positions: float series in [-1,1], representing exposure
        """
        if self.slippage_pct == 0 and self.commission_fixed == 0:
            return base_strategy_returns

        delta_pos = positions.diff().fillna(positions.iloc[0]).abs()  # first bar: full entry

        slippage_cost_pct = self.slippage_pct / 100
        commission_cost_pct = self.commission_fixed / self.initial_capital  # already scaled

        cost_per_unit = slippage_cost_pct + commission_cost_pct

        total_cost_pct = delta_pos * cost_per_unit

        net_returns = base_strategy_returns - total_cost_pct

        return net_returns

    def _set_n_days(self):
        oldest = self.data['Datetime'].iloc[0]
        newest = self.data['Datetime'].iloc[-1]
        self.n_days = (newest - oldest).days

    def run_strategy(self, strategy_func, verbose: bool = False, **kwargs):
        """
        Run a trading strategy.

        EXECUTION MODEL:
        - Signal generated at Close[t] using all data up to time t
        - Trade executed at Open[t+1] (next period's open)
        - Return earned from Open[t+1] to Open[t+2]

        Returns a pd.Series of the data.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")

        import time
        start_time = time.time()

        self.strategy_output = strategy_func(self.data, **kwargs)
        raw_signals = self.strategy_output[0] if isinstance(self.strategy_output, tuple) else self.strategy_output
        if (sum(raw_signals > 1) > 0) or (sum(raw_signals < -1) > 0):
            print(f"[orange1]Warning: Signals must be between [-1,1]")
            raw_signals = np.clip(raw_signals, -1, 1) #clip signals to [-1,1]
        
        self.data['Open_Return'] = self.data['Open'].pct_change().shift(-1)
        
        self.data['Signal_Position'] = raw_signals.ffill().fillna(0).astype(float) #forward fill hold signals, default to flat at start
        self.data['Position'] = self.data['Signal_Position'].shift(1).fillna(0).astype(float) #delay

        base_strategy_returns = self.data['Position'] * self.leverage * self.data['Open_Return']
        self.data['Strategy_Returns'] = self._apply_trading_costs(
            base_strategy_returns=base_strategy_returns, positions=self.data['Position']
        )

        self.data['Linear_Profit'] = (self.initial_capital * self.data['Strategy_Returns']).cumsum()
        self.data['Portfolio_Value'] = self.initial_capital + self.data['Linear_Profit']
        self.data['Percent_Return'] = self.data['Portfolio_Value'] / self.initial_capital

        self.data['Peak'] = self.data['Portfolio_Value'].cummax()
        self.data['Drawdown'] = (self.data['Portfolio_Value'] - self.data['Peak']) / self.data['Peak']

        end_time = time.time()
        if verbose:
            print(f"[green]Strategy execution time: {end_time - start_time:.2f} seconds[/green]")
            if self.slippage_pct > 0 or self.commission_fixed > 0:
                print(f"[blue]Costs: {self.slippage_pct:.3f}% slippage + {self.commission_fixed:.2f} fixed[/blue]")
            if self.leverage != 1.0:
                print(f"[yellow]Leverage: {self.leverage}x[/yellow]")

        return self.data

    def get_performance_metrics(self):
        if self.data is None or 'Position' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        position = self.data['Position']
        strategy_returns = self.data['Strategy_Returns'].dropna()
        portfolio_value = self.data['Portfolio_Value'].dropna()
        
        try:
            total_return = (portfolio_value.iloc[-1] / self.initial_capital) - 1
        except:
            total_return = 0
        
        asset_returns = self.data['Open_Return'].dropna()
        benchmark_total_return = asset_returns.sum()
        
        active_return = total_return - benchmark_total_return
        
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = drawdown.min()

        sharpe_ratio, sharpe_t_stat = metrics.get_sharpe_ratio(strategy_returns, self.interval, self.n_days)
        sortino_ratio = metrics.get_sortino_ratio(strategy_returns, self.interval, self.n_days) if strategy_returns.std() != 0 else float('nan')
        info_ratio = metrics.get_information_ratio(strategy_returns, asset_returns, self.interval, self.n_days) if strategy_returns.std() != 0 else float('nan')
        alpha, beta = metrics.get_alpha_beta(strategy_returns, asset_returns, n_days=self.n_days, return_interval=self.interval)

        return {
            'Total_Return': total_return,
            'Alpha': alpha,
            'Beta': beta,
            'Active_Return': active_return,
            'Max_Drawdown': max_drawdown,
            'Sharpe_Ratio': sharpe_ratio,
            'Sharpe_T_Stat': sharpe_t_stat,
            'Sortino_Ratio': sortino_ratio,
            'Information_Ratio': info_ratio,
            'Total_Trades': metrics.get_total_trades(position),
        }

    def plot_performance(self, mode: str = "basic"):
        """
        Modes: "basic", "standard", "extended"
        Standard mode can plot indicators attached as a tuple in the strategy output containing either True or False for using price scale.

        Ex: return signals, (indicator, True)
        
        """
        if self.data is None or 'Portfolio_Value' not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        import plotly.graph_objects as go
        import plotly.io as pio
        pio.renderers.default = "browser"
        import matplotlib.pyplot as plt

        summary = self.get_performance_metrics()
        if mode == "basic":
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.data['Datetime'], self.data['Portfolio_Value'], color='orange')
            ax[0].plot(self.data['Datetime'], self.initial_capital * (1 + self.data['Open_Return']).cumprod(), color='blue')
            ax[0].legend(["Strategy Portfolio", "Buy & Hold"])

            ax[1].plot(self.data['Datetime'], self.data['Position'], color='green')
            ax[1].legend(["Position"])

            plt.title(f"{self.symbol} {self.n_days} days of {self.interval} | {self.age_days}d old | Linear | TR: {summary['Total_Return']*100:.3f}% | Alpha: {summary['Alpha']*100:.3f}% | Beta: {summary['Beta']:.3f} | Max DD: {summary['Max_Drawdown']*100:.3f}% | Sharpe: {summary['Sharpe_Ratio']:.3f} | Sortino: {summary['Sortino_Ratio']:.3f} | Trades: {summary['Total_Trades']}")
            plt.show()

        elif mode == "standard":
            #y1 for price, y2 for portfolio, y3 for active returns, y4 for indicators
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=self.data['Datetime'],
                    open=self.data['Open'],
                    high=self.data['High'],
                    low=self.data['Low'],
                    close=self.data['Close'],
                    increasing_fillcolor='#888888',
                    increasing_line_color='#888888',
                    decreasing_fillcolor='#00B4FF',
                    decreasing_line_color='#00B4FF',
                    name='Price',
                    yaxis='y'
                )
            )
            portfolio_value = self.data['Portfolio_Value'].values
            valid_returns = self.data['Open_Return'].fillna(0)
            hodl_value = self.initial_capital * (1 + valid_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=self.data['Datetime'],
                    y=portfolio_value,
                    mode='lines',
                    name='Strategy Portfolio',
                    line=dict(width=2),
                    yaxis='y2'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data['Datetime'],
                    y=hodl_value,
                    mode='lines',
                    name=f'Buy & Hold',
                    line=dict(width=2),
                    yaxis='y2',
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data['Datetime'],
                    y=portfolio_value - hodl_value,
                    mode='lines',
                    name='Active Return',
                    line=dict(dash='dash', width=2),
                    yaxis='y3',
                    visible="legendonly"
                )
            )

            position_changes = self.data['Position'].diff()
            long_entries = []
            short_entries = []
            flats = []
            long_entry_prices = []
            short_entry_prices = []
            flat_prices = []
            
            for i in range(1, len(self.data)):
                if position_changes.iloc[i] != 0 and not pd.isna(position_changes.iloc[i]):
                    current_pos = self.data['Position'].iloc[i]
                    price_at_execution = self.data['Open'].iloc[i]

                    if (current_pos == 0) and (position_changes.iloc[i] != 0):
                        flats.append(self.data['Datetime'].iloc[i])
                        flat_prices.append(price_at_execution)
                    elif position_changes.iloc[i] > 0:
                        long_entries.append(self.data['Datetime'].iloc[i])
                        long_entry_prices.append(price_at_execution)
                    elif position_changes.iloc[i] < 0:
                        short_entries.append(self.data['Datetime'].iloc[i])
                        short_entry_prices.append(price_at_execution)
            if long_entries:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries,
                        y=long_entry_prices,
                        mode='markers',
                        name='Add Long',
                        marker=dict(color="#26FF00", size=8, symbol='triangle-up'),
                        yaxis='y'
                    )
                )
            if short_entries:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries,
                        y=short_entry_prices,
                        mode='markers',
                        name='Add Short',
                        marker=dict(color='#ff073a', size=8, symbol='triangle-down'),
                        yaxis='y'
                    )
                )
            if flats:
                fig.add_trace(
                    go.Scatter(
                        x=flats,
                        y=flat_prices,
                        mode='markers',
                        name='Exit to Flat',
                        marker=dict(color='yellow', size=7, symbol='circle'),
                        yaxis='y'
                    )
                )
            if isinstance(self.strategy_output, tuple):
                for output_idx in range(1, len(self.strategy_output)):
                    if self.strategy_output[output_idx][1] == True: #if output is a tuple AND true, send to price axis
                        fig.add_trace(
                            go.Scatter(
                                x=self.data['Datetime'],
                                y=self.strategy_output[output_idx][0],
                                mode='lines',
                                name=f'Indicator {output_idx}',
                                yaxis='y'
                            )
                        )
                    elif self.strategy_output[output_idx][1] == False:
                        fig.add_trace(
                            go.Scatter(
                                x=self.data['Datetime'],
                                y=self.strategy_output[output_idx][0],
                                mode='lines',
                                name=f'Indicator {output_idx}',
                                yaxis='y4'
                            )
                        )
            fig.update_layout(
                title=f'{self.symbol} {self.n_days} days of {self.interval} | {self.age_days}d old | Linear | TR: {summary["Total_Return"]*100:.3f}% | Alpha: {summary["Alpha"]*100:.3f}% | Beta: {summary["Beta"]:.3f} | Max DD: {summary["Max_Drawdown"]*100:.3f}% | Sharpe: {summary["Sharpe_Ratio"]:.3f} | Sortino: {summary["Sortino_Ratio"]:.3f} | Trades: {summary["Total_Trades"]}',
                xaxis=dict(
                    title='Date',
                    rangeslider=dict(visible=False),
                    rangeselector=dict(visible=False)
                ),
                yaxis=dict(
                    title="Price ($)",
                    side="left",
                    showgrid=False,
                    anchor="x"
                ),
                yaxis2=dict(
                    title="Portfolio Value ($)",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                yaxis3=dict(
                    title="Active Return ($)",
                    overlaying="y",
                    side="right",
                    position=0.95,
                    showgrid=False
                ),
                yaxis4=dict(
                    title="Indicators",
                    overlaying="y",
                    side="left",
                    position=0.05,
                    showgrid=False
                ),
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
            
            fig.update_yaxes(title_text="Price ($)")
            fig.update_yaxes(title_text="Portfolio Value ($)")
            fig.show()

        elif mode == "tradingview":
            from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

            from lightweight_charts.widgets import QtChart

            app = QApplication([])
            window = QMainWindow()
            layout = QVBoxLayout()
            widget = QWidget()
            widget.setLayout(layout)

            window.resize(1920, 1080)
            layout.setContentsMargins(0, 0, 0, 0)

            chart = QtChart(widget)

            # Prepare candlestick data
            df = self.data.rename(columns={'Datetime': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}).drop(columns=['HL2', 'OHLC4', 'HLC3'], errors='ignore')
            chart.set(df)
            portfolio_line = chart.create_line(name="Portfolio Value")
            portfolio_line.set(pd.DataFrame({
                'time': self.data['Datetime'],
                'Portfolio Value': self.data['Portfolio_Value']
            }))
            hodl_line = chart.create_line(name="Buy & Hold")
            hodl_line.set(pd.DataFrame({
                'time': self.data['Datetime'],
                'Buy & Hold': self.initial_capital * (1 + self.data['Open_Return'].fillna(0)).cumprod()
            }))

            if isinstance(self.strategy_output, tuple):
                for output_idx in range(1, len(self.strategy_output)):
                    indicator_data = self.strategy_output[output_idx][0]
                    use_price_scale = self.strategy_output[output_idx][1]
                    
                    indicator_df = pd.DataFrame({
                        'time': self.data['Datetime'],
                        'value': indicator_data
                    })
                    
                    if use_price_scale:
                        # Add to main price pane
                        indicator_line = chart.add_line_series(name=f'Indicator {output_idx}', color='yellow', width=1)
                        indicator_line.set(indicator_df.to_dict('records'))
                    else:
                        # Add to separate pane (volume pane can be used or create new)
                        # For now, add to main pane with different color
                        indicator_line = chart.add_line_series(name=f'Indicator {output_idx}', color='cyan', width=1)
                        indicator_line.set(indicator_df.to_dict('records'))

            # Add entry/exit markers
            position_changes = self.data['Position'].diff()
            markers = []
            
            for i in range(1, len(self.data)):
                if position_changes.iloc[i] != 0 and not pd.isna(position_changes.iloc[i]):
                    current_pos = self.data['Position'].iloc[i]
                    timestamp = self.data['Datetime'].iloc[i]

                    if (current_pos == 0) and (position_changes.iloc[i] != 0):
                        markers.append({'time': timestamp, 'position': 'belowBar', 'color': 'yellow', 'shape': 'circle', 'text': 'Exit'})
                    elif position_changes.iloc[i] > 0:
                        markers.append({'time': timestamp, 'position': 'belowBar', 'color': '#26FF00', 'shape': 'arrowUp', 'text': 'Long'})
                    elif position_changes.iloc[i] < 0:
                        markers.append({'time': timestamp, 'position': 'aboveBar', 'color': '#ff073a', 'shape': 'arrowDown', 'text': 'Short'})
            
            # Add markers to chart
            # if markers:
            #     chart.marker(markers)

            # Set title with performance metrics
            title_text = f'{self.symbol} {self.n_days} days of {self.interval} | {self.age_days}d old | TR: {summary["Total_Return"]*100:.3f}% | Alpha: {summary["Alpha"]*100:.3f}% | Beta: {summary["Beta"]:.3f} | Max DD: {summary["Max_Drawdown"]*100:.3f}% | Sharpe: {summary["Sharpe_Ratio"]:.3f} | Sortino: {summary["Sortino_Ratio"]:.3f} | Trades: {summary["Total_Trades"]}'
            try:
                chart.topbar.textbox('title', title_text)
            except:
                # If topbar API doesn't work, set window title instead
                window.setWindowTitle(title_text)

            layout.addWidget(chart.get_webview())

            window.setCentralWidget(widget)
            window.show()

            app.exec_()
    def get_cost_summary(self) -> dict:
        """Get a summary of trading costs impact."""
        if self.data is None or 'Position' not in self.data.columns:
            return {"message": "No strategy results available"}
            
        # Count trades
        position_changes = self.data['Position'].diff().fillna(0)
        total_trades = (position_changes != 0).sum()
        
        if total_trades == 0:
            return {"message": "No trades executed"}
        
        # Calculate position sizes for cost estimation using pre-cost portfolio values (linear mode)
        portfolio_values = pd.Series(self.initial_capital, index=self.data.index)
        
        # Calculate total slippage and commission paid
        total_slippage_paid = 0
        total_commission_paid = 0
        
        for i in range(1, len(self.data)):
            prev_pos = self.data['Position'].iloc[i-1]
            curr_pos = self.data['Position'].iloc[i]
            
            if prev_pos != curr_pos:  # Position changed
                trade_capital = portfolio_values.iloc[i] * self.leverage  # Scale by leverage
                
                # Calculate slippage cost
                if prev_pos == 2:  # From flat (opening position)
                    slippage_cost = trade_capital * self.slippage_pct
                elif curr_pos == 2:  # To flat (closing position)
                    slippage_cost = trade_capital * self.slippage_pct
                else:  # Direct switch (close + open)
                    slippage_cost = trade_capital * self.slippage_pct * 2
                
                total_slippage_paid += slippage_cost
                
                # Calculate commission cost
                if prev_pos == 2:  # From flat (opening position)
                    commission_cost = self.commission_fixed
                elif curr_pos == 2:  # To flat (closing position)
                    commission_cost = self.commission_fixed
                else:  # Direct switch (close + open)
                    commission_cost = self.commission_fixed * 2
                
                total_commission_paid += commission_cost
        
        return {
            'total_trades': total_trades,
            'total_slippage_paid': total_slippage_paid,
            'total_commission_paid': total_commission_paid,
            'total_costs': total_slippage_paid + total_commission_paid
        }

class MultiAssetBacktesting:
    def __init__(self,
                 initial_capitals: list[float] = [10000],
                 slippage_pct: float = 0.001,
                 commission_fixed: float = 0.0,
                 leverage: float = 1.0):
        self.initial_capitals = initial_capitals
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.leverage = leverage

        self.data = {}
        self.equity_curves_pct = {}
        self.weighted_portfolio_value = None
    
    def fetch_data(self,
        symbols: list[str] = [],
        days: int = "None",
        interval: str = "None",
        age_days: int = "None",
        data_source: str = "kucoin",
        cache_expiry_hours: int = 24,
        retry_limit: int = 3,
        verbose: bool = True
    ):
        self.symbols = symbols
        self.days = days
        self.interval = interval
        self.age_days = age_days
        self.data_source = data_source
        if any([days, interval, age_days]) == "None":
            pass
        else:
            for symbol in self.symbols:
                self.data[symbol] = mt.fetch_data(symbol, days, interval, age_days, data_source=data_source, cache_expiry_hours=cache_expiry_hours, retry_limit=retry_limit, verbose=verbose)
            return self.data
    
    def run_strategy(self, strategy_func, verbose: bool = False, **kwargs):
        for symbol in self.data.keys():
            vb = VectorizedBacktesting(
                instance_name=symbol,
                initial_capital=self.initial_capitals[self.symbols.index(symbol)],
                slippage_pct=self.slippage_pct,
                commission_fixed=self.commission_fixed,
                leverage=self.leverage)
            vb.load_data(self.data[symbol])
            vb.run_strategy(strategy_func, verbose=verbose, **kwargs)
            self.equity_curves_pct[symbol] = vb.data['Percent_Return']
        
        weighted_curves = []
        for i, symbol in enumerate(self.equity_curves_pct.keys()):
            weighted_curve = self.equity_curves_pct[symbol] * self.initial_capitals[i]
            weighted_curves.append(weighted_curve)
        
        self.weighted_portfolio_value = sum(weighted_curves)
        self.weighted_portfolio_value.dropna(inplace=True)
        return self.weighted_portfolio_value
    
    def get_performance_metrics(self):
        if not self.equity_curves_pct:
            raise ValueError("No strategy results available. Run a strategy first.")
        total_return = (self.weighted_portfolio_value.iloc[-1] / self.weighted_portfolio_value.iloc[0]) - 1
        
        peak = self.weighted_portfolio_value.cummax()
        drawdown = (self.weighted_portfolio_value - peak) / peak
        max_drawdown = drawdown.min()
        
        strategy_returns = self.weighted_portfolio_value.pct_change().dropna()
        
        oldest = self.data[list(self.data.keys())[0]]['Datetime'].iloc[0]
        newest = self.data[list(self.data.keys())[0]]['Datetime'].iloc[-1]
        n_days = (newest - oldest).days
        
        sharpe_ratio = metrics.get_sharpe_ratio(strategy_returns, self.interval, n_days)
        
        sortino_ratio = metrics.get_sortino_ratio(strategy_returns, self.interval, n_days)
        
        return {
            'Total_Return': total_return,
            'Max_Drawdown': max_drawdown,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio
        }
    
    def plot_performance(self):
        import matplotlib.pyplot as plt
        summary = self.get_performance_metrics()
        plt.title(f"Multi-Asset Backtesting Performance {summary['Total_Return']*100:.3f}% | Sharpe: {summary['Sharpe_Ratio']:.3f} | Sortino: {summary['Sortino_Ratio']:.3f}")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.plot(self.weighted_portfolio_value)
        plt.show()