import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from rich import print

import trading.backtesting.metrics as metrics
import trading.model_tools as mt


def _performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    equity: pd.Series,
    positions: list[pd.Series],
    interval: str,
    n_days: int,
) -> dict:
    """Shared performance summary for single- and multi-asset books."""
    strategy_returns = pd.Series(strategy_returns).dropna()
    benchmark_returns = pd.Series(benchmark_returns).dropna()
    equity = pd.Series(equity).dropna()
    if len(strategy_returns) < 2 or len(equity) < 2:
        raise ValueError("Not enough observations to calculate metrics.")

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    peak = equity.cummax()
    max_drawdown = float((equity / peak - 1.0).min())

    sharpe_ratio, sharpe_t_stat = metrics.get_sharpe_ratio(
        strategy_returns, interval
    )
    sharpe_p_val = 1 - stats.t.cdf(
        sharpe_t_stat, df=len(strategy_returns) - 1
    )
    has_variance = strategy_returns.std() != 0
    sortino_ratio = (
        metrics.get_sortino_ratio(strategy_returns, interval, n_days)
        if has_variance
        else float("nan")
    )
    info_ratio = (
        metrics.get_information_ratio(
            strategy_returns, benchmark_returns, interval, n_days
        )
        if has_variance
        else float("nan")
    )
    alpha, beta = metrics.get_alpha_beta(
        strategy_returns,
        benchmark_returns,
        n_days=n_days,
        return_interval=interval,
    )
    r, r2 = metrics.get_r_and_r2(equity)
    cagr = metrics.get_cagr(strategy_returns, interval, n_days)
    calmar_ratio = metrics.get_calmar_ratio(
        strategy_returns, interval, n_days
    )
    total_trades = sum(
        metrics.get_total_trades(position) for position in positions
    )
    active_return = total_return - float(benchmark_returns.sum())

    return {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Alpha": alpha,
        "Beta": beta,
        "Active_Return": active_return,
        "Max_Drawdown": max_drawdown,
        "Calmar_Ratio": calmar_ratio,
        "Sharpe_Ratio": sharpe_ratio,
        "Sharpe_T_Stat": sharpe_t_stat,
        "Sharpe_P_Val": sharpe_p_val,
        "Sortino_Ratio": sortino_ratio,
        "Information_Ratio": info_ratio,
        "Total_Trades": total_trades,
        "R": r,
        "R2": r2,
    }


class VectorizedBacktest:
    def __init__(
        self,
        instance_name: str = "default",
        initial_capital: float = 10000.0,
        slippage_pct: float = 0.001,
        commission_fixed: float = 0.0,
        leverage: float = 1.0,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        self.instance_name = instance_name
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.leverage = leverage

        self.symbols = None
        self.chunks = None
        self.interval = None
        self.age_days = None
        self.n_days = None
        self.data_source = None
        self.data: pd.DataFrame = pd.DataFrame()

        self.strategy_output = None
        # strategy may return a signal and indicators as
        # return signals, (indicator, use_price_scale)

    def fetch_data(
        self,
        symbols: list[str],
        days: int,
        interval: str,
        age_days: int,
        data_source: str,
        cache_expiry_hours: int = 24,
        retry_limit: int = 3,
        verbose: bool = True,
    ):
        if not symbols:
            raise ValueError("symbols must be a non-empty list")
        self.symbols = list(symbols)
        self.days = days
        self.interval = interval
        self.age_days = age_days
        self.data_source = data_source

        self.data = mt.fetch_data(
            self.symbols,
            days,
            interval,
            age_days,
            data_source=data_source,
            cache_expiry_hours=cache_expiry_hours,
            retry_limit=retry_limit,
            verbose=verbose,
        )
        self._set_n_days()

        return self.data

    def load_data(
        self,
        data: pd.DataFrame,
        symbols: list[str] | None = None,
        interval: str = None,
        age_days: int = None,
    ):
        self.data = data.copy()
        if symbols is not None:
            self.symbols = symbols
        else:
            self.symbols = ["Unknown"]
        if interval is not None:
            self.interval = interval
        else:
            self.interval = "Unknown"
        if age_days is not None:
            self.age_days = age_days
        else:
            self.age_days = "Unknown"
        self._set_n_days()

        return self.data

    def _validate_data_quality(self):
        """Validate data quality and handle common issues."""
        if self.data.empty:
            raise ValueError("Data is empty")

        # Check for missing data
        missing_data = self.data.isnull().sum()
        if missing_data.any():
            print(
                f"[yellow]Warning: Missing data found: {missing_data[missing_data > 0].to_dict()}[/yellow]"
            )

        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            print(
                f"[yellow]Warning: {duplicates} duplicate rows found and removed[/yellow]"
            )
            self.data = self.data.drop_duplicates()

        # Check for data gaps (if datetime index)
        if "Datetime" in self.data.columns:
            time_diff = self.data["Datetime"].diff()
            expected_interval = (
                time_diff.mode()[0] if not time_diff.mode().empty else None
            )
            if expected_interval:
                large_gaps = time_diff > expected_interval * 2
                if large_gaps.any():
                    print(
                        f"[yellow]Warning: {large_gaps.sum()} large time gaps detected[/yellow]"
                    )

    def _apply_trading_costs(
        self, base_strategy_returns: pd.Series, positions: pd.Series
    ) -> pd.Series:
        """
        Apply slippage and fixed commission based on continuous position changes.

        positions: float series in [-1,1], representing exposure
        """
        if self.slippage_pct == 0 and self.commission_fixed == 0:
            return base_strategy_returns

        delta_pos = (
            positions.diff().fillna(positions.iloc[0]).abs()
        )  # first bar: full entry

        slippage_cost_pct = self.slippage_pct / 100
        commission_cost_pct = (
            self.commission_fixed / self.initial_capital
        )  # already scaled

        cost_per_unit = slippage_cost_pct + commission_cost_pct

        total_cost_pct = delta_pos * cost_per_unit

        net_returns = base_strategy_returns - total_cost_pct

        return net_returns

    def _set_n_days(self):
        oldest = self.data["Datetime"].iloc[0]
        newest = self.data["Datetime"].iloc[-1]
        self.n_days = (newest - oldest).days

    def run_strategy(self, strategy_func, verbose: bool = False, **kwargs):
        """
        Run a trading strategy.

        EXECUTION MODEL:
        - Signal generated at Close[t] using all data up to time t
        - Trade executed at Open[t+1] (next period's open)
        - Return earned from Open[t+1] to Open[t+2]

        additional returns following signal will be interpreted as indicators

        Returns a pd.Series of the data.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_data() first.")

        start_time = time.time()

        self.strategy_output = strategy_func(self.data, **kwargs)
        raw_signals = (
            self.strategy_output[0]
            if isinstance(self.strategy_output, tuple)
            else self.strategy_output
        )
        if (sum(raw_signals > 1) > 0) or (sum(raw_signals < -1) > 0):
            print(f"[orange1]Warning: Signals must be between [-1,1]")
            raw_signals = np.clip(raw_signals, -1, 1)  # clip signals to [-1,1]

        self.data["Open_Return"] = self.data["Open"].pct_change().shift(-1)

        self.data["Signal_Position"] = (
            raw_signals.ffill().fillna(0).astype(float)
        )  # forward fill hold signals, default to flat at start
        self.data["Position"] = (
            self.data["Signal_Position"].shift(1).fillna(0).astype(float)
        )  # delay

        base_strategy_returns = (
            self.data["Position"] * self.leverage * self.data["Open_Return"]
        )
        self.data["Strategy_Returns"] = self._apply_trading_costs(
            base_strategy_returns=base_strategy_returns, positions=self.data["Position"]
        )

        self.data["Linear_Profit"] = (
            self.initial_capital * self.data["Strategy_Returns"]
        ).cumsum()
        self.data["Portfolio_Value"] = self.initial_capital + self.data["Linear_Profit"]
        self.data["Percent_Return"] = (
            self.data["Portfolio_Value"] / self.initial_capital
        )

        self.data["Peak"] = self.data["Portfolio_Value"].cummax()
        self.data["Drawdown"] = (
            self.data["Portfolio_Value"] - self.data["Peak"]
        ) / self.data["Peak"]

        end_time = time.time()
        if verbose:
            print(
                f"[green]Strategy execution time: {end_time - start_time:.2f} seconds[/green]"
            )
            if self.slippage_pct > 0 or self.commission_fixed > 0:
                print(
                    f"[blue]Costs: {self.slippage_pct:.3f}% slippage + {self.commission_fixed:.2f} fixed[/blue]"
                )
            if self.leverage != 1.0:
                print(f"[yellow]Leverage: {self.leverage}x[/yellow]")

        return self.data

    def get_equity_curve(self):
        return self.data["Portfolio_Value"]

    def get_performance_metrics(self):
        if self.data is None or "Position" not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        return _performance_metrics(
            self.data["Strategy_Returns"],
            self.data["Open_Return"],
            self.data["Portfolio_Value"],
            [self.data["Position"]],
            self.interval,
            self.n_days,
        )

    def plot_performance(self, mode: str = "basic", save_name: str = None):
        """
        Modes: "basic", "standard"

        """
        if self.data is None or "Portfolio_Value" not in self.data.columns:
            raise ValueError("No strategy results available. Run a strategy first.")

        import plotly.graph_objects as go
        import plotly.io as pio

        pio.renderers.default = "browser"
        import matplotlib.pyplot as plt

        summary = self.get_performance_metrics()
        if mode == "basic":
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
            indicator_ax = None

            ax[0].plot(
                self.data["Datetime"],
                self.data["Portfolio_Value"],
                color="orange",
                label="Strategy Portfolio",
            )
            ax[0].plot(
                self.data["Datetime"],
                self.initial_capital * (1 + self.data["Open_Return"]).cumprod(),
                color="blue",
                label="Buy & Hold",
            )

            handles, labels = ax[0].get_legend_handles_labels()
            ax[0].legend(handles, labels, loc="upper right")

            ax[1].plot(self.data["Datetime"], self.data["Position"], color="green")
            ax[1].legend(["Position"], loc="upper right")

            ax[0].set_title(
                f"{self.symbols[0]} {self.n_days} days {self.interval} | TR: {summary['Total_Return'] * 100:.2f}% | Alpha: {summary['Alpha'] * 100:.2f}% | Beta: {summary['Beta']:.2f} | Max DD: {summary['Max_Drawdown'] * 100:.2f}% | Sharpe: {summary['Sharpe_Ratio']:.2f} | Trades: {summary['Total_Trades']}"
            )
            if save_name:
                plt.savefig(f"{save_name}", dpi=300)
            plt.show()

        elif mode == "standard":
            # y1 for price, y2 for portfolio, y3 for active returns, y4 for indicators
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=self.data["Datetime"],
                    open=self.data["Open"],
                    high=self.data["High"],
                    low=self.data["Low"],
                    close=self.data["Close"],
                    increasing_fillcolor="#888888",
                    increasing_line_color="#888888",
                    decreasing_fillcolor="#00B4FF",
                    decreasing_line_color="#00B4FF",
                    name="Price",
                    yaxis="y",
                )
            )
            portfolio_value = self.data["Portfolio_Value"].values
            valid_returns = self.data["Open_Return"].fillna(0)
            hodl_value = self.initial_capital * (1 + valid_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=self.data["Datetime"],
                    y=portfolio_value,
                    mode="lines",
                    name="Strategy Portfolio",
                    line=dict(width=2),
                    yaxis="y2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data["Datetime"],
                    y=hodl_value,
                    mode="lines",
                    name=f"Buy & Hold",
                    line=dict(width=2),
                    yaxis="y2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data["Datetime"],
                    y=portfolio_value - hodl_value,
                    mode="lines",
                    name="Active Return",
                    line=dict(dash="dash", width=2),
                    yaxis="y3",
                    visible="legendonly",
                )
            )

            position_changes = self.data["Position"].diff()
            long_entries = []
            short_entries = []
            flats = []
            long_entry_prices = []
            short_entry_prices = []
            flat_prices = []

            for i in range(1, len(self.data)):
                if position_changes.iloc[i] != 0 and not pd.isna(
                    position_changes.iloc[i]
                ):
                    current_pos = self.data["Position"].iloc[i]
                    price_at_execution = self.data["Open"].iloc[i]

                    if (current_pos == 0) and (position_changes.iloc[i] != 0):
                        flats.append(self.data["Datetime"].iloc[i])
                        flat_prices.append(price_at_execution)
                    elif position_changes.iloc[i] > 0:
                        long_entries.append(self.data["Datetime"].iloc[i])
                        long_entry_prices.append(price_at_execution)
                    elif position_changes.iloc[i] < 0:
                        short_entries.append(self.data["Datetime"].iloc[i])
                        short_entry_prices.append(price_at_execution)
            if long_entries:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries,
                        y=long_entry_prices,
                        mode="markers",
                        name="Add Long",
                        marker=dict(color="#26FF00", size=8, symbol="triangle-up"),
                        yaxis="y",
                    )
                )
            if short_entries:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries,
                        y=short_entry_prices,
                        mode="markers",
                        name="Add Short",
                        marker=dict(color="#ff073a", size=8, symbol="triangle-down"),
                        yaxis="y",
                    )
                )
            if flats:
                fig.add_trace(
                    go.Scatter(
                        x=flats,
                        y=flat_prices,
                        mode="markers",
                        name="Exit to Flat",
                        marker=dict(color="yellow", size=7, symbol="circle"),
                        yaxis="y",
                    )
                )
            if isinstance(self.strategy_output, tuple):
                for output_idx in range(1, len(self.strategy_output)):
                    if (
                        self.strategy_output[output_idx][1] == True
                    ):  # if output is a tuple AND true, send to price axis
                        fig.add_trace(
                            go.Scatter(
                                x=self.data["Datetime"],
                                y=self.strategy_output[output_idx][0],
                                mode="lines",
                                name=f"Indicator {output_idx}",
                                yaxis="y",
                            )
                        )
                    elif self.strategy_output[output_idx][1] == False:
                        fig.add_trace(
                            go.Scatter(
                                x=self.data["Datetime"],
                                y=self.strategy_output[output_idx][0],
                                mode="lines",
                                name=f"Indicator {output_idx}",
                                yaxis="y4",
                            )
                        )
            fig.update_layout(
                title=f"{self.symbols[0]} {self.n_days} days of {self.interval} | {self.age_days}d old | Linear | TR: {summary['Total_Return'] * 100:.3f}% | Alpha: {summary['Alpha'] * 100:.3f}% | Beta: {summary['Beta']:.3f} | Max DD: {summary['Max_Drawdown'] * 100:.3f}% | Sharpe: {summary['Sharpe_Ratio']:.3f} | Sortino: {summary['Sortino_Ratio']:.3f} | Trades: {summary['Total_Trades']}",
                xaxis=dict(
                    title="Date",
                    rangeslider=dict(visible=False),
                    rangeselector=dict(visible=False),
                ),
                yaxis=dict(title="Price ($)", side="left", showgrid=False, anchor="x"),
                yaxis2=dict(
                    title="Portfolio Value ($)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                yaxis3=dict(
                    title="Active Return ($)",
                    overlaying="y",
                    side="right",
                    position=0.95,
                    showgrid=False,
                ),
                yaxis4=dict(
                    title="Indicators",
                    overlaying="y",
                    side="left",
                    position=0.05,
                    showgrid=False,
                ),
                showlegend=True,
                template="plotly_dark",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1, xanchor="right", x=1
                ),
            )
            
            fig.show()

    def get_cost_summary(self) -> dict:
        """Get a summary of trading costs impact."""
        if self.data is None or "Position" not in self.data.columns:
            return {"message": "No strategy results available"}

        # Count trades
        position_changes = self.data["Position"].diff().fillna(0)
        total_trades = (position_changes != 0).sum()

        if total_trades == 0:
            return {"message": "No trades executed"}

        # Calculate position sizes for cost estimation using pre-cost portfolio values (linear mode)
        portfolio_values = pd.Series(self.initial_capital, index=self.data.index)

        # Calculate total slippage and commission paid
        total_slippage_paid = 0
        total_commission_paid = 0

        for i in range(1, len(self.data)):
            prev_pos = self.data["Position"].iloc[i - 1]
            curr_pos = self.data["Position"].iloc[i]

            if prev_pos != curr_pos:  # Position changed
                trade_capital = (
                    portfolio_values.iloc[i] * self.leverage
                )  # Scale by leverage

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
            "total_trades": total_trades,
            "total_slippage_paid": total_slippage_paid,
            "total_commission_paid": total_commission_paid,
            "total_costs": total_slippage_paid + total_commission_paid,
        }


class MultiAssetBacktest:
    def __init__(
        self,
        initial_capitals: float | list[float] = 10000.0,
        slippage_pct: float = 0.001,
        commission_fixed: float = 0.0,
        leverage: float = 1.0,
    ):
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        self.initial_capitals = self._normalize_capitals(initial_capitals)
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.leverage = leverage
        self.symbols: list[str] = []
        self.interval = "1h"
        self.age_days = 0
        self.data: dict[str, pd.DataFrame] = {}
        self.engines: dict[str, VectorizedBacktest] = {}
        self.weighted_portfolio_value: pd.Series | None = None

    @staticmethod
    def _normalize_capitals(initial_capitals: float | list[float]) -> list[float]:
        if isinstance(initial_capitals, (int, float)):
            initial_capitals = [initial_capitals]
        capitals = [float(value) for value in initial_capitals]
        if not capitals or any(value <= 0 for value in capitals):
            raise ValueError("initial_capitals must be positive")
        return capitals

    def _resolve_capitals(self, count: int) -> list[float]:
        if len(self.initial_capitals) == 1:
            return self.initial_capitals * count
        if len(self.initial_capitals) != count:
            raise ValueError(
                f"Expected 1 or {count} initial capitals, "
                f"got {len(self.initial_capitals)}"
            )
        return self.initial_capitals

    def fetch_data(
        self,
        symbols: list[str],
        days: int = 365,
        interval: str = "1h",
        age_days: int = 0,
        data_source: str = "binance",
        cache_expiry_hours: int = 24,
        retry_limit: int = 3,
        verbose: bool = True,
    ):
        if not symbols:
            raise ValueError("symbols must be a non-empty list")
        self.symbols = list(symbols)
        self.interval = interval
        self.age_days = age_days
        self.data = {}
        for symbol in self.symbols:
            frame = mt.fetch_data(
                symbols=[symbol],
                days=days,
                interval=interval,
                age_days=age_days,
                data_source=data_source,
                cache_expiry_hours=cache_expiry_hours,
                retry_limit=retry_limit,
                verbose=verbose,
            )
            if frame is None or frame.empty:
                raise ValueError(f"No data returned for {symbol}")
            self.data[symbol] = frame
        self.initial_capitals = self._resolve_capitals(len(self.symbols))
        return self.data

    def load_data(
        self,
        data: dict[str, pd.DataFrame],
        symbols: list[str] | None = None,
        interval: str = "1h",
        age_days: int = 0,
    ):
        if not data:
            raise ValueError("data must be a non-empty dict")
        self.symbols = list(symbols) if symbols is not None else list(data)
        missing = [symbol for symbol in self.symbols if symbol not in data]
        if missing:
            raise ValueError(f"Missing data for: {missing}")
        empty = [
            symbol
            for symbol in self.symbols
            if data[symbol] is None or data[symbol].empty
        ]
        if empty:
            raise ValueError(f"Empty data for: {empty}")
        self.data = {symbol: data[symbol] for symbol in self.symbols}
        self.interval = interval
        self.age_days = age_days
        self.initial_capitals = self._resolve_capitals(len(self.symbols))
        return self.data

    def run_strategy(self, strategy_func, verbose: bool = False, **kwargs):
        if not self.data:
            raise ValueError("No data available. Call fetch_data() or load_data() first.")
        if not (callable(strategy_func) or isinstance(strategy_func, dict)):
            raise TypeError("strategy_func must be callable or a dict")

        self.initial_capitals = self._resolve_capitals(len(self.symbols))
        self.engines = {}
        for symbol, capital in zip(self.symbols, self.initial_capitals):
            engine = VectorizedBacktest(
                instance_name=symbol,
                initial_capital=capital,
                slippage_pct=self.slippage_pct,
                commission_fixed=self.commission_fixed,
                leverage=self.leverage,
            )
            engine.load_data(self.data[symbol], interval=self.interval)
            strategy = (
                strategy_func[symbol]
                if isinstance(strategy_func, dict)
                else strategy_func
            )
            if isinstance(strategy, pd.Series):
                values = strategy
                strategy = lambda frame, v=values: pd.Series(
                    v.to_numpy(), index=frame.index
                )
            engine.run_strategy(strategy, verbose=verbose, **kwargs)
            self.engines[symbol] = engine

        self.weighted_portfolio_value = sum(
            engine.data["Percent_Return"] * capital
            for engine, capital in zip(
                self.engines.values(), self.initial_capitals
            )
        ).dropna()
        return self.weighted_portfolio_value

    def get_equity_curve(self) -> pd.Series:
        if self.weighted_portfolio_value is None:
            raise ValueError("No strategy results available. Run a strategy first.")
        return self.weighted_portfolio_value

    def get_performance_metrics(self):
        if not self.engines:
            raise ValueError("No strategy results available. Run a strategy first.")
        weights = pd.Series(self.initial_capitals, index=self.symbols)
        returns = pd.DataFrame(
            {
                symbol: engine.data["Strategy_Returns"]
                for symbol, engine in self.engines.items()
            }
        )
        benchmark = pd.DataFrame(
            {
                symbol: engine.data["Open_Return"].fillna(0.0)
                for symbol, engine in self.engines.items()
            }
        )
        first = self.data[self.symbols[0]]["Datetime"]
        n_days = (first.iloc[-1] - first.iloc[0]).days
        return _performance_metrics(
            returns.mul(weights, axis=1).sum(axis=1) / weights.sum(),
            benchmark.mul(weights, axis=1).sum(axis=1) / weights.sum(),
            self.weighted_portfolio_value,
            [engine.data["Position"] for engine in self.engines.values()],
            self.interval,
            n_days,
        )

    def plot_performance(self, mode: str = "basic", save_name: str | None = None):
        import matplotlib.pyplot as plt

        summary = self.get_performance_metrics()
        plt.title(
            f"Multi-Asset Backtesting Performance {summary['Total_Return'] * 100:.3f}% | Sharpe: {summary['Sharpe_Ratio']:.3f} | Sortino: {summary['Sortino_Ratio']:.3f}"
        )
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.plot(self.weighted_portfolio_value)
        if save_name:
            plt.savefig(save_name, dpi=150)
            plt.close()
        else:
            plt.show()
