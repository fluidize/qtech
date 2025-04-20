import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
import gym
from gym import spaces
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append("trading")
import pandas_indicators as ta
from spot_trade_env import TradingEnvironment, Portfolio

class SpotTradingEnv(gym.Env):
    """
    A Gym environment for cryptocurrency spot trading using reinforcement learning.
    This wrapper interfaces with the TradingEnvironment class from spot_trade_env.py.
    """
    
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        initial_capital: float = 10000.0,
        chunks: int = 1,
        interval: str = "5min",
        age_days: int = 10,
        window_size: int = 20,
        commission_rate: float = 0.001,
        reward_scaling: float = 1.0,
        include_position: bool = True,
        use_kucoin: bool = True
    ):
        """
        Initialize the trading environment
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USDT")
            initial_capital: Starting capital amount
            chunks: Number of data chunks to fetch
            interval: Trading interval (e.g., "5min", "1h")
            age_days: Age of data in days
            window_size: Number of past observations to include in state
            commission_rate: Trading fee percentage
            reward_scaling: Scalar to adjust reward magnitude
            include_position: Whether to include current position in state
            use_kucoin: Whether to use Kucoin data source
        """
        super(SpotTradingEnv, self).__init__()
        
        # Initialize the trading environment
        self.trading_env = TradingEnvironment(
            symbol=symbol,
            instance_name="rl_trader",
            initial_capital=initial_capital,
            chunks=chunks,
            interval=interval,
            age_days=age_days
        )
        
        # Set parameters
        self.symbol = symbol
        self.window_size = window_size
        self.commission_rate = commission_rate
        self.reward_scaling = reward_scaling
        self.include_position = include_position
        self.use_kucoin = use_kucoin
        
        # Track metrics
        self.episode_reward = 0.0
        self.last_portfolio_value = initial_capital
        self.returns_history = []
        self.position_history = []
        
        # Define action and observation spaces
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space depends on the features we extract
        # We'll define this after initializing data to know feature dimensions
        self._initialize_data()
        
    def _initialize_data(self):
        """
        Fetch the data and initialize the observation space based on feature dimensions
        """
        # Fetch data
        self.trading_env.fetch_data(kucoin=self.use_kucoin)
        
        # Extract features for a sample state to determine observation space dimension
        sample_state = self._get_observation()
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=sample_state.shape, 
            dtype=np.float32
        )
        
    def _get_observation(self) -> np.ndarray:
        """
        Extract features from the current state
        
        Returns:
            Normalized feature vector representing the current state
        """
        # Get context data (historical price data)
        context = self.trading_env.context.copy()
        
        # Extract basic OHLCV features
        ohlcv = context[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # Calculate returns
        returns = context['Close'].pct_change().fillna(0).values
        log_returns = ta.log_returns(context['Close']).fillna(0).values
        
        # Calculate moving averages
        ma_short = ta.sma(context['Close'], timeperiod=10).bfill().values
        ma_long = ta.sma(context['Close'], timeperiod=50).bfill().values
        ma_cross = ma_short - ma_long
        
        rsi = ta.rsi(context['Close'], timeperiod=14)
        
        # Normalize data
        normalized_close = context['Close'] / context['Close'].iloc[0]
        volatility = context['Close'].rolling(window=20).std().fillna(0) / context['Close']
        
        # Combine features
        features = np.column_stack([
            ohlcv[-self.window_size:] / ohlcv[-self.window_size:, 3:4],  # Normalize by close price
            returns[-self.window_size:],
            log_returns[-self.window_size:],
            ma_cross[-self.window_size:] / ma_long[-self.window_size:],
            rsi[-self.window_size:] / 100,  # Scale to [0, 1]
            volatility[-self.window_size:]
        ])
        
        features = features.flatten().astype(np.float32)
        
        if self.include_position:
            # Encode position: 0 = no position, 1 = long position
            position = 1.0 if self.symbol in self.trading_env.portfolio.positions else 0.0
            features = np.append(features, position)
            
            # Add portfolio metrics
            portfolio_value_ratio = self.trading_env.portfolio.total_value / self.trading_env.portfolio.initial_capital
            cash_ratio = self.trading_env.portfolio.cash / self.trading_env.portfolio.total_value
            
            features = np.append(features, [portfolio_value_ratio, cash_ratio])
        
        return features
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial observation
        """
        self.trading_env.reset()
        self.episode_reward = 0.0
        self.last_portfolio_value = self.trading_env.portfolio.total_value
        self.returns_history = []
        self.position_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment
        
        Args:
            action: The action to take (0 = Hold, 1 = Buy, 2 = Sell)
            
        Returns:
            observation: The new state observation
            reward: The reward for taking the action
            done: Whether the episode is complete
            info: Additional information
        """
        # Get current price and portfolio value
        current_price = self.trading_env.get_current_price()
        current_portfolio_value = self.trading_env.portfolio.total_value
        
        # Get current position state
        has_position = self.symbol in self.trading_env.portfolio.positions
        
        # Execute action
        if action == 1 and not has_position:  # Buy
            self.trading_env.portfolio.buy_max(
                self.symbol, 
                current_price["Close"], 
                self.trading_env.get_current_timestamp()
            )
        elif action == 2 and has_position:  # Sell
            self.trading_env.portfolio.sell_max(
                self.symbol, 
                current_price["Close"], 
                self.trading_env.get_current_timestamp()
            )
            
        # Update environment
        done = not self.trading_env.step()
        
        # Get new state
        new_observation = self._get_observation()
        
        ### REWARD FUNCTION
        new_portfolio_value = self.trading_env.portfolio.total_value
        reward = new_portfolio_value / self.last_portfolio_value * self.reward_scaling
        
        # Additional metrics for tracking
        self.episode_reward += reward
        self.last_portfolio_value = new_portfolio_value
        
        # Update position history
        current_position = 1.0 if self.symbol in self.trading_env.portfolio.positions else 0.0
        self.position_history.append(current_position)
        
        # Store return for this step
        self.returns_history.append(reward)
        
        # Info dictionary with metrics
        info = {
            "portfolio_value": new_portfolio_value,
            "portfolio_return": (new_portfolio_value / self.trading_env.portfolio.initial_capital) - 1.0,
            "episode_reward": self.episode_reward,
            "position": current_position,
            "price": current_price["Close"],
            "timestamp": self.trading_env.get_current_timestamp()
        }
        
        return new_observation, reward, done, info
    
    def render(self, mode="human"):
        """
        Render the environment
        
        Args:
            mode: The rendering mode
        """
        if mode == "human":
            console = Console()
            table = Table(show_header=True)
            
            # Add columns
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            
            # Get data
            portfolio_value = self.trading_env.portfolio.total_value
            initial_capital = self.trading_env.portfolio.initial_capital
            return_pct = ((portfolio_value / initial_capital) - 1.0) * 100
            current_price = self.trading_env.get_current_price()["Close"]
            timestamp = self.trading_env.get_current_timestamp()
            position = 'Long' if self.symbol in self.trading_env.portfolio.positions else 'None'
            
            # Calculate additional metrics
            cash = self.trading_env.portfolio.cash
            cash_ratio = (cash / portfolio_value) * 100
            
            # Add rows
            table.add_row("Symbol", self.symbol)
            table.add_row("Timestamp", str(timestamp))
            table.add_row("Current Price", f"${current_price:.2f}")
            table.add_row("Portfolio Value", f"${portfolio_value:.2f}")
            table.add_row("Initial Capital", f"${initial_capital:.2f}")
            table.add_row("Return", f"{return_pct:.2f}%", style="green" if return_pct > 0 else "red")
            table.add_row("Cash", f"${cash:.2f}")
            table.add_row("Cash Ratio", f"{cash_ratio:.2f}%")
            table.add_row("Position", position, style="green" if position == 'Long' else "")
            
            # Add most recent return value
            if len(self.returns_history) > 0:
                last_return = self.returns_history[-1] * 100
                table.add_row("Last Step Return", f"{last_return:.4f}%", 
                              style="green" if last_return > 0 else "red")
            
            # Create and print the panel
            title = f"[bold]{self.symbol} Trading Environment[/bold]"
            console.print(Panel(table, title=title, expand=False))
            
            return
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")
            
    def close(self):
        """
        Clean up environment resources
        """
        pass
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate and return performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        portfolio_value = self.trading_env.portfolio.total_value
        initial_capital = self.trading_env.portfolio.initial_capital
        total_return = (portfolio_value / initial_capital) - 1.0
        
        # Calculate more comprehensive metrics
        trading_days = 365  # Annualization factor
        
        # Convert returns to numpy array for calculations
        returns_array = np.array(self.returns_history)
        
        # Calculate drawdown
        cumulative_returns = (1 + returns_array).cumprod()
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe and Sortino ratios
        risk_free_rate = 0.00  # Could be adjusted based on current rates
        daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
        
        if len(returns_array) > 0:
            excess_returns = returns_array - daily_rf
            sharpe_ratio = np.sqrt(trading_days) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            downside_returns = returns_array[returns_array < 0]
            sortino_ratio = np.sqrt(trading_days) * (np.mean(returns_array) - daily_rf) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calculate win rate and other trade metrics
        position_changes = np.diff(self.position_history + [0])  # Add 0 at end to capture final position change
        
        # Trades are when position goes from 0->1 (entry) and 1->0 (exit)
        entries = np.where(position_changes > 0)[0]
        exits = np.where(position_changes < 0)[0]
        
        # Calculate trade-level metrics
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        # Calculate pnl for each completed trade
        pnl_list = []
        
        # Must have both entries and exits
        if len(entries) > 0 and len(exits) > 0:
            # Process only completed trades
            num_completed_trades = min(len(entries), len(exits))
            
            for i in range(num_completed_trades):
                # Calculate trade P&L
                entry_idx = entries[i]
                exit_idx = exits[i] if i < len(exits) else -1
                
                if exit_idx > entry_idx:  # Valid trade
                    trade_returns = np.sum(returns_array[entry_idx:exit_idx+1])
                    pnl_list.append(trade_returns)
                    
                    if trade_returns > 0:
                        winning_trades += 1
                        total_profit += trade_returns
                    else:
                        losing_trades += 1
                        total_loss += trade_returns
        
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate RR ratio - average win size to average loss size
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        rr_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
        
        # Breakeven rate
        be_rate = 1 / (rr_ratio + 1) if rr_ratio > 0 else 0
        
        # Profit factor
        profit_factor = abs(total_profit / total_loss) if total_loss < 0 else 0
        
        # Per trade ratio - average return per trade
        pt_ratio = (np.sum(returns_array) / total_trades) * 100 if total_trades > 0 else 0
        
        return {
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Breakeven Rate': be_rate,
            'RR Ratio': rr_ratio,
            'PT Ratio': pt_ratio,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio, 
            'Sortino Ratio': sortino_ratio,
            'Total Trades': total_trades,
            'Portfolio Value': portfolio_value,
            'Initial Capital': initial_capital
        }
    
    def plot_performance(self, show_graph=True, advanced=False):
        """
        Plot the trading performance
        
        Args:
            show_graph: Whether to display the graph
            advanced: Whether to show advanced performance plots
            
        Returns:
            The plotly figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Get performance metrics
        metrics = self.get_performance_metrics()
        
        if not advanced:
            # Get data from trading environment
            portfolio_df = self.trading_env.portfolio.history
            
            # Convert position history to a pandas series
            position_df = pd.DataFrame({
                'Position': self.position_history
            }, index=portfolio_df.index[:len(self.position_history)])
            
            # Create figure
            fig = go.Figure()
            
            # Add portfolio value trace
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['total_value'],
                mode='lines',
                name='Strategy Portfolio Value',
                line=dict(color='green')
            ))
            
            # Add buy signals
            buy_signals = position_df.diff()
            buy_points = buy_signals[buy_signals['Position'] > 0].index
            if len(buy_points) > 0:
                fig.add_trace(go.Scatter(
                    x=buy_points,
                    y=portfolio_df.loc[buy_points, 'total_value'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='triangle-up'
                    )
                ))
            
            # Add sell signals
            sell_signals = position_df.diff()
            sell_points = sell_signals[sell_signals['Position'] < 0].index
            if len(sell_points) > 0:
                fig.add_trace(go.Scatter(
                    x=sell_points,
                    y=portfolio_df.loc[sell_points, 'total_value'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-down'
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title=f"{self.symbol} | TR: {metrics['Total Return']*100:.2f}% | Max DD: {metrics['Max Drawdown']*100:.2f}% | WR: {metrics['Win Rate']*100:.2f}% | Sharpe: {metrics['Sharpe Ratio']:.2f} | Trades: {metrics['Total Trades']}",
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
        else:
            # Advanced plotting
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Equity Curve", "Drawdown",
                    "Returns Distribution", "Positions",
                    "Win Rate", "Trade Returns"
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Get data from trading environment
            portfolio_df = self.trading_env.portfolio.history
            
            # Convert position history and returns to pandas series
            position_df = pd.DataFrame({
                'Position': self.position_history
            }, index=portfolio_df.index[:len(self.position_history)])
            
            returns_df = pd.DataFrame({
                'Returns': self.returns_history
            }, index=portfolio_df.index[:len(self.returns_history)])
            
            # 1. Equity Curve - Row 1, Col 1
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['total_value'],
                    mode='lines',
                    name='Portfolio Value'
                ),
                row=1, col=1
            )
            
            # 2. Drawdown - Row 1, Col 2
            cumulative_returns = (1 + returns_df['Returns']).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            # 3. Returns Distribution - Row 2, Col 1
            fig.add_trace(
                go.Histogram(
                    x=returns_df['Returns'] * 100,
                    name='Returns Distribution %',
                    nbinsx=50
                ),
                row=2, col=1
            )
            
            # 4. Positions - Row 2, Col 2
            fig.add_trace(
                go.Scatter(
                    x=position_df.index,
                    y=position_df['Position'],
                    mode='lines',
                    name='Position',
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
            
            # 5. Win Rate - Row 3, Col 1
            # Calculate cumulative win rate
            if len(self.position_history) > 0:
                position_changes = np.diff(self.position_history + [0])
                entries = np.where(position_changes > 0)[0]
                exits = np.where(position_changes < 0)[0]
                
                win_rates = []
                win_count = 0
                trade_count = 0
                
                win_rate_index = []
                
                # Process only completed trades
                if len(entries) > 0 and len(exits) > 0:
                    num_completed_trades = min(len(entries), len(exits))
                    
                    for i in range(num_completed_trades):
                        entry_idx = entries[i]
                        exit_idx = exits[i] if i < len(exits) else -1
                        
                        if exit_idx > entry_idx:  # Valid trade
                            trade_return = np.sum(np.array(self.returns_history)[entry_idx:exit_idx+1])
                            trade_count += 1
                            
                            if trade_return > 0:
                                win_count += 1
                                
                            win_rate = win_count / trade_count if trade_count > 0 else 0
                            win_rates.append(win_rate)
                            win_rate_index.append(portfolio_df.index[exit_idx])
                    
                    if win_rates:
                        win_rate_df = pd.DataFrame({
                            'Win Rate': win_rates
                        }, index=win_rate_index)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=win_rate_df.index,
                                y=win_rate_df['Win Rate'] * 100,
                                mode='lines',
                                name='Win Rate %'
                            ),
                            row=3, col=1
                        )
                        
                        # Add breakeven line
                        be_rate = metrics['Breakeven Rate']
                        fig.add_trace(
                            go.Scatter(
                                x=[win_rate_df.index[0], win_rate_df.index[-1]],
                                y=[be_rate * 100, be_rate * 100],
                                mode='lines',
                                name=f'Breakeven ({be_rate*100:.1f}%)',
                                line=dict(dash='dash', color='red')
                            ),
                            row=3, col=1
                        )
            
            # 6. Trade Returns - Row 3, Col 2
            if len(self.position_history) > 0:
                position_changes = np.diff(self.position_history + [0])
                entries = np.where(position_changes > 0)[0]
                exits = np.where(position_changes < 0)[0]
                
                trade_returns = []
                trade_index = []
                
                # Process only completed trades
                if len(entries) > 0 and len(exits) > 0:
                    num_completed_trades = min(len(entries), len(exits))
                    
                    for i in range(num_completed_trades):
                        entry_idx = entries[i]
                        exit_idx = exits[i] if i < len(exits) else -1
                        
                        if exit_idx > entry_idx:  # Valid trade
                            trade_return = np.sum(np.array(self.returns_history)[entry_idx:exit_idx+1]) * 100
                            trade_returns.append(trade_return)
                            trade_index.append(i+1)  # Trade number
                    
                    if trade_returns:
                        fig.add_trace(
                            go.Bar(
                                x=trade_index,
                                y=trade_returns,
                                name='Trade Returns %',
                                marker_color=['green' if x > 0 else 'red' for x in trade_returns]
                            ),
                            row=3, col=2
                        )
            
            # Update layout
            fig.update_layout(
                title=f"{self.symbol} Performance | TR: {metrics['Total Return']*100:.2f}% | Max DD: {metrics['Max Drawdown']*100:.2f}% | WR: {metrics['Win Rate']*100:.2f}% | RR: {metrics['RR Ratio']:.2f} | Sharpe: {metrics['Sharpe Ratio']:.2f}",
                showlegend=True,
                template="plotly_dark",
                height=900,
                width=1200
            )
        
        if show_graph:
            fig.show()
            
        return fig