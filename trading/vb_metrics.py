import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import statsmodels.api as sm

def stateful_position_to_multiplier(position: pd.Series) -> pd.Series:
    """Convert stateful position to multiplier."""
    position_multiplier = position.copy()
    position_multiplier[position == 1] = -1  # Short
    position_multiplier[position == 2] = 0   # Flat
    position_multiplier[position == 3] = 1   # Long

    # Handle hold positions (0) - use previous position's multiplier
    for i in range(len(position_multiplier)):
        if position.iloc[i] == 0 and i > 0:  # Hold
            position_multiplier.iloc[i] = position_multiplier.iloc[i-1]
        elif position.iloc[i] == 0 and i == 0:  # First position is hold, default to flat
            position_multiplier.iloc[i] = 0

    return position_multiplier

def get_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate strategy returns from position and close prices."""
    returns = close_prices.pct_change()
    position_multiplier = stateful_position_to_multiplier(position)
    return position_multiplier.shift(1) * returns

def get_cumulative_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate cumulative returns from position and close prices."""
    returns = get_returns(position, close_prices)
    return (1 + returns).cumprod()

def get_total_return(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate total return from position and close prices."""
    cumulative_returns = get_cumulative_returns(position, close_prices)
    return cumulative_returns.iloc[-1] - 1

def get_benchmark_returns(close_prices: pd.Series) -> pd.Series:
    """Calculate benchmark returns. (if you just held the asset)"""
    return (1 + close_prices.pct_change()).cumprod()

def get_benchmark_total_return(close_prices: pd.Series) -> float:
    """Calculate total return of benchmark."""
    benchmark_returns = get_benchmark_returns(close_prices)
    return benchmark_returns.iloc[-1] - 1

def get_portfolio_value(position: pd.Series, close_prices: pd.Series, initial_capital: float) -> pd.Series:
    """Calculate portfolio value from position, close prices, and initial capital."""
    cumulative_returns = get_cumulative_returns(position, close_prices)
    return initial_capital * cumulative_returns

def get_alpha_beta(position: pd.Series, close_prices: pd.Series, n_days: int = None, annualize: bool = True):
    """Calculate Jensen's alpha and beta using regression, annualized and scaled by simulation days."""
    strategy_returns = get_returns(position, close_prices)
    market_returns = close_prices.pct_change()

    # Align time
    common_index = strategy_returns.dropna().index.intersection(market_returns.dropna().index)
    strategy_returns = strategy_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]

    if len(strategy_returns) < 2 or len(market_returns) < 2:
        return float('nan'), float('nan')

    X = sm.add_constant(market_returns.values)
    y = strategy_returns.values
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]  # Intercept is Jensen's alpha
    beta = model.params[1]   # Slope is beta
    if annualize and n_days > 0:
        alpha = alpha * (365 / n_days)
    return alpha, beta

def get_alpha(position: pd.Series, close_prices: pd.Series, n_days: int = None, annualize: bool = True) -> float:
    """Return annualized Jensen's alpha."""
    alpha, _ = get_alpha_beta(position, close_prices, annualize=annualize, n_days=n_days)
    return alpha

def get_beta(position: pd.Series, close_prices: pd.Series) -> float:
    """Return beta from regression."""
    _, beta = get_alpha_beta(position, close_prices, annualize=False)
    return beta

def get_active_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate active returns of strategy compared to benchmark."""
    returns = get_total_return(position, close_prices)
    benchmark_returns = get_benchmark_total_return(close_prices)
    return returns - benchmark_returns

def get_drawdown(position: pd.Series, close_prices: pd.Series, initial_capital: float) -> pd.Series:
    """Calculate drawdown from position, close prices, and initial capital."""
    portfolio_value = get_portfolio_value(position, close_prices, initial_capital)
    peak = portfolio_value.cummax()
    return (portfolio_value - peak) / peak

def get_max_drawdown(position: pd.Series, close_prices: pd.Series, initial_capital: float) -> float:
    """Calculate maximum drawdown from position, close prices, and initial capital."""
    drawdown = get_drawdown(position, close_prices, initial_capital)
    return drawdown.min()

def get_sharpe_ratio(position: pd.Series, close_prices: pd.Series, risk_free_rate: float = 0.00, trading_days: int = 365) -> float:
    """Calculate Sharpe ratio from position, close prices, risk-free rate, and trading days."""
    returns = get_returns(position, close_prices)
    daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
    excess_returns = returns - daily_rf
    return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()

def get_sortino_ratio(position: pd.Series, close_prices: pd.Series, risk_free_rate: float = 0.00, trading_days: int = 365) -> float:
    """Calculate Sortino ratio from position, close prices, risk-free rate, and trading days."""
    returns = get_returns(position, close_prices)
    daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
    downside_returns = returns[returns < 0]
    return np.sqrt(trading_days) * (returns.mean() - daily_rf) / downside_returns.std()

def get_trade_pnls(position: pd.Series, close_prices: pd.Series) -> List[float]:
    """Calculate P&L for each trade from position and close prices."""
    pnl_list = []
    positions_and_entries = []  # Store (position_type, entry_price)
    
    position_changes = position.diff()
    
    for idx in range(len(position)):
        if pd.isna(position_changes.iloc[idx]):  # Skip first row (NaN)
            continue
            
        prev_pos = position.iloc[idx - 1]
        current_pos = position.iloc[idx]
        
        # Check if we're entering a new position (from flat or switching positions)
        if current_pos != 2 and prev_pos != current_pos:
            # If we had a previous position, close it first
            if prev_pos != 2 and positions_and_entries:
                position_type, entry_price = positions_and_entries.pop(0)
                exit_price = close_prices.iloc[idx]
                if position_type == 3:  # Long position
                    pnl = exit_price - entry_price
                else:  # Short position (position_type == 1)
                    pnl = entry_price - exit_price
                pnl_list.append(pnl)
            
            # Enter new position
            positions_and_entries.append((current_pos, close_prices.iloc[idx]))
            
        # Check if we're exiting to flat
        elif current_pos == 2 and prev_pos != 2 and positions_and_entries:
            position_type, entry_price = positions_and_entries.pop(0)
            exit_price = close_prices.iloc[idx]
            if position_type == 3:  # Long position
                pnl = exit_price - entry_price
            else:  # Short position (position_type == 1)
                pnl = entry_price - exit_price
            pnl_list.append(pnl)
    
    return pnl_list

def get_win_rate(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate win rate from position and close prices."""
    pnl_list = get_trade_pnls(position, close_prices)
    winning_trades = sum(1 for pnl in pnl_list if pnl > 0)
    total_trades = len(pnl_list)
    return winning_trades / total_trades if total_trades > 0 else 0

def get_rr_ratio(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate risk/reward ratio from position and close prices."""
    pnl_list = get_trade_pnls(position, close_prices)
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    return (avg_win / abs(avg_loss)) if avg_loss < 0 else 0

def get_breakeven_rate(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate breakeven rate from position and close prices."""
    rr_ratio = get_rr_ratio(position, close_prices)
    return 1 / (rr_ratio + 1) if rr_ratio > 0 else 0

def get_pt_ratio(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate profit/trade ratio from position and close prices."""
    returns = get_returns(position, close_prices)
    pnl_list = get_trade_pnls(position, close_prices)
    total_trades = len(pnl_list)
    return (returns.sum() / total_trades) * 100 if total_trades > 0 else 0

def get_profit_factor(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate profit factor from position and close prices."""
    returns = get_returns(position, close_prices)
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum())
    return profit_factor

def get_total_trades(position: pd.Series) -> int:
    """Calculate total number of trades from position."""
    # Use the trade PnL function and count the trades
    dummy_prices = pd.Series(range(len(position)), index=position.index)  # Dummy prices for counting
    pnl_list = get_trade_pnls(position, dummy_prices)
    return len(pnl_list)