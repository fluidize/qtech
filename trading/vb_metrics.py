import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def get_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate strategy returns from position and close prices."""
    returns = close_prices.pct_change()
    return position.shift(1) * returns

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

def get_alpha(position: pd.Series, close_prices: pd.Series, annualize: bool = True) -> float:
    """Calculate CAPM alpha using average daily returns. Optionally annualize (default: True)."""
    strategy_returns = get_returns(position, close_prices)
    market_returns = close_prices.pct_change()

    # Align time
    common_index = strategy_returns.dropna().index.intersection(market_returns.dropna().index)
    strategy_returns = strategy_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]

    beta = strategy_returns.cov(market_returns) / market_returns.var() if market_returns.var() != 0 else 0
    avg_strategy_return = strategy_returns.mean()
    avg_market_return = market_returns.mean()
    alpha = avg_strategy_return - beta * avg_market_return
    if annualize:
        alpha = alpha * 365
    return alpha

def get_beta(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate beta from position and close prices."""
    # Get strategy returns (these are already aligned with position)
    strategy_returns = get_returns(position, close_prices)
    
    # Get market returns (not cumulative, just the daily percentage changes)
    market_returns = close_prices.pct_change()
    
    # Make sure both return series have the same index
    common_index = strategy_returns.dropna().index.intersection(market_returns.dropna().index)
    
    # Align both returns to the common timeframe
    aligned_strategy_returns = strategy_returns.loc[common_index]
    aligned_market_returns = market_returns.loc[common_index]
    
    # Calculate beta using covariance/variance
    covariance = aligned_strategy_returns.cov(aligned_market_returns)
    variance = aligned_market_returns.var()
    
    if variance == 0:
        return 0  # Avoid division by zero
        
    beta = covariance / variance
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
    entry_prices = []
    
    position_changes = position.diff()
    for idx in range(len(position)):
        if position_changes[idx] == 1:
            entry_prices.append(close_prices.iloc[idx])
        elif position_changes[idx] == -1 and entry_prices:
            entry_price = entry_prices.pop(0)
            exit_price = close_prices.iloc[idx]
            pnl = exit_price - entry_price
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
    position_changes = position.diff()
    return (position_changes != 0).sum() // 2  # Each trade has an entry and exit