import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import statsmodels.api as sm
import time

def stateful_position_to_multiplier(position: pd.Series) -> pd.Series:
    """Convert stateful position to multiplier."""
    start_time = time.time()
    
    # Vectorized implementation
    position_multiplier = position.copy().astype(float)
    position_multiplier[position == 1] = -1  # Short
    position_multiplier[position == 2] = 0   # Flat
    position_multiplier[position == 3] = 1   # Long
    position_multiplier[position == 0] = np.nan  # Hold positions
    
    # Forward fill to handle hold positions, default to 0 (flat) at start
    position_multiplier = position_multiplier.ffill().fillna(0)
    
    elapsed = time.time() - start_time
    print(f"[vb_metrics] stateful_position_to_multiplier: {elapsed:.4f}s")
    return position_multiplier

def get_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate strategy returns from position and close prices."""
    start_time = time.time()
    returns = close_prices.pct_change()
    position_multiplier = stateful_position_to_multiplier(position)
    result = position_multiplier.shift(1) * returns
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_returns: {elapsed:.4f}s")
    return result

def get_cumulative_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate cumulative returns from position and close prices."""
    start_time = time.time()
    returns = get_returns(position, close_prices)
    result = (1 + returns).cumprod()
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_cumulative_returns: {elapsed:.4f}s")
    return result

def get_total_return(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate total return from position and close prices."""
    start_time = time.time()
    cumulative_returns = get_cumulative_returns(position, close_prices)
    result = cumulative_returns.iloc[-1] - 1
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_total_return: {elapsed:.4f}s")
    return result

def get_benchmark_returns(close_prices: pd.Series) -> pd.Series:
    """Calculate benchmark returns. (if you just held the asset)"""
    start_time = time.time()
    result = (1 + close_prices.pct_change()).cumprod()
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_benchmark_returns: {elapsed:.4f}s")
    return result

def get_benchmark_total_return(close_prices: pd.Series) -> float:
    """Calculate total return of benchmark."""
    start_time = time.time()
    benchmark_returns = get_benchmark_returns(close_prices)
    result = benchmark_returns.iloc[-1] - 1
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_benchmark_total_return: {elapsed:.4f}s")
    return result

def get_portfolio_value(position: pd.Series, close_prices: pd.Series, initial_capital: float) -> pd.Series:
    """Calculate portfolio value from position, close prices, and initial capital."""
    start_time = time.time()
    cumulative_returns = get_cumulative_returns(position, close_prices)
    result = initial_capital * cumulative_returns
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_portfolio_value: {elapsed:.4f}s")
    return result

def get_alpha_beta(position: pd.Series, close_prices: pd.Series, n_days: int = None, annualize: bool = True):
    """Calculate Jensen's alpha and beta using regression, annualized and scaled by simulation days."""
    start_time = time.time()
    strategy_returns = get_returns(position, close_prices)
    market_returns = close_prices.pct_change()

    # Align time
    common_index = strategy_returns.dropna().index.intersection(market_returns.dropna().index)
    strategy_returns = strategy_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]

    if len(strategy_returns) < 2 or len(market_returns) < 2:
        elapsed = time.time() - start_time
        print(f"[vb_metrics] get_alpha_beta: {elapsed:.4f}s")
        return float('nan'), float('nan')

    X = sm.add_constant(market_returns.values)
    y = strategy_returns.values
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]  # Intercept is Jensen's alpha
    beta = model.params[1]   # Slope is beta
    if annualize and n_days > 0:
        alpha = alpha * (365 / n_days)
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_alpha_beta: {elapsed:.4f}s")
    return alpha, beta

def get_alpha(position: pd.Series, close_prices: pd.Series, n_days: int = None, annualize: bool = True) -> float:
    """Return annualized Jensen's alpha."""
    start_time = time.time()
    alpha, _ = get_alpha_beta(position, close_prices, annualize=annualize, n_days=n_days)
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_alpha: {elapsed:.4f}s")
    return alpha

def get_beta(position: pd.Series, close_prices: pd.Series) -> float:
    """Return beta from regression."""
    start_time = time.time()
    _, beta = get_alpha_beta(position, close_prices, annualize=False)
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_beta: {elapsed:.4f}s")
    return beta

def get_active_returns(position: pd.Series, close_prices: pd.Series) -> pd.Series:
    """Calculate active returns of strategy compared to benchmark."""
    start_time = time.time()
    returns = get_total_return(position, close_prices)
    benchmark_returns = get_benchmark_total_return(close_prices)
    result = returns - benchmark_returns
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_active_returns: {elapsed:.4f}s")
    return result

def get_drawdown(position: pd.Series, close_prices: pd.Series, initial_capital: float) -> pd.Series:
    """Calculate drawdown from position, close prices, and initial capital."""
    start_time = time.time()
    portfolio_value = get_portfolio_value(position, close_prices, initial_capital)
    peak = portfolio_value.cummax()
    result = (portfolio_value - peak) / peak
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_drawdown: {elapsed:.4f}s")
    return result

def get_max_drawdown(position: pd.Series, close_prices: pd.Series, initial_capital: float) -> float:
    """Calculate maximum drawdown from position, close prices, and initial capital."""
    start_time = time.time()
    drawdown = get_drawdown(position, close_prices, initial_capital)
    result = drawdown.min()
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_max_drawdown: {elapsed:.4f}s")
    return result

def get_sharpe_ratio(position: pd.Series, close_prices: pd.Series, risk_free_rate: float = 0.00, trading_days: int = 365) -> float:
    """Calculate Sharpe ratio from position, close prices, risk-free rate, and trading days."""
    start_time = time.time()
    returns = get_returns(position, close_prices)
    daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
    excess_returns = returns - daily_rf
    result = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_sharpe_ratio: {elapsed:.4f}s")
    return result

def get_sortino_ratio(position: pd.Series, close_prices: pd.Series, risk_free_rate: float = 0.00, trading_days: int = 365) -> float:
    """Calculate Sortino ratio from position, close prices, risk-free rate, and trading days."""
    start_time = time.time()
    returns = get_returns(position, close_prices)
    daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
    downside_returns = returns[returns < 0]
    result = np.sqrt(trading_days) * (returns.mean() - daily_rf) / downside_returns.std()
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_sortino_ratio: {elapsed:.4f}s")
    return result

def get_trade_pnls(position: pd.Series, close_prices: pd.Series) -> List[float]:
    """Calculate P&L for each trade from position and close prices."""
    start_time = time.time()
    
    # Vectorized implementation
    position_changes = position.diff()
    change_indices = position_changes[position_changes != 0].index
    
    if len(change_indices) == 0:
        elapsed = time.time() - start_time
        print(f"[vb_metrics] get_trade_pnls: {elapsed:.4f}s")
        return []
    
    pnl_list = []
    active_positions = []  # Stack of (position_type, entry_price, entry_idx)
    
    # Process all position changes
    prev_pos = position.iloc[0] if len(position) > 0 else 2
    
    for idx in change_indices:
        current_pos = position.loc[idx]
        
        # Close existing position if switching or going to flat
        if prev_pos != 2 and prev_pos != current_pos:
            if active_positions:
                pos_type, entry_price, _ = active_positions.pop()
                exit_price = close_prices.loc[idx]
                if pos_type == 3:  # Long
                    pnl = exit_price - entry_price
                else:  # Short (pos_type == 1)
                    pnl = entry_price - exit_price
                pnl_list.append(pnl)
        
        # Open new position if not going to flat
        if current_pos != 2:
            entry_price = close_prices.loc[idx]
            active_positions.append((current_pos, entry_price, idx))
        
        prev_pos = current_pos
    
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_trade_pnls: {elapsed:.4f}s")
    return pnl_list

def get_win_rate(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate win rate from position and close prices."""
    start_time = time.time()
    pnl_list = get_trade_pnls(position, close_prices)
    winning_trades = sum(1 for pnl in pnl_list if pnl > 0)
    total_trades = len(pnl_list)
    result = winning_trades / total_trades if total_trades > 0 else 0
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_win_rate: {elapsed:.4f}s")
    return result

def get_rr_ratio(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate risk/reward ratio from position and close prices."""
    start_time = time.time()
    pnl_list = get_trade_pnls(position, close_prices)
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    result = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_rr_ratio: {elapsed:.4f}s")
    return result

def get_breakeven_rate(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate breakeven rate from position and close prices."""
    start_time = time.time()
    rr_ratio = get_rr_ratio(position, close_prices)
    result = 1 / (rr_ratio + 1) if rr_ratio > 0 else 0
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_breakeven_rate: {elapsed:.4f}s")
    return result

def get_pt_ratio(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate profit/trade ratio from position and close prices."""
    start_time = time.time()
    returns = get_returns(position, close_prices)
    pnl_list = get_trade_pnls(position, close_prices)
    total_trades = len(pnl_list)
    result = (returns.sum() / total_trades) * 100 if total_trades > 0 else 0
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_pt_ratio: {elapsed:.4f}s")
    return result

def get_profit_factor(position: pd.Series, close_prices: pd.Series) -> float:
    """Calculate profit factor from position and close prices."""
    start_time = time.time()
    returns = get_returns(position, close_prices)
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum())
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_profit_factor: {elapsed:.4f}s")
    return profit_factor

def get_total_trades(position: pd.Series) -> int:
    """Calculate total number of trades from position."""
    start_time = time.time()
    dummy_prices = pd.Series(range(len(position)), index=position.index)  # Dummy prices for counting
    pnl_list = get_trade_pnls(position, dummy_prices)
    result = len(pnl_list)
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_total_trades: {elapsed:.4f}s")
    return result

def get_rr_ratio_from_pnls(pnl_list: List[float]) -> float:
    """Calculate risk/reward ratio from pre-calculated PnL list."""
    start_time = time.time()
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    result = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_rr_ratio_from_pnls: {elapsed:.4f}s")
    return result

def get_breakeven_rate_from_pnls(pnl_list: List[float]) -> float:
    """Calculate breakeven rate from pre-calculated PnL list."""
    start_time = time.time()
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    rr_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
    result = 1 / (rr_ratio + 1) if rr_ratio > 0 else 0
    elapsed = time.time() - start_time
    print(f"[vb_metrics] get_breakeven_rate_from_pnls: {elapsed:.4f}s")
    return result