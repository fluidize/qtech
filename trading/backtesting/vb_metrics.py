import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import statsmodels.api as sm

def stateful_position_to_multiplier(position: pd.Series) -> pd.Series:
    """Convert stateful position to multiplier."""
    
    # Vectorized implementation
    position_multiplier = position.copy().astype(float)
    position_multiplier[position == 1] = -1  # Short
    position_multiplier[position == 2] = 0   # Flat
    position_multiplier[position == 3] = 1   # Long
    position_multiplier[position == 0] = np.nan  # Hold positions
    
    # Forward fill to handle hold positions, default to 0 (flat) at start
    position_multiplier = position_multiplier.ffill().fillna(0)
    
    return position_multiplier

def get_returns(position: pd.Series, open_prices: pd.Series) -> pd.Series:
    """Calculate strategy returns from position and open prices."""
    returns = open_prices.pct_change()
    position_multiplier = stateful_position_to_multiplier(position)
    result = position_multiplier.shift(1) * returns
    return result

def get_cumulative_returns(position: pd.Series, open_prices: pd.Series) -> pd.Series:
    """Calculate cumulative returns from position and open prices."""
    returns = get_returns(position, open_prices)
    result = (1 + returns).cumprod()
    return result

def get_total_return(position: pd.Series, open_prices: pd.Series) -> float:
    """Calculate total return from position and open prices."""
    cumulative_returns = get_cumulative_returns(position, open_prices)
    result = cumulative_returns.iloc[-1] - 1
    return result

def get_benchmark_returns(open_prices: pd.Series) -> pd.Series:
    """Calculate benchmark returns per period using open prices."""
    return open_prices.pct_change()

def get_benchmark_total_return(open_prices: pd.Series) -> float:
    """Calculate benchmark total return using open prices."""
    returns = get_benchmark_returns(open_prices).dropna()
    cum_return = (1 + returns).prod() - 1
    return cum_return

def get_portfolio_value(position: pd.Series, open_prices: pd.Series, initial_capital: float) -> pd.Series:
    """Calculate portfolio value from position, open prices, and initial capital."""
    cumulative_returns = get_cumulative_returns(position, open_prices)
    result = initial_capital * cumulative_returns
    return result

def get_alpha_beta(strategy_returns: pd.Series, market_returns: pd.Series, n_days: int, return_interval: str = None):
    """Calculate annualized Jensen's alpha and beta using regression with actual strategy returns."""
    # Align time
    common_index = strategy_returns.dropna().index.intersection(market_returns.dropna().index)
    strategy_returns_aligned = strategy_returns.loc[common_index]
    market_returns_aligned = market_returns.loc[common_index]

    if len(strategy_returns_aligned) < 2 or len(market_returns_aligned) < 2:
        return float('nan'), float('nan')

    X = sm.add_constant(market_returns_aligned.values)
    y = strategy_returns_aligned.values
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]  # Intercept is Jensen's alpha (per period)
    beta = model.params[1]   # Slope is beta

    # Annualize alpha properly
    if return_interval == '1m':
        periods_per_year = 365 * 24 * 60  # 525,600 minutes per year
    elif return_interval == '3m':
        periods_per_year = 365 * 24 * 20  # 175,200 3-minute periods per year
    elif return_interval == '5m':
        periods_per_year = 365 * 24 * 12  # 105,120 5-minute periods per year
    elif return_interval == '15m':
        periods_per_year = 365 * 24 * 4   # 35,040 15-minute periods per year
    elif return_interval == '30m':
        periods_per_year = 365 * 24 * 2   # 17,520 30-minute periods per year
    elif return_interval == '1h':
        periods_per_year = 365 * 24       # 8,760 hours per year
    elif return_interval == '4h':
        periods_per_year = 365 * 6        # 2,190 4-hour periods per year
    elif return_interval == '1d':
        periods_per_year = 365             # 365 days per year
    else:
        periods_per_year = 252  # Default to trading days

    alpha_annualized = alpha * np.sqrt(periods_per_year)

    return alpha_annualized, beta

def get_alpha(strategy_returns: pd.Series, market_returns: pd.Series, n_days: int, return_interval: str = None) -> float:
    """Return annualized Jensen's alpha using actual strategy returns."""
    alpha, _ = get_alpha_beta(strategy_returns, market_returns, n_days=n_days, return_interval=return_interval)
    return alpha

def get_beta(strategy_returns: pd.Series, market_returns: pd.Series, n_days: int, return_interval: str = None) -> float:
    """Return beta from regression using actual strategy returns."""
    _, beta = get_alpha_beta(strategy_returns, market_returns, n_days=n_days, return_interval=return_interval)
    return beta

def get_active_returns(position: pd.Series, open_prices: pd.Series) -> pd.Series:
    """Calculate the active return of strategy compared to benchmark (per period)."""
    returns = get_returns(position, open_prices)
    benchmark_returns = get_benchmark_returns(open_prices)
    return returns - benchmark_returns

def get_total_active_return(position: pd.Series, open_prices: pd.Series) -> pd.Series:
    """Calculate the total active return of strategy compared to benchmark."""
    return_total = get_total_return(position, open_prices)
    return_benchmark = get_benchmark_total_return(open_prices)
    result = return_total - return_benchmark
    return result

def get_drawdown(position: pd.Series, open_prices: pd.Series, initial_capital: float) -> pd.Series:
    """Calculate drawdown from position, open prices, and initial capital."""
    portfolio_value = get_portfolio_value(position, open_prices, initial_capital)
    peak = portfolio_value.cummax()
    result = (portfolio_value - peak) / peak
    return result

def get_max_drawdown(position: pd.Series, open_prices: pd.Series, initial_capital: float) -> float:
    """Calculate maximum drawdown from position, open prices, and initial capital."""
    drawdown = get_drawdown(position, open_prices, initial_capital)
    result = drawdown.min()
    return result

def get_sharpe_ratio(strategy_returns: pd.Series, return_interval: str, n_days: int, risk_free_rate: float = 0.00) -> float:
    """Calculate annualized Sharpe ratio using actual strategy returns that include costs."""
    excess_returns = strategy_returns - risk_free_rate
    if excess_returns.std(ddof=1) == 0:
        return float('nan')
    
    # Calculate per-period Sharpe ratio
    sharpe_per_period = excess_returns.mean() / excess_returns.std(ddof=1)

    # Determine periods per year for annualization
    if return_interval == '1m':
        periods_per_year = 365 * 24 * 60  # 525,600 minutes per year
    elif return_interval == '3m':
        periods_per_year = 365 * 24 * 20  # 175,200 3-minute periods per year
    elif return_interval == '5m':
        periods_per_year = 365 * 24 * 12  # 105,120 5-minute periods per year
    elif return_interval == '15m':
        periods_per_year = 365 * 24 * 4   # 35,040 15-minute periods per year
    elif return_interval == '30m':
        periods_per_year = 365 * 24 * 2   # 17,520 30-minute periods per year
    elif return_interval == '1h':
        periods_per_year = 365 * 24       # 8,760 hours per year
    elif return_interval == '4h':
        periods_per_year = 365 * 6        # 2,190 4-hour periods per year
    elif return_interval == '1d':
        periods_per_year = 365             # 365 days per year
    else:
        periods_per_year = 252  # Default to trading days

    # Annualize Sharpe ratio: sharpe_annual = sharpe_per_period * sqrt(periods_per_year)
    sharpe_annualized = sharpe_per_period * np.sqrt(periods_per_year)
    
    return sharpe_annualized

def get_sortino_ratio(strategy_returns: pd.Series, return_interval: str, n_days: int, risk_free_rate: float = 0.00) -> float:
    """Calculate annualized Sortino ratio using actual strategy returns that include costs."""
    excess_returns = strategy_returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std(ddof=1) == 0:
        return float('nan')
    
    # Calculate per-period Sortino ratio
    sortino_per_period = excess_returns.mean() / downside_returns.std(ddof=1)

    # Determine periods per year for annualization
    if return_interval == '1m':
        periods_per_year = 365 * 24 * 60  # 525,600 minutes per year
    elif return_interval == '3m':
        periods_per_year = 365 * 24 * 20  # 175,200 3-minute periods per year
    elif return_interval == '5m':
        periods_per_year = 365 * 24 * 12  # 105,120 5-minute periods per year
    elif return_interval == '15m':
        periods_per_year = 365 * 24 * 4   # 35,040 15-minute periods per year
    elif return_interval == '30m':
        periods_per_year = 365 * 24 * 2   # 17,520 30-minute periods per year
    elif return_interval == '1h':
        periods_per_year = 365 * 24       # 8,760 hours per year
    elif return_interval == '4h':
        periods_per_year = 365 * 6        # 2,190 4-hour periods per year
    elif return_interval == '1d':
        periods_per_year = 365             # 365 days per year
    else:
        periods_per_year = 252  # Default to trading days

    # Annualize Sortino ratio: sortino_annual = sortino_per_period * sqrt(periods_per_year)
    sortino_annualized = sortino_per_period * np.sqrt(periods_per_year)
    
    return sortino_annualized

def get_trade_pnls(position: pd.Series, open_prices: pd.Series) -> List[float]:
    """Calculate P&L for each trade from position and open prices."""
    
    # Vectorized implementation
    position_changes = position.diff()
    change_indices = position_changes[position_changes != 0].index
    
    if len(change_indices) == 0:
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
                exit_price = open_prices.loc[idx]
                if pos_type == 3:  # Long
                    pnl = exit_price - entry_price
                else:  # Short (pos_type == 1)
                    pnl = entry_price - exit_price
                pnl_list.append(pnl)
        
        # Open new position if not going to flat
        if current_pos != 2:
            entry_price = open_prices.loc[idx]
            active_positions.append((current_pos, entry_price, idx))
        
        prev_pos = current_pos
    
    return pnl_list

def get_win_rate(position: pd.Series, open_prices: pd.Series) -> float:
    """Calculate win rate from position and open prices."""
    pnl_list = get_trade_pnls(position, open_prices)
    winning_trades = sum(1 for pnl in pnl_list if pnl > 0)
    total_trades = len(pnl_list)
    result = winning_trades / total_trades if total_trades > 0 else 0
    return result

def get_rr_ratio(position: pd.Series, open_prices: pd.Series) -> float:
    """Calculate risk/reward ratio from position and open prices."""
    pnl_list = get_trade_pnls(position, open_prices)
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    result = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
    return result

def get_breakeven_rate(position: pd.Series, open_prices: pd.Series) -> float:
    """Calculate breakeven rate from position and open prices."""
    rr_ratio = get_rr_ratio(position, open_prices)
    result = 1 / (rr_ratio + 1) if rr_ratio > 0 else 0
    return result

def get_pt_ratio(position: pd.Series, open_prices: pd.Series) -> float:
    """Calculate average profit/trade ratio from position and open prices."""
    total_return = get_total_return(position, open_prices)
    total_trades = get_total_trades(position)
    result = (total_return / total_trades) if total_trades > 0 else 0
    return result

def get_profit_factor(position: pd.Series, open_prices: pd.Series) -> float:
    """Calculate profit factor from position and open prices."""
    returns = get_returns(position, open_prices)
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum())
    return profit_factor

def get_total_trades(position: pd.Series) -> int:
    """Calculate total number of trades from position."""
    dummy_prices = pd.Series(range(len(position)), index=position.index)  # Dummy prices for counting
    pnl_list = get_trade_pnls(position, dummy_prices)
    result = len(pnl_list)
    return result

def get_rr_ratio_from_pnls(pnl_list: List[float]) -> float:
    """Calculate risk/reward ratio from pre-calculated PnL list."""
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    result = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
    return result

def get_breakeven_rate_from_pnls(pnl_list: List[float]) -> float:
    """Calculate breakeven rate from pre-calculated PnL list."""
    winning_trades = [pnl for pnl in pnl_list if pnl > 0]
    losing_trades = [pnl for pnl in pnl_list if pnl < 0]
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    rr_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0
    result = 1 / (rr_ratio + 1) if rr_ratio > 0 else 0
    return result

def get_information_ratio(strategy_returns: pd.Series, market_returns: pd.Series, return_interval: str, n_days: int) -> float:
    """Calculate annualized information ratio using actual strategy returns that include costs."""
    # Align the series
    common_index = strategy_returns.index.intersection(market_returns.index)
    strategy_aligned = strategy_returns.loc[common_index]
    market_aligned = market_returns.loc[common_index]
    
    active_returns = strategy_aligned - market_aligned
    active_returns = active_returns.dropna()
    
    mean_active_return = active_returns.mean()
    tracking_error = active_returns.std(ddof=1)
    if tracking_error == 0:
        return float('nan')

    # Calculate per-period information ratio
    info_ratio_per_period = mean_active_return / tracking_error

    # Determine periods per year for annualization
    if return_interval == '1m':
        periods_per_year = 365 * 24 * 60  # 525,600 minutes per year
    elif return_interval == '3m':
        periods_per_year = 365 * 24 * 20  # 175,200 3-minute periods per year
    elif return_interval == '5m':
        periods_per_year = 365 * 24 * 12  # 105,120 5-minute periods per year
    elif return_interval == '15m':
        periods_per_year = 365 * 24 * 4   # 35,040 15-minute periods per year
    elif return_interval == '30m':
        periods_per_year = 365 * 24 * 2   # 17,520 30-minute periods per year
    elif return_interval == '1h':
        periods_per_year = 365 * 24       # 8,760 hours per year
    elif return_interval == '4h':
        periods_per_year = 365 * 6        # 2,190 4-hour periods per year
    elif return_interval == '1d':
        periods_per_year = 365             # 365 days per year
    else:
        periods_per_year = 252  # Default to trading days

    # Annualize information ratio: IR_annual = IR_per_period * sqrt(periods_per_year)
    info_ratio_annualized = info_ratio_per_period * np.sqrt(periods_per_year)
    
    return info_ratio_annualized
