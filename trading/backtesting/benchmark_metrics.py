"""
Benchmark script to evaluate execution times of all performance metrics functions.
Tests scaling behavior across different data sizes to analyze Big O complexity.
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import matplotlib.pyplot as plt

import sys
sys.path.append("")
sys.path.append("trading/backtesting")
import vb_metrics as metrics
from backtesting import VectorizedBacktesting
from basic_strategies import signal_spam

def generate_backtest_data(n_days: int, interval: str = "1h") -> dict:
    """Generate backtest data for a given number of days."""
    vb = VectorizedBacktesting(
        initial_capital=10000.0,
        slippage_pct=0.001,
        commission_fixed=1.0,
        leverage=1.0
    )
    
    try:
        vb.fetch_data(
            symbol="SOL-USDT",
            days=n_days,
            interval=interval,
            age_days=0,
            data_source="binance",
            cache_expiry_hours=720,
            verbose=False
        )
    except:
        # Fallback to synthetic data if fetch fails
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq=interval)
        close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        vb.data = pd.DataFrame({
            'Datetime': dates,
            'Open': close_prices + np.random.randn(n_days) * 0.1,
            'High': close_prices + np.abs(np.random.randn(n_days) * 0.5),
            'Low': close_prices - np.abs(np.random.randn(n_days) * 0.5),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 100000, n_days)
        })
        vb.interval = interval
        vb._set_n_days()
    
    vb.run_strategy(signal_spam, verbose=False)
    
    position = vb.data['Position']
    open_prices = vb.data['Open']
    strategy_returns = vb.data['Strategy_Returns'].dropna()
    portfolio_value = vb.data['Portfolio_Value'].dropna()
    asset_returns = vb.data['Open_Return'].dropna()
    
    return {
        'position': position,
        'open_prices': open_prices,
        'strategy_returns': strategy_returns,
        'asset_returns': asset_returns,
        'portfolio_value': portfolio_value,
        'n_days': vb.n_days,
        'interval': vb.interval,
        'initial_capital': vb.initial_capital,
        'data_size': len(position)
    }

def benchmark_function_at_size(func_name: str, func, sample_data: dict, n_runs: int = 5) -> float:
    """Benchmark a single metrics function and return average time."""
    times = []
    
    for _ in range(n_runs):
        try:
            if func_name == 'get_sharpe_ratio':
                start = time.perf_counter()
                result = func(sample_data['strategy_returns'], sample_data['interval'], sample_data['n_days'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_sortino_ratio':
                start = time.perf_counter()
                result = func(sample_data['strategy_returns'], sample_data['interval'], sample_data['n_days'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_information_ratio':
                start = time.perf_counter()
                result = func(sample_data['strategy_returns'], sample_data['asset_returns'], sample_data['interval'], sample_data['n_days'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_alpha_beta':
                start = time.perf_counter()
                result = func(sample_data['strategy_returns'], sample_data['asset_returns'], sample_data['n_days'], sample_data['interval'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_trade_pnls':
                start = time.perf_counter()
                result = func(sample_data['position'], sample_data['open_prices'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_win_rate':
                start = time.perf_counter()
                result = func(sample_data['position'], sample_data['open_prices'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_rr_ratio':
                start = time.perf_counter()
                result = func(sample_data['position'], sample_data['open_prices'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_breakeven_rate':
                start = time.perf_counter()
                result = func(sample_data['position'], sample_data['open_prices'])
                elapsed = time.perf_counter() - start
            elif func_name == 'get_profit_factor':
                start = time.perf_counter()
                result = func(sample_data['position'], sample_data['open_prices'])
                elapsed = time.perf_counter() - start
            else:
                return float('inf')
            
            if isinstance(result, tuple):
                _ = [r for r in result]
            elif isinstance(result, list):
                _ = len(result)
            else:
                _ = result
            
            times.append(elapsed)
        except Exception:
            return float('inf')
    
    return np.mean(times) * 1000  # Return in milliseconds

def benchmark_scaling(func_name: str, func, data_sizes: List[int], interval: str = "1h", n_runs: int = 5) -> Tuple[List[int], List[float]]:
    """Benchmark a function across multiple data sizes."""
    sizes = []
    times = []
    
    for size in data_sizes:
        print(f"  Testing {func_name} with {size} days...", end='\r')
        sample_data = generate_backtest_data(size, interval)
        avg_time = benchmark_function_at_size(func_name, func, sample_data, n_runs)
        
        if avg_time != float('inf'):
            sizes.append(sample_data['data_size'])
            times.append(avg_time)
    
    print(f"  {func_name}: {len(sizes)} sizes tested")
    return sizes, times

def benchmark_all_metrics(data_sizes: List[int] = [30, 60, 90, 180, 365], interval: str = "1h", n_runs: int = 5) -> Dict[str, Tuple[List[int], List[float]]]:
    """Benchmark all metrics functions across multiple data sizes."""
    print(f"Benchmarking metrics across data sizes: {data_sizes} days")
    print(f"Interval: {interval}, Runs per test: {n_runs}\n")
    
    metrics_to_test = [
        ('get_sharpe_ratio', metrics.get_sharpe_ratio),
        ('get_sortino_ratio', metrics.get_sortino_ratio),
        ('get_information_ratio', metrics.get_information_ratio),
        ('get_alpha_beta', metrics.get_alpha_beta),
        ('get_trade_pnls', metrics.get_trade_pnls),
        ('get_win_rate', metrics.get_win_rate),
        ('get_rr_ratio', metrics.get_rr_ratio),
        ('get_breakeven_rate', metrics.get_breakeven_rate),
        ('get_profit_factor', metrics.get_profit_factor),
    ]
    
    results = {}
    
    for i, (func_name, func) in enumerate(metrics_to_test, 1):
        print(f"[{i}/{len(metrics_to_test)}] Benchmarking {func_name}...")
        sizes, times = benchmark_scaling(func_name, func, data_sizes, interval, n_runs)
        results[func_name] = (sizes, times)
    
    return results

def analyze_complexity(sizes: List[int], times: List[float]) -> str:
    """Analyze Big O complexity from scaling data."""
    if len(sizes) < 2:
        return "Unknown"
    
    sizes_arr = np.array(sizes)
    times_arr = np.array(times)
    
    # Remove zeros and invalid values
    valid = (sizes_arr > 0) & (times_arr > 0)
    if valid.sum() < 2:
        return "Unknown"
    
    sizes_arr = sizes_arr[valid]
    times_arr = times_arr[valid]
    
    # Log-log regression to estimate exponent
    log_sizes = np.log(sizes_arr)
    log_times = np.log(times_arr)
    
    # Linear regression: log(time) = a * log(size) + b
    # Exponent = a
    if len(log_sizes) > 1 and log_sizes.std() > 0:
        exponent = np.polyfit(log_sizes, log_times, 1)[0]
        
        if exponent < 0.3:
            return "O(1)"
        elif exponent < 0.7:
            return "O(log n)"
        elif exponent < 1.3:
            return "O(n)"
        elif exponent < 1.7:
            return "O(n log n)"
        elif exponent < 2.3:
            return "O(n²)"
        else:
            return f"O(n^{exponent:.1f})"
    
    return "Unknown"

def display_results(results: Dict[str, Tuple[List[int], List[float]]]):
    """Display benchmark results and create scaling plots."""
    console = Console()
    
    # Create summary table with complexity analysis
    table = Table(title="Performance Metrics - Scaling Analysis", show_header=True, header_style="bold magenta")
    
    table.add_column("Function Name", style="green", width=25)
    table.add_column("Avg Time @ 365d (ms)", style="yellow", width=20, justify="right")
    table.add_column("Complexity", style="cyan", width=15)
    table.add_column("Sizes Tested", style="blue", width=15, justify="right")
    
    # Find time at largest size for ranking
    function_times = []
    for func_name, (sizes, times) in results.items():
        if sizes and times:
            complexity = analyze_complexity(sizes, times)
            # Get time at largest size
            max_idx = np.argmax(sizes)
            max_time = times[max_idx]
            function_times.append((max_time, func_name, complexity, len(sizes)))
    
    # Sort by time at largest size
    function_times.sort()
    
    for max_time, func_name, complexity, num_sizes in function_times:
        sizes, times = results[func_name]
        table.add_row(
            func_name,
            f"{max_time:.4f}",
            complexity,
            str(num_sizes)
        )
    
    console.print(table)
    
    # Create scaling plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Metrics Function Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot all functions on linear scale
    for func_name, (sizes, times) in results.items():
        if sizes and times:
            ax.plot(sizes, times, marker='o', label=func_name, linewidth=2, markersize=6)
    ax.set_xlabel('Data Size (number of periods)', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Execution Time vs Data Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    valid_results = [(name, sizes, times) for name, (sizes, times) in results.items() if sizes and times]
    if valid_results:
        all_times = [t for _, _, times in valid_results for t in times]
        rprint(f"\n[bold]Summary:[/bold]")
        rprint(f"  Functions tested: {len(valid_results)}")
        rprint(f"  Data sizes tested: {sorted(set(s for _, sizes, _ in valid_results for s in sizes))}")
        rprint(f"  Average time range: {min(all_times):.4f} - {max(all_times):.4f} ms")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark performance metrics functions with scaling analysis')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per function (default: 5)')
    parser.add_argument('--sizes', type=str, default='15,30,45,60,75,90,135,180,270,365, 730, 1095', help='Comma-separated list of days to test (default: 15,30,45,60,75,90,135,180,270,365, 730, 1095)')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval (default: 1h)')
    
    args = parser.parse_args()
    
    data_sizes = [int(x.strip()) for x in args.sizes.split(',')]
    
    results = benchmark_all_metrics(data_sizes=data_sizes, interval=args.interval, n_runs=args.runs)
    display_results(results)
