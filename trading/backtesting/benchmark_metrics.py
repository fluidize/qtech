"""
Benchmark script to evaluate execution times of all performance metrics functions.
Uses real backtest data from VectorizedBacktesting to ensure accurate benchmarking.
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich import print as rprint

import sys
sys.path.append("")
sys.path.append("trading/backtesting")
import vb_metrics as metrics
from backtesting import VectorizedBacktesting
from basic_strategies import signal_spam

def run_backtest_and_extract_data() -> tuple:
    """Run a real backtest and extract the data for benchmarking."""
    print("Generating synthetic backtest data...")
    
    vb = VectorizedBacktesting(
        initial_capital=10000.0,
        slippage_pct=0.001,
        commission_fixed=1.0,
        reinvest=False,
        leverage=1.0
    )
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='1d')
    close_prices = 100 + np.cumsum(np.random.randn(365) * 0.5)
    
    vb.data = pd.DataFrame({
        'Datetime': dates,
        'Open': close_prices + np.random.randn(365) * 0.1,
        'High': close_prices + np.abs(np.random.randn(365) * 0.5),
        'Low': close_prices - np.abs(np.random.randn(365) * 0.5),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 100000, 365)
    })
    vb.interval = '1d'
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
        'initial_capital': vb.initial_capital
    }

def benchmark_function(func_name: str, func, sample_data: dict, n_runs: int = 5) -> Dict:
    """Benchmark a single metrics function."""
    times = []
    error = None
    
    for _ in range(n_runs):
        try:
            if func_name == 'get_sharpe_ratio':
                start = time.perf_counter()
                result = func(
                    sample_data['strategy_returns'],
                    sample_data['interval'],
                    sample_data['n_days']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_sortino_ratio':
                start = time.perf_counter()
                result = func(
                    sample_data['strategy_returns'],
                    sample_data['interval'],
                    sample_data['n_days']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_information_ratio':
                start = time.perf_counter()
                result = func(
                    sample_data['strategy_returns'],
                    sample_data['asset_returns'],
                    sample_data['interval'],
                    sample_data['n_days']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_alpha_beta':
                start = time.perf_counter()
                result = func(
                    sample_data['strategy_returns'],
                    sample_data['asset_returns'],
                    sample_data['n_days'],
                    sample_data['interval']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_trade_pnls':
                start = time.perf_counter()
                result = func(
                    sample_data['position'],
                    sample_data['open_prices']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_win_rate':
                start = time.perf_counter()
                result = func(
                    sample_data['position'],
                    sample_data['open_prices']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_rr_ratio':
                start = time.perf_counter()
                result = func(
                    sample_data['position'],
                    sample_data['open_prices']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_breakeven_rate':
                start = time.perf_counter()
                result = func(
                    sample_data['position'],
                    sample_data['open_prices']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_profit_factor':
                start = time.perf_counter()
                result = func(
                    sample_data['position'],
                    sample_data['open_prices']
                )
                elapsed = time.perf_counter() - start
            elif func_name == 'get_r_and_r2':
                start = time.perf_counter()
                result = func(sample_data['portfolio_value'])
                elapsed = time.perf_counter() - start
            else:
                error = f"Unknown function: {func_name}"
                break
            
            if isinstance(result, tuple):
                _ = [r for r in result]
            elif isinstance(result, list):
                _ = len(result)
            else:
                _ = result
            
            times.append(elapsed)
            
        except Exception as e:
            error = str(e)
            break
    
    if error:
        return {
            'name': func_name,
            'avg_time': float('inf'),
            'min_time': float('inf'),
            'max_time': float('inf'),
            'error': error
        }
    
    return {
        'name': func_name,
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'error': None
    }

def benchmark_all_metrics(n_runs: int = 5) -> List[Dict]:
    """Benchmark all performance metrics functions used in get_performance_metrics."""
    sample_data = run_backtest_and_extract_data()
    
    print(f"Data extracted: {len(sample_data['position'])} rows")
    print(f"Strategy returns: {len(sample_data['strategy_returns'])} rows")
    print(f"Trades: {len(metrics.get_trade_pnls(sample_data['position'], sample_data['open_prices']))}")
    
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
        ('get_r_and_r2', metrics.get_r_and_r2),
    ]
    
    print(f"\nBenchmarking {len(metrics_to_test)} metrics functions ({n_runs} runs each)...")
    results = []
    
    for i, (func_name, func) in enumerate(metrics_to_test, 1):
        print(f"[{i}/{len(metrics_to_test)}] Benchmarking {func_name}...", end='\r')
        
        result = benchmark_function(func_name, func, sample_data, n_runs)
        results.append(result)
    
    print(f"\nCompleted benchmarking {len(metrics_to_test)} functions.")
    return results

def display_results(results: List[Dict]):
    """Display benchmark results in a ranked table."""
    sorted_results = sorted(results, key=lambda x: x['avg_time'])
    
    console = Console()
    table = Table(title="Performance Metrics Execution Time Rankings", show_header=True, header_style="bold magenta")
    
    table.add_column("Rank", style="cyan", width=6, justify="right")
    table.add_column("Function Name", style="green", width=25)
    table.add_column("Avg Time (ms)", style="yellow", width=15, justify="right")
    table.add_column("Min Time (ms)", style="blue", width=15, justify="right")
    table.add_column("Max Time (ms)", style="blue", width=15, justify="right")
    table.add_column("Std Dev (ms)", style="blue", width=15, justify="right")
    table.add_column("Status", style="red", width=25)
    
    for rank, result in enumerate(sorted_results, 1):
        name = result['name']
        avg_ms = result['avg_time'] * 1000
        min_ms = result['min_time'] * 1000
        max_ms = result['max_time'] * 1000
        
        if result['error']:
            status = f"ERROR: {result['error'][:20]}"
            avg_ms_str = "N/A"
            min_ms_str = "N/A"
            max_ms_str = "N/A"
            std_ms_str = "N/A"
        else:
            status = "OK"
            avg_ms_str = f"{avg_ms:.4f}"
            min_ms_str = f"{min_ms:.4f}"
            max_ms_str = f"{max_ms:.4f}"
            std_ms_str = f"{result.get('std_time', 0) * 1000:.4f}"
        
        table.add_row(
            str(rank),
            name,
            avg_ms_str,
            min_ms_str,
            max_ms_str,
            std_ms_str,
            status
        )
    
    console.print(table)
    
    valid_results = [r for r in sorted_results if not r['error']]
    if valid_results:
        times = [r['avg_time'] * 1000 for r in valid_results]
        rprint(f"\n[bold]Summary:[/bold]")
        rprint(f"  Total functions: {len(results)}")
        rprint(f"  Valid functions: {len(valid_results)}")
        rprint(f"  Failed functions: {len(results) - len(valid_results)}")
        rprint(f"  Fastest: {valid_results[0]['name']} ({times[0]:.4f} ms)")
        rprint(f"  Slowest: {valid_results[-1]['name']} ({times[-1]:.4f} ms)")
        rprint(f"  Median: {np.median(times):.4f} ms")
        rprint(f"  Mean: {np.mean(times):.4f} ms")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark performance metrics functions')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per function (default: 5)')
    
    args = parser.parse_args()
    
    results = benchmark_all_metrics(n_runs=args.runs)
    display_results(results)
