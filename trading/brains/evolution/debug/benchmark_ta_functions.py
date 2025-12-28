"""
Benchmark script to evaluate execution times of all available TA functions for GP.
Ranks functions from fastest to slowest.
"""
import time
import pandas as pd
import numpy as np
import inspect
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich import print as rprint

import sys
sys.path.append("")
import trading.technical_analysis as ta
from param_space import registered_param_specs, FunctionSpec
from gp_tools import get_indicators

def generate_sample_data(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n, freq='1h')
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    data = pd.DataFrame({
        'Datetime': dates,
        'Open': close_prices + np.random.randn(n) * 0.1,
        'High': close_prices + np.abs(np.random.randn(n) * 0.5),
        'Low': close_prices - np.abs(np.random.randn(n) * 0.5),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 100000, n)
    })
    
    # Ensure High >= Low and High/Low contain Open/Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1) + np.abs(np.random.randn(n) * 0.1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1) - np.abs(np.random.randn(n) * 0.1)
    
    return data

def get_default_params(func_name: str, func_spec: FunctionSpec) -> Dict:
    """Get default parameter values from search space."""
    params = {}
    for param_spec in func_spec.parameters:
        param_name = param_spec.parameter_name
        search_space = param_spec.search_space
        
        if isinstance(search_space, tuple) and len(search_space) == 2:
            if isinstance(search_space[0], int):
                params[param_name] = (search_space[0] + search_space[1]) // 2
            else:
                params[param_name] = (search_space[0] + search_space[1]) / 2.0
        elif isinstance(search_space, list):
            params[param_name] = search_space[len(search_space) // 2]
        else:
            params[param_name] = 20  # fallback
    
    return params

def get_function_args(func, func_name: str, data: pd.DataFrame) -> Tuple[List, Dict]:
    """Determine which data columns to pass to the function based on its signature."""
    sig = inspect.signature(func)
    args = []
    kwargs = {}
    
    data_keywords = {
        "series": "Close",
        "high": "High",
        "low": "Low",
        "open_price": "Open",
        "open": "Open",
        "close": "Close",
        "volume": "Volume"
    }
    
    for param_name, param in sig.parameters.items():
        if param_name in data_keywords:
            col = data_keywords[param_name]
            args.append(data[col])
        elif param_name == "data":
            args.append(data)
        else:
            # This is a parameter, will be filled by get_default_params
            pass
    
    return args, kwargs

def benchmark_function(func, func_name: str, data: pd.DataFrame, n_runs: int = 5) -> Dict:
    """Benchmark a single TA function."""
    if func_name not in registered_param_specs:
        return {
            'name': func_name,
            'avg_time': float('inf'),
            'min_time': float('inf'),
            'max_time': float('inf'),
            'error': 'Not in registered_param_specs'
        }
    
    func_spec = registered_param_specs[func_name]
    default_params = get_default_params(func_name, func_spec)
    
    times = []
    error = None
    
    for _ in range(n_runs):
        try:
            args, kwargs = get_function_args(func, func_name, data)
            kwargs.update(default_params)
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            # Ensure result is computed (not lazy)
            if isinstance(result, tuple):
                _ = [r.values for r in result]
            elif hasattr(result, 'values'):
                _ = result.values
            
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

def benchmark_all_functions(data_size: int = 1000, n_runs: int = 5) -> List[Dict]:
    """Benchmark all available TA functions."""
    print(f"Generating sample data ({data_size} rows)...")
    data = generate_sample_data(data_size)
    
    print(f"Getting available indicators...")
    indicators = get_indicators()
    
    print(f"Found {len(indicators)} indicators. Benchmarking...")
    results = []
    
    for i, func in enumerate(indicators, 1):
        func_name = func.__name__
        print(f"[{i}/{len(indicators)}] Benchmarking {func_name}...", end='\r')
        
        result = benchmark_function(func, func_name, data, n_runs)
        results.append(result)
    
    print(f"\nCompleted benchmarking {len(indicators)} functions.")
    return results

def display_results(results: List[Dict]):
    """Display benchmark results in a ranked table."""
    # Sort by average time (fastest first)
    sorted_results = sorted(results, key=lambda x: x['avg_time'])
    
    console = Console()
    table = Table(title="TA Function Execution Time Rankings", show_header=True, header_style="bold magenta")
    
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
    
    # Summary statistics
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
    
    parser = argparse.ArgumentParser(description='Benchmark TA functions for GP')
    parser.add_argument('--data-size', type=int, default=1000, help='Number of data points (default: 1000)')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per function (default: 5)')
    
    args = parser.parse_args()
    
    results = benchmark_all_functions(data_size=args.data_size, n_runs=args.runs)
    display_results(results)

