import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hurst import compute_Hc

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from rich.console import Console
from rich.table import Table

import sys
sys.path.append('trading')
import model_tools as mt
import technical_analysis as ta
from backtesting.backtesting import VectorizedBacktest

class EntropyAnalyzer:
    def __init__(self, data: pd.DataFrame
                 , column: str = 'Close', variable: str = 'log_return'):
        self.data = data
        self.column = column
        self.selected_column = self.data[column]
        self.variable = variable
        
        if self.variable == 'log_return':
            self.processed_column = ta.log_return(self.selected_column).dropna()
        elif self.variable == 'percent_change':
            self.processed_column = self.selected_column.pct_change().dropna()
        else:
            raise ValueError(f"Invalid variable: {self.variable}")

    def calculate_entropy(self, base: int = 2, use_bins: bool = True, bins: int = 50, plot: bool = False):
        variable = self.processed_column.values.reshape(-1, 1)
        scaled_variable  = StandardScaler().fit_transform(variable)

        if use_bins:
            hist, bin_edges = np.histogram(scaled_variable, bins=bins, density=True)
            bin_widths = np.diff(bin_edges)
            probs = hist * bin_widths
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -np.sum(probs * np.log(probs))
            entropy_bits = entropy / np.log(base)
            if plot:
                sns.histplot(scaled_variable, bins=bins, kde=False, stat='density', color='skyblue', label='Histogram')
                plt.title('Histogram of Selected Column')
                plt.xlabel(self.column)
                plt.ylabel('Density')
                plt.legend()
                plt.show()
            return entropy_bits
        else:
            from scipy.spatial import cKDTree
            from scipy.special import digamma, gamma
            n = len(variable)
            k = 3
            tree = cKDTree(variable)
            dists, _ = tree.query(variable, k=k+1, p=2)  # k+1 because the point itself is included
            eps = dists[:, k]
            eps[eps == 0] = np.finfo(float).eps  # Avoid log(0)
            entropy = digamma(n) - digamma(k) + np.log(np.pi**0.5 / gamma(1 + 0.5)) + np.mean(np.log(eps))
            entropy_bits = entropy / np.log(base)
            if plot:
                plt.hist(variable.flatten(), bins=50, color='skyblue', alpha=0.7, label='Data')
                plt.title('Histogram of Selected Column')
                plt.xlabel(self.column)
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()
            return entropy_bits
    
    def zscore_inefficiency(self):
        vb = VectorizedBacktest()
        vb.load_data(self.data)
        vb.run_strategy(vb.zscore_momentum_strategy)
        momentum_strategy_metrics = vb.get_performance_metrics()
        vb.run_strategy(vb.zscore_reversion_strategy)
        reversion_strategy_metrics = vb.get_performance_metrics()

        best_strategy = np.argmax([momentum_strategy_metrics['Information_Ratio'], reversion_strategy_metrics['Information_Ratio']])
        strategy_name = ['momentum', 'reversion'][best_strategy]
        strategy_metrics = [momentum_strategy_metrics, reversion_strategy_metrics][best_strategy]
        return strategy_name, strategy_metrics

    def calculate_hurst_exponent(self):
        if self.variable == 'percent_change':
            H, c, data_reg  = compute_Hc(self.processed_column, kind='change')
            return H
        elif self.variable == 'log_return':
            H, c, data_reg  = compute_Hc(self.processed_column, kind='change')
            return H
        else:
            raise ValueError(f"Invalid variable: {self.variable}")
        
    def informational_profile(self, verbose: bool = False):
        binned_entropy = self.calculate_entropy(use_bins=True, bins=100, plot=False, base=2)
        knn_entropy = self.calculate_entropy(use_bins=False, plot=False, base=2)
        hurst_exponent = self.calculate_hurst_exponent()
        zscore_inefficiency = self.zscore_inefficiency()
        if verbose:
            console = Console()

            info = Table(title="Informational Profile")
            info.add_column("Metric", justify="right")
            info.add_column("Value", justify="left")
            info.add_row("Binned Entropy (bits)", str(binned_entropy))
            info.add_row("KNN Entropy (bits)", str(knn_entropy))
            info.add_row("Hurst Exponent", str(hurst_exponent))
            
            zscore_stats = Table(title="Z-Score Inefficiency")
            zscore_stats.add_column("Metric", justify="right")
            zscore_stats.add_column("Value", justify="left")
            zscore_stats.add_row("Strategy", zscore_inefficiency[0])
            zscore_stats.add_row("Sharpe Ratio", str(zscore_inefficiency[1]['Sharpe_Ratio']))
            zscore_stats.add_row("Information Ratio", str(zscore_inefficiency[1]['Information_Ratio']))
            zscore_stats.add_row("Total Return %", str(zscore_inefficiency[1]['Total_Return']*100))

            console.print(info)
            console.print(zscore_stats)

        output = {
            'binned_entropy': binned_entropy,
            'knn_entropy': knn_entropy,
            'hurst_exponent': hurst_exponent,
            'zscore_strategy': zscore_inefficiency[0],
            'zscore_strategy_metrics': zscore_inefficiency[1]
        }
        return output
        

def scan_and_sort_by_binned_entropy(symbols, interval='5m', chunks=50, age_days=0, column='Close', variable='log_return', bins=100):
    results = []
    for symbol in symbols:
        try:
            data = mt.fetch_data(symbol,
                                 interval=interval,
                                 chunks=chunks,
                                 age_days=age_days)
            analyzer = EntropyAnalyzer(data=data, column=column, variable=variable)
            binned_entropy = analyzer.calculate_entropy(use_bins=True, bins=bins, plot=False, base=2)
            results.append((symbol, binned_entropy))
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    # Sort by binned entropy (ascending)
    results.sort(key=lambda x: x[1])
    return results


symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'GRIFFAIN-USDT', 'RENDER-USDT']
sorted_entropies = scan_and_sort_by_binned_entropy(symbols)
for symbol, entropy in sorted_entropies:
    print(f"{symbol}: {entropy}")
