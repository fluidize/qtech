import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity

import sys
sys.path.append('trading')
import model_tools as mt
import technical_analysis as ta
from vectorized_backtesting import VectorizedBacktesting

class EntropyAnalyzer:
    def __init__(self, data: pd.DataFrame
                 , column: str = 'Close', variable: str = 'log_return'):
        self.raw_data = data
        self.column = column
        self.variable = variable
        self.data = self.raw_data[column]

        if self.variable == 'log_return':
            self.data = ta.log_return(self.data)
        elif self.variable == 'percent_change':
            self.data = ta.percent_change(self.data)
        else:
            raise ValueError(f"Invalid variable: {self.variable}")

    def calculate_entropy(self, base: int = 2):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.data)
        log_prob = kde.score_samples(self.data.values)
        entropy = -np.mean(log_prob)
        entropy_bits = entropy / np.log(base) # convert to base
        return entropy_bits
    
    def zscore_inefficiency(self):
        vb = VectorizedBacktesting()
        vb.load_data(self.data)

        

data = mt.fetch_data('BTC-USDT',
                     interval='1min',
                     chunks=1,
                     age_days=1,
                     )

entropy = EntropyAnalyzer(data=data, column='Close', variable='log_return')
print(entropy.calculate_entropy())
