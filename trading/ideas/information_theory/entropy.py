import pandas as pd
import model_tools as mt
import numpy as np
import technical_analysis as ta

from sklearn.neighbors import KernelDensity

class EntropyAnalyzer:
    def __init__(self, data: pd.DataFrame['Open', 'High', 'Low', 'Close', 'Volume'], column: str = 'Close', variable: str = 'log_return'):
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
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.data.values)
        log_prob = kde.score_samples(self.data.values)
        entropy = -np.mean(log_prob)
        entropy_bits = entropy / np.log(base) # convert to base
        return entropy_bits

data = mt.fetch_data('BTC-USDT',
                     interval='1h',
                     chunks=1,
                     age_days=1,
                     )

entropy = EntropyAnalyzer(data)
print(entropy.calculate_entropy())
