import pandas as pd
import numpy as np

class EmpiricalDistribution:
    def __init__(self, F: pd.Series):
        self.F = F
        counts, bin_edges = np.histogram(F, bins=100)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_edges = bin_edges
        self.bin_widths = np.diff(bin_edges)
        self.probabilities = counts / (sum(counts))
        self.pdf = pd.DataFrame({
            'bin_centers': self.bin_centers,
            'frequency': counts,
            'probability': self.probabilities
        })
    
    def single_pdf(self, x: float) -> float:
        closest_index = np.argmin(np.abs(self.bin_centers - x))
        return self.probabilities[closest_index]

    def single_cdf(self, x: float) -> float:
        return sum(self.pdf['probability'][self.pdf['bin_centers'] < x])

    def multiple_cdf(self, X: pd.Series) -> pd.Series:
        return X.apply(self.single_cdf)

    def multiple_pdf(self, X: pd.Series) -> pd.Series:
        return X.apply(self.single_pdf)