import pandas as pd
import numpy as np

class EmpiricalDistribution:
    def __init__(self, F: pd.Series):
        self.F = F
        counts, bin_edges = np.histogram(F, bins=100)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_edges = bin_edges
        self.bin_widths = np.diff(bin_edges)
        self.densities = counts / (self.bin_widths * len(F))
        self.pdf = pd.DataFrame({
            'bin_centers': self.bin_centers,
            'counts': counts,
            'densities': self.densities
        })
    
    def single_pdf(self, x: float) -> float:
        closest_index = np.argmin(np.abs(self.bin_centers - x))
        return self.densities[closest_index]

    def single_cdf(self, x: float) -> float:
        lower_sum = sum(self.pdf['densities'][self.pdf['bin_centers'] < x])
        upper_sum = sum(self.pdf['densities'][self.pdf['bin_centers'] > x])
        return lower_sum / (lower_sum + upper_sum)

    def multiple_cdf(self, X: pd.Series) -> pd.Series:
        return X.apply(self.single_cdf)

    def multiple_pdf(self, X: pd.Series) -> pd.Series:
        return X.apply(self.single_pdf)