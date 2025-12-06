import pandas as pd
import numpy as np

import sys
sys.path.append("")

import trading.backtesting.backtesting as bt
import trading.technical_analysis as ta

class Core:
    def __init__(self):
        self.individual = None
        self.genes = [Indicators(), Logic()]

    def _chromosome_builder(self, genes: list[callable]) -> str:
        chromosome = ""
        for func in genes:
            chromosome += func() + "\n"

        return chromosome

    def construct_individual(self) -> callable:

        def individual(data: pd.DataFrame) -> pd.DataFrame:
            signals = pd.Series(0, index=data.index)
            eval(self._chromosome_builder(self.genes))
            return signals
        
        self.individual = individual

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.individual(data)

# Each 'gene' will return a string to be evaluated on call.
# 'Alleles' are to be optimized per individual

class Indicators:
    def __init__(self):
        pass

    def __call__(self) -> pd.DataFrame:
        pass

class Logic:
    def __init__(self):
        pass

    def __call__(self) -> pd.DataFrame:
        pass

