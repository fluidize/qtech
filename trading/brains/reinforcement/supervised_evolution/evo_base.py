import pandas as pd
import numpy as np
import inspect

import sys
sys.path.append("")

import trading.backtesting.backtesting as bt
import trading.technical_analysis as ta

class Core:
    def __init__(self, genes: list[callable]):
        self.individual = None #callable
        self.genes = genes #[indicator, logic]

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

class Indicator:
    def __init__(self, indicator: callable):
        self.indicator = indicator
        self.indicator_params = inspect.signature(indicator).parameters

    def __call__(self) -> pd.DataFrame:
        print(self.indicator_params)

class Logic:
    def __init__(self):
        pass

    def __call__(self) -> pd.DataFrame:
        pass


ind = Indicator(ta.sma)
ind()