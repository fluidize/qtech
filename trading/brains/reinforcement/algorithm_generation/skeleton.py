import numpy as np
import pandas as pd

import sys

sys.path.append("trading")
import technical_analysis as ta

class Logic:
    OPERATORS = {
        "GREATER_THAN": lambda x, y: x > y,
        "LESS_THAN": lambda x, y: x < y,
        "GREATER_THAN_OR_EQUAL_TO": lambda x, y: x >= y,
        "LESS_THAN_OR_EQUAL_TO": lambda x, y: x <= y,
        "EQUAL_TO": lambda x, y: x == y,
        "NOT_EQUAL_TO": lambda x, y: x != y,
    }
    def __init__(self, operator: str):
        self.operator = operator
        self.operator_func = self.OPERATORS[operator]
    def __call__(self, x: float, y: float) -> bool:
        return self.operator_func(x, y)
        
class Skeleton:
    INDICATORS = {
        "SMA": ta.sma,
        "RSI": ta.rsi,
        "MACD": ta.macd,
        "STOCH": ta.stoch,
        "ADX": ta.adx,
        "CCI": ta.cci,
        "ROC": ta.roc,
    }
    def __init__(self, indicator: str, params: list[int]):
        self._set_indicator(indicator)
        self._set_params(params)

    def _set_indicator(self, indicator: str):
        self.indicator = indicator
        self.indicator_func = self.INDICATORS[indicator]

    def _set_params(self, indicator_params: list[int]):
        self.indicator_params = indicator_params

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.Series(0, index=data.index)
        long_logic = Logic(Logic.OPERATORS.keys()[self.indicator_params[0]])
        short_logic = Logic(Logic.OPERATORS.keys()[self.indicator_params[1]])
        signals[long_logic(self.indicator_func(data['Close'], timeperiod=self.indicator_params[2]), self.long_threshold)] = 3
        signals[short_logic(self.indicator_func(data['Close'], timeperiod=self.indicator_params[3]), self.short_threshold)] = 1
        return signals