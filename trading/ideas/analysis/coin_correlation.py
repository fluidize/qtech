import sys
sys.path.append("trading")

import model_tools as mt

class CoinCorrelationAnalysis:
    def __init__(self, base_symbol, base_chunks, base_interval, base_age_days, base_data_source="binance"):
        self.base_data = mt.fetch_data(symbol=base_symbol,
                                       chunks=base_chunks,
                                       interval=base_interval,
                                       age_days=base_age_days,
                                       data_source=base_data_source)

    def compare_to(self, compare_symbol, compare_chunks, compare_interval, compare_age_days, compare_data_source="binance", column = "HLC3"):
        compare_data = mt.fetch_data(symbol=compare_symbol,
                                     chunks=compare_chunks,
                                     interval=compare_interval,
                                     age_days=compare_age_days,
                                     data_source=compare_data_source)
        
        correlation = self.base_data[column].corr(compare_data[column])

        return correlation

CCA = CoinCorrelationAnalysis(
    base_symbol="ETH-USDT",
    base_chunks=365,
    base_interval="1d",
    base_age_days=0,
    base_data_source="binance"
)

corr = CCA.compare_to(
    compare_symbol="DOGE-USDT",
    compare_chunks=365,
    compare_interval="1d",
    compare_age_days=0,
    compare_data_source="binance",
    column="OHLC4"
)

print(f"Correlation {corr*100:.2f}%")