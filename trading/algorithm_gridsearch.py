from spot_trade_env import BacktestEnvironment
from spot_trade_env import TradingEnvironment

def Custom_Scalper_GridSearch(self, env: TradingEnvironment, context, current_ohlcv):
    """Basic strategy that buys when current close is higher that prev close. STD is used to stop out in volatile markets. Performs well in 1m."""
    current_close = current_ohlcv['Close']

    std = self._calculate_std(context, 10)
    current_std = std.iloc[-1]
    prev_close = context['Close'].iloc[-2]
    
    # Calculate price change percentage
    price_change_pct = ((current_close - prev_close) / prev_close) * 100

    std_threshold = 0.025
    pct_threshold = 0.05
    
    buy_conditions = [price_change_pct > pct_threshold, current_std > std_threshold]
    sell_conditions = [price_change_pct < -pct_threshold]

    if all(buy_conditions):
        env.portfolio.buy_max(self.current_symbol, current_close, env.get_current_timestamp())
    elif all(sell_conditions):
        env.portfolio.sell_max(self.current_symbol, current_close, env.get_current_timestamp())

ma_grid = [10, 20, 30, 40, 50]
for ma in ma_grid:
    env = BacktestEnvironment(instance_name=f"Custom_Scalper_{ma}", symbols=['SOL-USD'], initial_capital=1000, chunks=1, interval='1m', age_days=0)
    env.run([Custom_Scalper_GridSearch])
