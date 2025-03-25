import spot_trade_env as spot
from spot_trade_env import BacktestEnvironment
from spot_trade_env import TradingEnvironment

ma_grid = [10, 20, 30, 40, 50]
for ma in ma_grid:

    def Custom_Scalper_GridSearch(env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']

        std = spot._calculate_std(context, ma)
        current_std = std.iloc[-1]
        prev_close = context['Close'].iloc[-2]
        
        # Calculate price change percentage
        price_change_pct = ((current_close - prev_close) / prev_close) * 100

        std_threshold = 0.025
        pct_threshold = 0.05
        
        buy_conditions = [price_change_pct > pct_threshold, current_std > std_threshold]
        sell_conditions = [price_change_pct < -pct_threshold]

        if all(buy_conditions):
            env.portfolio.buy_max(env.get_current_symbol(), current_close, env.get_current_timestamp())
        elif all(sell_conditions):
            env.portfolio.sell_max(env.get_current_symbol(), current_close, env.get_current_timestamp())
    
    env = BacktestEnvironment()
    summary = env.run([Custom_Scalper_GridSearch])
    print(summary)
    print(f"MA: {ma}, Total Return: {summary['Custom_Scalper_GridSearch']['PnL']:.2f}%")
