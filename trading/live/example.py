"""
Example usage of the modular live trading system.
"""

import asyncio
import sys

sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")

from backtesting import Strategy

from live_system import LiveTradingSystem

def my_signal_callback(signal_info):
    """Custom callback function for handling trading signals."""
    print(f"🚨 SIGNAL: {signal_info['action']} {signal_info['symbol']} @ ${signal_info['current_price']:.4f}")

async def run_binance_example():
    """Example: Run live system with Binance."""
    system = LiveTradingSystem(
        symbol="BTCUSDT",
        data_source="binance",
        buffer_size=200,
        strategy_func=Strategy.ema_cross_strategy,
        strategy_params={'fast_period': 9, 'slow_period': 21},
        signal_callback=my_signal_callback
    )
    
    await system.start()

if __name__ == "__main__":
    asyncio.run(run_binance_example()) 