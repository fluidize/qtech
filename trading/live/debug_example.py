#!/usr/bin/env python3
"""
Example usage of LiveTradingSystem with debug plotting enabled.
This will show OHLC candles with signal markers similar to backtesting.py
"""

import asyncio
import sys
import os

# Add paths for imports
sys.path.append("")

from live_system import LiveTradingSystem
import trading.backtesting.cstrats as cs

async def debug_live_trading():
    """Example of running live trading system with debug plotting"""
    
    # Create live trading system with debug plotting enabled
    live_system = LiveTradingSystem(
        symbol="BTC-USDT",
        interval="1m",
        data_source="binance",
        buffer_size=500,
        strategy_func=cs.trend_reversal_strategy_v1,
        strategy_params={
            'supertrend_window': 37,
            'supertrend_multiplier': 2,
            'ma_window': 35,
        },
        debug_plot=True,
        debug_plot_window=500,
        signal_callback=debug_signal_callback,
        always_call_callback=True
    )
    
    print("Starting live trading system with debug plotting...")
    print("- Debug plots will open in browser on each candle close")
    print("- Data quality information will be printed to console")
    print("- Press Ctrl+C to stop")
    
    try:
        await live_system.start()
    except KeyboardInterrupt:
        print("\nStopping live trading system...")
        live_system.stop()

def debug_signal_callback(signal_info):
    """Callback function to handle signal changes"""
    print(f"🚨 SIGNAL: {signal_info['new_signal']} @ ${signal_info['current_price']:.4f}")
    print(f"   Previous: {signal_info['previous_signal']} -> New: {signal_info['new_signal']}")
    print(f"   Time: {signal_info['timestamp'].strftime('%H:%M:%S')}")
    print("-" * 50)

if __name__ == "__main__":
    # Run the debug example
    asyncio.run(debug_live_trading())
