import argparse
import asyncio
import sys
from rich.console import Console
from typing import Dict
import requests

sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")
sys.path.append("solana/jupiter")

from live_system import LiveTradingSystem, SimulatedPortfolio
import strategy

def create_discord_callback(webhook_url: str):
    def discord_callback(signal_info: Dict):
        if signal_info['new_signal'] != signal_info['previous_signal'] and signal_info['new_signal'] != 0:
            if signal_info['new_signal'] == 3:
                message = {
                    "content": f"{signal_info['timestamp']} LONG {signal_info['symbol']} @ {signal_info['current_price']}"
                }
            elif signal_info['new_signal'] == 2:
                message = {
                    "content": f"{signal_info['timestamp']} SHORT {signal_info['symbol']} @ {signal_info['current_price']}"
                }
            elif signal_info['new_signal'] == 1:
                message = {
                    "content": f"{signal_info['timestamp']} FLAT {signal_info['symbol']} @ {signal_info['current_price']}"
                }
            requests.post(webhook_url, json=message)
    return discord_callback

async def main(webhook_url: str = None):
    """Run a single coin portfolio test using Binance signals and Jupiter execution."""
    discord_callback = create_discord_callback(webhook_url)

    params = {
        'supertrend_window': 8,
        'supertrend_multiplier': 5,
        'bb_window': 67,
        'bb_dev': 3,
        'bbw_ma_window': 90
    }

    system = LiveTradingSystem(
        symbol="JTO-USDT",
        interval="1h",
        data_source="binance",
        buffer_size=500,
        strategy_func=strategy.trend_reversal_strategy,
        strategy_params=params,
        signal_callback=discord_callback
    )

    if webhook_url:
        requests.post(webhook_url, json={"content": f"Starting Algorithmic Call Generator | Strategy: {system.strategy_func.__name__} {system.strategy_params}"})

    await system.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run call generator system")
    parser.add_argument("--webhook_url", type=str, default=None, help="Discord webhook URL for notifications")
    args = parser.parse_args()
    
    asyncio.run(main(webhook_url=args.webhook_url))