import argparse
import asyncio
import sys
from rich.console import Console
from typing import Dict
import requests
from discord_webhook import DiscordWebhook, DiscordEmbed

sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")
sys.path.append("solana/jupiter")

from live_system import LiveTradingSystem, SimulatedPortfolio
import strategy

def create_discord_callback(webhook_url: str):
    def discord_callback(signal_info: Dict):
        # Only send Discord notifications on actual signal changes, not on every candle close
        if signal_info['new_signal'] != signal_info['previous_signal'] and signal_info['new_signal'] != 0:
            webhook = DiscordWebhook(url=webhook_url)
            if signal_info['new_signal'] == 3:
                color = "39FF14"
            elif signal_info['new_signal'] == 2:
                color = "FFFF33"
            elif signal_info['new_signal'] == 1:
                color = "FF073A"

            embed = DiscordEmbed(title=f"{signal_info['symbol']}", color=color)
            embed.add_embed_field(name="Action", value=f"***{signal_info['action']}***", inline=True)
            embed.add_embed_field(name="Price", value=f"{signal_info['current_price']}", inline=True)
            embed.add_embed_field(name="Timestamp", value=f"{signal_info['timestamp']}", inline=True)
            webhook.add_embed(embed)
            webhook.execute()
    return discord_callback

async def main(webhook_url: str = None):
    """Run a single coin portfolio test using Binance signals and Jupiter execution."""
    discord_callback = create_discord_callback(webhook_url)
    params = {'supertrend_window': 2, 'supertrend_multiplier': 7, 'bb_window': 47, 'bb_dev': 8, 'bbw_ma_window': 49}

    system = LiveTradingSystem(
        symbol="SOL-USDT",
        interval="1h",
        data_source="binance",
        buffer_size=500,
        strategy_func=strategy.trend_reversal_strategy,
        strategy_params=params,
        signal_callback=discord_callback,
        always_call_callback=True
    )

    if webhook_url:
        webhook = DiscordWebhook(url=webhook_url)
        embed = DiscordEmbed(title="Starting Algorithmic Call Generator", color="1BFFFF")
        embed.add_embed_field(name="Strategy", value=f"`{system.strategy_func.__name__}`", inline=True)
        embed.add_embed_field(name="Symbol", value=f"`{system.symbol}`", inline=True)
        embed.add_embed_field(name="Interval", value=f"`{system.interval}`", inline=True)
        embed.add_embed_field(name="Data Source", value=f"`{system.data_source}`", inline=True)
        embed.add_embed_field(name="Parameters", value=f"`{system.strategy_params}`", inline=True)
        webhook.add_embed(embed)
        webhook.execute()

    await system.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run call generator system")
    parser.add_argument("--webhook_url", type=str, default=None, help="Discord webhook URL for notifications")
    args = parser.parse_args()
    
    asyncio.run(main(webhook_url=args.webhook_url))