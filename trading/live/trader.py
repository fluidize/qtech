import argparse
import asyncio
import sys
from typing import Dict
import numpy as np

sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")
sys.path.append("solana/jupiter")

from live_system import LiveTradingSystem
import strategy
import jupiter as jup

def create_trader_callback(wallethandler: jup.JupiterWalletHandler, webhook_url: str = None):
    """
    Create a trading callback for live execution using Jupiter.
    Uses Jupiter for SOL-USDC pricing and execution while getting signals from Binance.
    
    Args:
        wallethandler: JupiterWalletHandler for getting real prices and slippage
        webhook_url: Discord webhook URL for notifications (optional)
    
    Returns:
        Trading callback function
    """
    
    def trader_callback(signal_info: Dict):
        new_signal = signal_info['new_signal']
        
        if new_signal != 0:
            estimated_trade_size_sol = 1.0
            estimated_trade_size_usd = 100.0
            
            if new_signal == 1:
                wallethandler.get_order(jup.Token.SOL, jup.Token.USDC, estimated_trade_size_sol, retry=True)
            elif new_signal == 3:
                wallethandler.get_order(jup.Token.USDC, jup.Token.SOL, estimated_trade_size_usd, retry=True)
            else:
                wallethandler.get_order(jup.Token.SOL, jup.Token.USDC, estimated_trade_size_sol, retry=True)

    return trader_callback

async def run_live_trader(webhook_url: str = None, private_key: str = None):
    """Run live trading system using Binance signals and Jupiter execution."""
    
    if not private_key:
        return
    
    wallethandler = jup.JupiterWalletHandler(private_key)
    
    callback = create_trader_callback(wallethandler, webhook_url)

    optim_set = {'symbol': 'SOL-USDT', 'interval': '15m', 'metric': np.float64(10.340510002150115), 'params': {'supertrend_window': 8, 'supertrend_multiplier': 1.224743711629419, 'bb_window': 55, 'bb_dev': 2, 'bbw_ma_window': 10}}
        
    system = LiveTradingSystem(
        symbol=optim_set['symbol'],
        interval=optim_set['interval'],
        data_source="binance",
        buffer_size=500,
        strategy_func=strategy.trend_reversal_strategy,
        strategy_params=optim_set['params'],
        signal_callback=callback
    )
    
    try:
        await system.start()
    except KeyboardInterrupt:
        system.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live SOL-USDT Trader using Jupiter")
    parser.add_argument("--webhook_url", type=str, default=None, help="Discord webhook URL for notifications")
    parser.add_argument("--private_key", type=str, required=True, help="Solana private key for Jupiter wallet")
    args = parser.parse_args()
    
    asyncio.run(run_live_trader(webhook_url=args.webhook_url, private_key=args.private_key))
