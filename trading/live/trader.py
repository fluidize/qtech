import argparse
import asyncio
import sys
from typing import Dict
import numpy as np

sys.path.append("")

from live_system import LiveTradingSystem
import trading.backtesting.testing.cstrats as strategy
import solana.jupiter.jupiter_api as jup

def create_trader_callback(wallethandler: jup.JupiterWalletHandler, starting_usdc_size: float = 100.0, webhook_url: str = None):
    """
    Create a trading callback for live execution using Jupiter.
    Uses Jupiter for SOL-USDC pricing and execution while getting signals from Binance.
    
    Args:
        wallethandler: JupiterWalletHandler for getting real prices and slippage
        starting_usdc_size: Amount of USDC to trade
        webhook_url: Discord webhook URL for notifications (optional)
    
    Returns:
        Trading callback function
    """
    
    # Track the exact amount of SOL bought
    usdc_amount_to_buy = 0.0
    sol_amount_to_sell = 0.0
    
    def trader_callback(signal_info: Dict):
        nonlocal sol_amount_to_sell, usdc_amount_to_buy
        new_signal = signal_info['new_signal']
        last_signal = signal_info['previous_signal']
        
        if (new_signal != 0) & (new_signal != last_signal):
            
            if new_signal == 2:  # Buy SOL with USDC
                response_info, signature = wallethandler.order_and_execute(jup.Token.USDC, jup.Token.SOL, starting_usdc_size, retry=True)
                if response_info and signature:
                    sol_amount_to_sell = response_info['out_amount_decimals']
                    
            elif new_signal == 3:  # Sell SOL for USDC
                if sol_amount_to_sell > 0:
                    response_info, signature = wallethandler.order_and_execute(jup.Token.SOL, jup.Token.USDC, sol_amount_to_sell, retry=True)
                    if response_info and signature:
                        usdc_amount_to_buy = response_info['out_amount_decimals']
                else:
                    print("No SOL to sell")

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
        signal_callback=callback,
        always_call_callback=True
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
