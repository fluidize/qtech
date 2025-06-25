import argparse
import asyncio
import sys
from rich.console import Console
from typing import Dict
import requests

# Add the trading directory to the path
sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")
sys.path.append("solana/jupiter")

from live_system import LiveTradingSystem, SimulatedPortfolio
import strategy
import jupiter as jup

def create_enhanced_portfolio_callback(portfolio: SimulatedPortfolio, wallethandler: jup.JupiterWalletHandler, console=None, log_interval: int = 5, webhook_url: str = None):
    """
    Create an enhanced callback with more detailed logging and statistics.
    Uses Jupiter for SOL-USDC pricing and slippage while getting signals from Binance.
    
    Args:
        portfolio: SimulatedPortfolio instance
        wallethandler: JupiterWalletHandler for getting real prices and slippage
        console: Rich console for output (optional)
        log_interval: How often to log detailed statistics (every N trades)
        webhook_url: Discord webhook URL for notifications
    
    Returns:
        Enhanced callback function
    """
    trade_count = 0
    
    def enhanced_portfolio_callback(signal_info: Dict):
        nonlocal trade_count
        
        timestamp = signal_info['timestamp']
        binance_price = float(signal_info['current_price'])  # This is the Binance SOL-USDT price for signals
        new_signal = signal_info['new_signal']
        
        # Get real Jupiter SOL-USDC price for execution
        jupiter_price = jup.get_usd_price(jup.Token.SOL)
        if jupiter_price is None:
            console.print(f"[red]Warning: Could not get Jupiter price, using Binance price ${binance_price:.2f}[/red]")
            execution_price = binance_price
            jupiter_slippage_bps = 50.0  # Default 0.5% slippage
        else:
            execution_price = jupiter_price
            
            # Calculate trade size based on portfolio value to estimate slippage
            portfolio_value = portfolio.get_portfolio_value(execution_price)
            estimated_trade_size_sol = portfolio_value / execution_price if execution_price > 0 else 1.0
            estimated_trade_size_usd = portfolio_value
        
        # print(estimated_trade_size_sol)
        # debug_order = wallethandler.get_order(jup.Token.SOL, jup.Token.USDC, estimated_trade_size_sol)
        # print(debug_order)

        if new_signal != 0:  # Not HOLD - execute trade
            if new_signal == 1:  # SHORT
                order_result = wallethandler.get_order(jup.Token.SOL, jup.Token.USDC, estimated_trade_size_sol, retry=True)
            elif new_signal == 3:  # LONG
                order_result = wallethandler.get_order(jup.Token.USDC, jup.Token.SOL, estimated_trade_size_usd, retry=True)
            else:  # FLAT (2) - close position
                order_result = wallethandler.get_order(jup.Token.SOL, jup.Token.USDC, estimated_trade_size_sol, retry=True)
            
            if order_result is None:
                if console:
                    console.print(f"[red]Warning: Could not get Jupiter order, using default slippage[/red]")
                jupiter_slippage_bps = 50.0  # Default 0.5% slippage
            else:
                in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx = order_result
                jupiter_slippage_bps = price_impact_pct * 100.0 

            trade_result = portfolio.execute_trade(new_signal, execution_price, timestamp, slippage_bps=jupiter_slippage_bps)
            
            if jupiter_price:
                price_diff = jupiter_price - binance_price
                price_diff_pct = (price_diff / binance_price) * 100
                console.print(f"[yellow]Price Difference: Binance ${binance_price:.2f} vs Jupiter ${jupiter_price:.2f} ({price_diff_pct:+.2f}%)[/yellow]")
        else:  # HOLD (0) - no trade
            jupiter_slippage_bps = 50.0  # Default slippage for HOLD
            trade_result = portfolio.execute_trade(new_signal, execution_price, timestamp, slippage_bps=jupiter_slippage_bps)
        
        if trade_result['action'] != 'HOLD':
            trade_count += 1
            action = trade_result['action']
            position_change = trade_result['position_change']
            slippage_cost = trade_result['slippage_cost']
            post_value = trade_result['post_trade_value']
            execution_price_used = trade_result['execution_price']
            slippage_bps_used = trade_result['slippage_bps_used']
            
            trade_color = 'green' if 'BUY' in action else 'red' if 'SELL' in action else 'yellow'
            console.print(f"[{trade_color}]{timestamp.strftime('%H:%M:%S')} | TRADE #{trade_count} | {action}[/{trade_color}]")
            console.print(f"[{trade_color}]  Binance Signal: ${binance_price:.2f} | Jupiter Execution: ${execution_price_used:.2f} | Position: {position_change:+.4f}[/{trade_color}]")
            console.print(f"[{trade_color}]  Jupiter Slippage: {slippage_bps_used:.1f} bps | Slippage Cost: ${slippage_cost:.2f}[/{trade_color}]")
            
            pre_value = trade_result['pre_trade_value']
            value_change = post_value - pre_value
            value_color = 'green' if value_change >= 0 else 'red'
            console.print(f"[{value_color}]  Portfolio: ${pre_value:.2f} → ${post_value:.2f} (${value_change:+.2f})[/{value_color}]")
            
            if 'realized_pnl' in trade_result:
                realized_pnl = trade_result['realized_pnl']
                pnl_color = 'green' if realized_pnl >= 0 else 'red'
                console.print(f"[{pnl_color}]  Realized P&L: ${realized_pnl:+.2f}[/{pnl_color}]")
        
        if trade_count > 0 and trade_count % log_interval == 0:
            stats = portfolio.get_statistics()
            console.print(f"[cyan]{'='*60}[/cyan]")
            console.print(f"[cyan]PORTFOLIO STATISTICS (Trade #{trade_count})[/cyan]")
            console.print(f"[cyan]{'='*60}[/cyan]")
            console.print(f"Initial Balance: ${stats.get('initial_balance', 0):.2f}")
            console.print(f"Current Value: ${stats.get('current_value', 0):.2f}")
            console.print(f"Total Return: {stats.get('total_return_pct', 0):.2f}%")
            console.print(f"Total Trades: {stats.get('total_trades', 0)}")
            console.print(f"Win Rate: {stats.get('win_rate_pct', 0):.1f}%")
            console.print(f"Total P&L: ${stats.get('total_pnl', 0):.2f}")
            console.print(f"Total Slippage Cost: ${stats.get('total_slippage_cost', 0):.2f}")
            console.print(f"Current Position: {stats.get('current_position', 0):.4f}")
            console.print(f"Available Cash: ${stats.get('current_cash', 0):.2f}")
            console.print(f"[cyan]{'='*60}[/cyan]")

            message = [
                f"Portfolio Statistics (Trade #{trade_count}):",
                f"Initial Balance: ${stats.get('initial_balance', 0):.2f}",
                f"Current Value: ${stats.get('current_value', 0):.2f}",
                f"Total Return: {stats.get('total_return_pct', 0):.2f}%",
                f"Total Trades: {stats.get('total_trades', 0)}",
                f"Win Rate: {stats.get('win_rate_pct', 0):.1f}%",
                f"Total P&L: ${stats.get('total_pnl', 0):.2f}",
                f"Total Slippage Cost: ${stats.get('total_slippage_cost', 0):.2f}",
                f"Current Position: {stats.get('current_position', 0):.4f}",
                f"Available Cash: ${stats.get('current_cash', 0):.2f}",
            ]

            data = {
                "content": "\n".join(message),
            }
            if webhook_url:
                requests.post(webhook_url, json=data)

    return enhanced_portfolio_callback

async def run_single_coin_test(webhook_url: str = None, private_key: str = None):
    """Run a single coin portfolio test using Binance signals and Jupiter execution."""
    
    wallethandler = jup.JupiterWalletHandler(private_key)
    
    portfolio = SimulatedPortfolio(initial_balance=200.0, slippage_rate=0.005)
    console = Console()
    
    callback = create_enhanced_portfolio_callback(portfolio, wallethandler, console, log_interval=1, webhook_url=webhook_url)
        
    system = LiveTradingSystem(
        symbol="SOL-USDT",
        interval="1m",
        data_source="binance",
        buffer_size=500,
        strategy_func=strategy.signal_spam,
        strategy_params={},
        signal_callback=callback
    )
    
    if webhook_url:
        requests.post(webhook_url, json={"content": f"Starting Binance-Jupiter Trading System | Strategy: {system.strategy_func.__name__}"})
    
    console.print("[green]Starting Binance-Jupiter Trading System[/green]")
    console.print(f"[blue]Signal Source: Binance {system.symbol}[/blue]")
    console.print(f"[blue]Execution Source: Jupiter SOL-USDC[/blue]")
    console.print(f"[blue]Strategy: {system.strategy_func.__name__}[/blue]")
    console.print(f"[blue]Parameters: {system.strategy_params}[/blue]")
    console.print(f"[blue]Initial Balance: ${portfolio.initial_balance:.2f}[/blue]")
    console.print(f"[blue]Default Slippage Rate: {portfolio.slippage_rate*100:.2f}%[/blue]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")
    
    try:
        await system.start()
    except KeyboardInterrupt:
        console.print("\n[red]Shutting down...[/red]")
        system.stop()
        
        final_stats = portfolio.get_statistics()
        console.print(f"\n[cyan]{'='*60}[/cyan]")
        console.print(f"[cyan]FINAL PORTFOLIO SUMMARY[/cyan]")
        console.print(f"[cyan]{'='*60}[/cyan]")
        console.print(f"Initial Balance: ${final_stats.get('initial_balance', 0):.2f}")
        console.print(f"Final Value: ${final_stats.get('current_value', 0):.2f}")
        console.print(f"Total Return: {final_stats.get('total_return_pct', 0):.2f}%")
        console.print(f"Total Trades: {final_stats.get('total_trades', 0)}")
        console.print(f"Win Rate: {final_stats.get('win_rate_pct', 0):.1f}%")
        console.print(f"Total P&L: ${final_stats.get('total_pnl', 0):.2f}")
        console.print(f"Total Slippage Cost: ${final_stats.get('total_slippage_cost', 0):.2f}")
        console.print(f"[cyan]{'='*60}[/cyan]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Binance→Jupiter trading system")
    parser.add_argument("--webhook_url", type=str, default=None, help="Discord webhook URL for notifications")
    parser.add_argument("--private_key", type=str, default=None, help="Solana private key for Jupiter wallet")
    args = parser.parse_args()
    
    asyncio.run(run_single_coin_test(webhook_url="", private_key="2BmZhw6gq2VyyvQNhzbXSPp1riXVDQqfiBNPeALf54gsZ9Wh4bLzQrzbysRUgxZVmi862VcXTwFvcAnfC1KYwWsz"))