#!/usr/bin/env python3
"""
Single coin portfolio test using SimulatedPortfolio with LiveTradingSystem.
"""

import asyncio
import sys
from rich.console import Console

# Add the trading directory to the path
sys.path.append("trading")
sys.path.append("trading/backtesting")
sys.path.append("trading/live")

from live_system import LiveTradingSystem, SimulatedPortfolio, create_enhanced_portfolio_callback
import strategy

async def run_single_coin_test():
    """Run a single coin portfolio test."""
    
    # Create portfolio with $10,000 initial balance and 0.1% slippage
    portfolio = SimulatedPortfolio(initial_balance=10000.0, slippage_rate=0.001)
    console = Console()
    
    # Create enhanced callback with detailed logging every 5 trades
    callback = create_enhanced_portfolio_callback(portfolio, console, log_interval=5)
    
    # Create trading system
    system = LiveTradingSystem(
        symbol="SOL-USDT",
        interval="1m",
        data_source="binance",
        buffer_size=500,
        strategy_func=strategy.scalper_strategy,
        signal_callback=callback
    )
    
    console.print("[green]Starting Single Coin Portfolio Test[/green]")
    console.print(f"[blue]Symbol: {system.symbol}[/blue]")
    console.print(f"[blue]Strategy: {system.strategy_func.__name__}[/blue]")
    console.print(f"[blue]Parameters: {system.strategy_params}[/blue]")
    console.print(f"[blue]Initial Balance: ${portfolio.initial_balance:.2f}[/blue]")
    console.print(f"[blue]Slippage Rate: {portfolio.slippage_rate*100:.2f}%[/blue]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")
    
    try:
        await system.start()
    except KeyboardInterrupt:
        console.print("\n[red]Shutting down...[/red]")
        system.stop()
        
        # Final summary
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
    asyncio.run(run_single_coin_test()) 