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

from live_system import LiveTradingSystem, SimulatedPortfolio
import strategy

def create_enhanced_portfolio_callback(portfolio: SimulatedPortfolio, console=None, log_interval: int = 5, webhook_url: str = None):
    """
    Create an enhanced callback with more detailed logging and statistics.
    
    Args:
        portfolio: SimulatedPortfolio instance
        console: Rich console for output (optional)
        log_interval: How often to log detailed statistics (every N trades)
    
    Returns:
        Enhanced callback function
    """
    trade_count = 0
    
    def enhanced_portfolio_callback(signal_info: Dict):
        nonlocal trade_count
        
        timestamp = signal_info['timestamp']
        current_price = signal_info['current_price']
        new_signal = signal_info['new_signal']
        
        # Execute trade
        trade_result = portfolio.execute_trade(new_signal, current_price, timestamp)
        
        # Log trade execution
        if trade_result['action'] != 'HOLD':
            trade_count += 1
            action = trade_result['action']
            position_change = trade_result['position_change']
            slippage_cost = trade_result['slippage_cost']
            post_value = trade_result['post_trade_value']
            execution_price = trade_result['execution_price']
            

            # Create a detailed trade log
            trade_color = 'green' if 'BUY' in action else 'red' if 'SELL' in action else 'yellow'
            console.print(f"[{trade_color}]{timestamp.strftime('%H:%M:%S')} | TRADE #{trade_count} | {action}[/{trade_color}]")
            console.print(f"[{trade_color}]  Market: ${current_price:.2f} | Execution: ${execution_price:.2f} | Position: {position_change:+.4f} | Slippage: ${slippage_cost:.2f}[/{trade_color}]")
            
            # Log portfolio value change
            pre_value = trade_result['pre_trade_value']
            value_change = post_value - pre_value
            value_color = 'green' if value_change >= 0 else 'red'
            console.print(f"[{value_color}]  Portfolio: ${pre_value:.2f} → ${post_value:.2f} (${value_change:+.2f})[/{value_color}]")
            
            # Log realized P&L if position was closed
            if 'realized_pnl' in trade_result:
                realized_pnl = trade_result['realized_pnl']
                pnl_color = 'green' if realized_pnl >= 0 else 'red'
                console.print(f"[{pnl_color}]  Realized P&L: ${realized_pnl:+.2f}[/{pnl_color}]")
        
        # Log detailed statistics periodically
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

            message = (
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
            )

            data = {
                "content": "\n".join(message),
                }
            if webhook_url:
                requests.post(webhook_url, json=data)

    return enhanced_portfolio_callback

async def run_single_coin_test(webhook_url: str = None):
    """Run a single coin portfolio test."""
    
    # Create portfolio with $10,000 initial balance and 0.1% slippage
    portfolio = SimulatedPortfolio(initial_balance=10000.0, slippage_rate=0.001)
    console = Console()
    
    callback = create_enhanced_portfolio_callback(portfolio, console, log_interval=1, webhook_url=webhook_url)
        
    system = LiveTradingSystem(
        symbol="SOL-USDT",
        interval="5m",
        data_source="binance",
        buffer_size=500,
        strategy_func=strategy.signal_spam,
        strategy_params={},
        signal_callback=callback
    )
    if webhook_url:
        requests.post(webhook_url, json={"content": f"Starting Single Coin Portfolio Test {system.strategy_func.__name__}"})
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--webhook_url", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run_single_coin_test(args.webhook_url)) 