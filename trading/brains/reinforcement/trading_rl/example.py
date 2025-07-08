"""
Simple example demonstrating PPO trading system usage.
"""

from train import PPOTrainer
from inference import load_and_test_model, PPOInference
from rich import print

def quick_training_example():
    """Quick training example with minimal episodes"""
    
    print("[bold blue]PPO Trading System - Quick Training Example[/bold blue]")
    
    # Create trainer with reasonable settings for quick demo
    trainer = PPOTrainer(
        symbol='BTC-USDT',
        chunks=5,  # Small dataset for quick demo
        interval='15m',
        age_days=3,
        context_length=10,
        data_source='binance',
        initial_capital=10000,
        slippage_pct=0.001,
        commission_fixed=0.0
    )
    
    print(f"[green]Environment setup complete:[/green]")
    print(f"  - Symbol: {trainer.env.symbol}")
    print(f"  - Data points: {len(trainer.env.data)}")
    print(f"  - State dimension: {trainer.env.state_dim}")
    print(f"  - Max steps per episode: {trainer.env.max_steps}")
    
    trained_agent = trainer.train(
        episodes=200,
        update_frequency=5,
        save_frequency=10,
        model_dir="example_models"
    )
    
    print("[bold green]Training completed![/bold green]")
    return trained_agent

def test_trained_model():
    """Test a trained model"""
    
    print("\n[bold blue]Testing Trained Model[/bold blue]")
    
    try:
        # Test the model
        results = load_and_test_model(
            model_path="example_models/ppo_model_final.pth",
            symbol="BTC-USDT",
            chunks=5,
            interval="15m",
            age_days=3
        )
        
        return results
        
    except FileNotFoundError:
        print("[red]No trained model found. Please run training first.[/red]")
        return None

def demonstrate_inference():
    """Demonstrate inference capabilities"""
    
    print("\n[bold blue]Inference Demonstration[/bold blue]")
    
    try:
        # Load model for inference
        inference = PPOInference("example_models/ppo_model_final.pth", state_dim=19)
        
        # Generate signals for test data
        signals = inference.generate_signals(
            symbol="BTC-USDT",
            chunks=3,
            interval="15m",
            age_days=1
        )
        
        print(f"[green]Generated {len(signals)} signals[/green]")
        print(f"Signal distribution:")
        print(f"  - Short (1): {(signals == 1).sum()}")
        print(f"  - Flat (2): {(signals == 2).sum()}")
        print(f"  - Long (3): {(signals == 3).sum()}")
        
        return signals
        
    except FileNotFoundError:
        print("[red]No trained model found for inference.[/red]")
        return None

def main():
    """Main demonstration function"""
    
    print("[bold cyan]PPO Trading System Demonstration[/bold cyan]")
    print("This example shows the complete workflow:\n")
    
    # Step 1: Training
    print("[yellow]Step 1: Training PPO Agent[/yellow]")
    trained_agent = quick_training_example()
    
    # Step 2: Testing
    print("\n[yellow]Step 2: Testing Trained Model[/yellow]")
    test_results = test_trained_model()
    
    if test_results:
        metrics = test_results['metrics']
        print(f"\n[bold green]Final Performance Summary:[/bold green]")
        print(f"  📈 Total Return: {metrics['Total_Return']*100:.2f}%")
        print(f"  📊 Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
        print(f"  🎯 Win Rate: {metrics['Win_Rate']*100:.1f}%")
        print(f"  📉 Max Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
        print(f"  🔄 Total Trades: {metrics['Total_Trades']}")
    
    # Step 3: Inference demo
    print("\n[yellow]Step 3: Signal Generation[/yellow]")
    signals = demonstrate_inference()
    
    print("\n[bold green]Demonstration completed![/bold green]")
    print("You can now:")
    print("  1. Run 'python train.py' for full training")
    print("  2. Run 'python inference.py' to test models")
    print("  3. Integrate signals into your trading system")

if __name__ == "__main__":
    main() 