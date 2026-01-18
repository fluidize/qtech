import numpy as np
from trading.backtesting.backtesting import VectorizedBacktesting
import vb_metrics as metrics

class MonteCarloAnalysis:
    def __init__(self, strategy, strategy_params, engine: VectorizedBacktesting):
        self.strategy = strategy
        self.strategy_params = strategy_params
        self.engine = engine

        self.pnls = None
    
    def build_distribution(self):
        self.engine.run_strategy(self.strategy, **self.strategy_params)
        self.pnls = metrics.get_trade_pnls(self.engine.data['Position'], self.engine.data['Open'], self.engine.initial_capital)

    def run_simulation(self, num_simulations: int, num_trades: int):
        if self.pnls is None:
            raise ValueError("Distribution not built. Call build_distribution() first.")
        simulated_pnls = np.random.choice(self.pnls, size=(num_simulations, num_trades), replace=True)
        return simulated_pnls
    
    def spaghetti_plot(self, num_simulations: int, num_trades: int):
        if self.pnls is None:
            raise ValueError("Distribution not built. Call build_distribution() first.")
        import matplotlib.pyplot as plt
        simulated_pnls = self.run_simulation(num_simulations, num_trades)
        simulated_portfolios = np.cumsum(simulated_pnls, axis=0)
        plt.figure(figsize=(10, 6))

        plt.plot(simulated_portfolios)
        plt.axhline(y=np.median(simulated_portfolios[:,-1]), color='red', linestyle='--')

        plt.title(f"Spaghetti Plot of {num_simulations} Simulations of {num_trades} Trades")
        plt.xlabel("Trade Number")
        plt.ylabel("P&L ($)")
        plt.show()