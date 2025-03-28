import fintech.trading.algorithmic.spot_trade_env as spot
from fintech.trading.algorithmic.spot_trade_env import BacktestEnvironment
from fintech.trading.algorithmic.spot_trade_env import TradingEnvironment
from numba import jit
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm

class GridSearch:
    def __init__(self, env: TradingEnvironment):
        self.env = env
        self.current_symbol = env.get_current_symbol()
        self.results = []

    def Custom_Scalper_GridSearch(self, data: pd.DataFrame, std_threshold: float, pct_threshold: float, window_size: int = 5):
        std = spot._calculate_std(data, window_size)
        
        closes = data['Close'].values
        prev_closes = np.roll(closes, 1)
        price_change_pct = ((closes - prev_closes) / prev_closes) * 100
        
        buy_signals = (price_change_pct > pct_threshold) & (std > std_threshold)
        sell_signals = price_change_pct < -pct_threshold
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'timestamps': data['Datetime'].values,
            'prices': closes
        }
    
    def evaluate_parameters(self, std_threshold: float, pct_threshold: float) -> float:
        self.env.reset()
        
        data = self.env.data[self.current_symbol]
        
        signals = self.Custom_Scalper_GridSearch(data, std_threshold, pct_threshold)
        
        for i in range(len(data)):
            current_signals = {
                'buy': signals['buy_signals'][i],
                'sell': signals['sell_signals'][i]
            }
            self.env.step()
            self.env.execute_dict(current_signals)
        
        summary = self.env.get_summary()
        return summary['PT Ratio']
    
    def run(self, grid1: np.ndarray,
            grid2: np.ndarray,
            show_graph: bool = False):

        self.env.fetch_data()
        print("Starting Grid Search")
        
        results = pd.DataFrame(columns=['grid1', 'grid2', 'pt_ratio'])
        
        total_iterations = len(grid1) * len(grid2)
        progress_bar = tqdm(total=total_iterations, desc="Grid Search Progress")
        
        for i, std_threshold in enumerate(grid1):
            for j, pct_threshold in enumerate(grid2):
                pt_ratio = self.evaluate_parameters(std_threshold, pct_threshold)
                results = pd.concat([results, pd.DataFrame({'grid1': std_threshold, 'grid2': pct_threshold, 'pt_ratio': pt_ratio})], ignore_index=True)
                progress_bar.update(1)
        
        progress_bar.close()
        
        fig = go.Figure(data=go.Heatmap(
            z=results['pt_ratio'],
            x=grid2,
            y=grid1,
            colorscale='Viridis',
            colorbar=dict(title="Profit to Trade Ratio (%)"),
            text=np.round(results['pt_ratio'], 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverinfo='text',
        ))
        
        fig.update_layout(
            title=f"Grid Search Results - {self.current_symbol}",
            xaxis_title="Price Change Threshold (%)",
            yaxis_title="Standard Deviation Threshold",
            xaxis=dict(tickmode='linear'),
            yaxis=dict(tickmode='linear'),
            autosize=True,
        )
        
        if show_graph:
            fig.show()

# Example usage
env = TradingEnvironment(symbols=['SOL-USD'], initial_capital=1000, chunks=1, interval='1m', age_days=0)
grid_search = GridSearch(env)
results = grid_search.run(np.array([0.005, 0.01, 0.02]), np.array([0.02, 0.05, 0.1]), show_graph=True)
        