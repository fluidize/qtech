import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append("trading")
import technical_analysis as ta
import vb_metrics as metrics

class RegimeAnalyzer:
    """
    Analyzes trading strategy performance across different market regimes.
    
    Runs backtesting automatically and identifies periods of strong/weak performance
    correlated with various market indicators like volatility, trend strength, momentum, etc.
    """
    
    def __init__(self, backtest_instance, strategy_func, **strategy_kwargs):
        """
        Initialize the analyzer and run backtesting.
        
        Args:
            backtest_instance: VectorizedBacktesting instance with data already loaded
            strategy_func: Strategy function to run
            **strategy_kwargs: Additional arguments for the strategy function
        """
        self.backtest = backtest_instance
        self.regime_indicators = {}
        self.performance_metrics = {}
        
        # Run the backtest
        print("Running backtest...")
        self.data = self.backtest.run_strategy(strategy_func, **strategy_kwargs)
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Position', 'Strategy_Returns', 'Portfolio_Value']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after backtest: {missing_cols}")
            
        print(f"RegimeAnalyzer initialized with {len(self.data)} data points")
        
    def calculate_regime_indicators(self):
        """Calculate various market regime indicators."""
        print("Calculating regime indicators...")
        
        # Volatility indicators
        self.regime_indicators['ATR'] = ta.atr(self.data['High'], self.data['Low'], self.data['Close'])
        self.regime_indicators['Choppiness'] = ta.choppiness_index(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Trend indicators
        adx, plus_di, minus_di = ta.adx(self.data['High'], self.data['Low'], self.data['Close'])
        self.regime_indicators['ADX'] = adx
        self.regime_indicators['Plus_DI'] = plus_di
        self.regime_indicators['Minus_DI'] = minus_di
        
        # Market structure
        self.regime_indicators['Hurst'] = ta.hurst_exponent(self.data['Close'], max_lag=20)
        self.regime_indicators['ZScore'] = ta.zscore(self.data['Close'], timeperiod=20)
        
        # Momentum indicators
        self.regime_indicators['RSI'] = ta.rsi(self.data['Close'])
        macd, macd_signal = ta.macd(self.data['Close'])
        self.regime_indicators['MACD'] = macd
        self.regime_indicators['MACD_Signal'] = macd_signal
        
        # Market sentiment
        self.regime_indicators['ROC'] = ta.roc(self.data['Close'], timeperiod=10)
        self.regime_indicators['Momentum'] = ta.mom(self.data['Close'], timeperiod=10)
        
        # Volatility bands
        bb_upper, bb_middle, bb_lower = ta.bbands(self.data['Close'])
        self.regime_indicators['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # Volume indicators (if available)
        if 'Volume' in self.data.columns:
            self.regime_indicators['OBV'] = ta.obv(self.data['Close'], self.data['Volume'])
            self.regime_indicators['MFI'] = ta.mfi(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])
        
        print("Regime indicators calculated successfully.")
        
    def analyze_regime_performance(self) -> Dict:
        """Analyze strategy performance across different market regimes."""
        if not self.regime_indicators:
            self.calculate_regime_indicators()
            
        analysis = {}
        
        # Classify market regimes
        market_regimes = self._classify_market_regimes()
        analysis['market_regimes'] = market_regimes
        
        # Performance by market regime
        analysis['regime_stats'] = {}
        for regime_type in market_regimes.unique():
            if regime_type != 'Unknown':
                regime_mask = market_regimes == regime_type
                regime_returns = self.data.loc[regime_mask, 'Strategy_Returns']
                
                analysis['regime_stats'][regime_type] = {
                    'count': len(regime_returns),
                    'avg_return': regime_returns.mean(),
                    'total_return': regime_returns.sum(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).mean(),
                    'avg_adx': self.regime_indicators['ADX'].loc[regime_mask].mean() if 'ADX' in self.regime_indicators else np.nan,
                    'avg_volatility': self.regime_indicators['ATR'].loc[regime_mask].mean() if 'ATR' in self.regime_indicators else np.nan,
                    'avg_hurst': self.regime_indicators['Hurst'].loc[regime_mask].mean() if 'Hurst' in self.regime_indicators else np.nan,
                }
        
        # Correlation analysis between indicators and strategy returns
        corr_data = self.data[['Strategy_Returns']].copy()
        for name, indicator in self.regime_indicators.items():
            if isinstance(indicator, pd.Series) and len(indicator) == len(corr_data):
                corr_data[name] = indicator
                
        analysis['correlations'] = corr_data.corr()[['Strategy_Returns']].drop(['Strategy_Returns'])
        
        self.performance_metrics = analysis
        return analysis
    
    def _classify_market_regimes(self) -> pd.Series:
        """Classify market conditions into regimes based on multiple indicators."""
        regimes = pd.Series('Unknown', index=self.data.index)
        
        if 'ADX' in self.regime_indicators and 'Hurst' in self.regime_indicators:
            adx = self.regime_indicators['ADX']
            hurst = self.regime_indicators['Hurst']
            volatility = self.regime_indicators.get('ATR', pd.Series(0, index=self.data.index))
            
            # High trend + high persistence = Strong Trend
            strong_trend = (adx > 25) & (hurst > 0.55)
            regimes[strong_trend] = 'Strong_Trend'
            
            # Low trend + mean reversion = Sideways
            sideways = (adx < 20) & (hurst < 0.45)
            regimes[sideways] = 'Sideways'
            
            # High volatility = Volatile
            if volatility.std() > 0:
                vol_threshold = volatility.quantile(0.75)
                volatile = volatility > vol_threshold
                regimes[volatile] = 'Volatile'
            
            # Default to trending for moderate conditions
            trending = (adx >= 20) & (adx <= 25) & (hurst >= 0.45) & (hurst <= 0.55)
            regimes[trending] = 'Trending'
            
        return regimes
    
    def plot_simple_analysis(self, show_plot: bool = True) -> go.Figure:
        """
        Create a simple regime analysis plot.
        
        Args:
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object
        """
        if not self.regime_indicators:
            self.calculate_regime_indicators()
            
        fig = go.Figure()
        
        # Plot portfolio value
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Portfolio_Value'], 
                      name='Portfolio Value', line=dict(color='blue'))
        )
        
        # Add market regime backgrounds
        market_regimes = self._classify_market_regimes()
        regime_colors = {
            'Strong_Trend': 'green', 
            'Trending': 'lightgreen',
            'Sideways': 'orange', 
            'Volatile': 'red'
        }
        
        for regime_type in regime_colors.keys():
            regime_periods = self.data[market_regimes == regime_type]
            if not regime_periods.empty:
                # Simple background coloring for regime periods
                for start_idx in regime_periods.index:
                    fig.add_vrect(
                        x0=start_idx-0.5, x1=start_idx+0.5,
                        fillcolor=regime_colors[regime_type], opacity=0.2,
                        layer="below", line_width=0
                    )
        
        fig.update_layout(
            title="Strategy Performance with Market Regime Analysis",
            xaxis_title="Time",
            yaxis_title="Portfolio Value",
            template="plotly_dark"
        )
        
        if show_plot:
            fig.show()
            
        return fig
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get a summary table of performance by market regime."""
        if not self.performance_metrics:
            self.analyze_regime_performance()
        
        if 'regime_stats' in self.performance_metrics:
            summary_data = []
            for regime, stats in self.performance_metrics['regime_stats'].items():
                summary_data.append({
                    'Regime': regime,
                    'Count': stats['count'],
                    'Avg_Return_pct': stats['avg_return'] * 100,
                    'Total_Return_pct': stats['total_return'] * 100,
                    'Sharpe_Ratio': stats['sharpe'],
                    'Win_Rate_pct': stats['win_rate'] * 100,
                    'Avg_ADX': stats['avg_adx'],
                    'Avg_Volatility': stats['avg_volatility'],
                    'Avg_Hurst': stats['avg_hurst']
                })
            
            return pd.DataFrame(summary_data)
        
        return pd.DataFrame()
    
    def identify_best_regimes(self, metric: str = 'total_return') -> Dict:
        """
        Identify which market regimes the strategy performs best in.
        
        Args:
            metric: Performance metric to use ('total_return', 'sharpe', 'win_rate')
            
        Returns:
            Dictionary with best performing conditions
        """
        if not self.performance_metrics:
            self.analyze_regime_performance()
        
        results = {
            'best_regime': None,
            'regime_analysis': {},
            'recommendations': []
        }
        
        # Analyze market regimes
        if 'regime_stats' in self.performance_metrics:
            regime_performance = self.performance_metrics['regime_stats']
            
            # Find best performing regime
            if regime_performance:
                best_regime = max(regime_performance.items(), key=lambda x: x[1][metric])
                worst_regime = min(regime_performance.items(), key=lambda x: x[1][metric])
                
                results['best_regime'] = {
                    'regime': best_regime[0],
                    'metric_value': best_regime[1][metric],
                    'stats': best_regime[1]
                }
                
                results['regime_analysis'] = regime_performance
                
                # Generate recommendations
                results['recommendations'] = [
                    f"Strategy performs best in {best_regime[0]} markets (Total Return: {best_regime[1]['total_return']*100:.2f}%)",
                    f"Strategy performs worst in {worst_regime[0]} markets (Total Return: {worst_regime[1]['total_return']*100:.2f}%)",
                    f"Consider position sizing adjustments during {worst_regime[0]} periods",
                    f"Monitor ADX levels - strategy shows different performance at various trend strengths"
                ]
        
        return results

    def get_basic_metrics(self) -> Dict:
        """Get basic strategy performance metrics."""
        return self.backtest.get_performance_metrics()

if __name__ == "__main__":
    # Example usage
    from backtesting import VectorizedBacktesting, Strategy

    vb = VectorizedBacktesting()
    vb.fetch_data(symbol="BTC-USDT", interval="5min", chunks=10, age_days=0)

    ra = RegimeAnalyzer(vb, Strategy.ETHBTC_trader, chunks=vb.chunks, interval=vb.interval, age_days=vb.age_days)
    analysis = ra.analyze_regime_performance()
    print(analysis)


