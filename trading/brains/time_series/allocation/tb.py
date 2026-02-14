class TorchBacktest:
    def __init__(self, device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device
        self.dataset = None

    def load_dataset(self, dataset):
        "Expects a dataset with .X and .data"
        self.dataset = dataset
        return self.dataset
    
    def _model_wrapper(self, model):
        X_tensor = self.torch.tensor(self.dataset.X, dtype=self.torch.float32, device=self.device)
        predictions = model(X_tensor)
        raw_signals_t = self.torch.zeros(len(self.dataset.data), device=self.device, dtype=predictions.dtype)
        valid_positions = self.dataset.data.index.get_indexer(self.dataset.valid_indices)
        raw_signals_t[valid_positions] = predictions
        return raw_signals_t

    def _torch_shift(self, series, shift: int = 1):
        if shift > 0:
            return self.torch.cat([self.torch.zeros(shift, device=self.device, dtype=series.dtype), series[:-shift]])
        elif shift < 0:
            return self.torch.cat([series[-shift:], self.torch.zeros(-shift, device=self.device, dtype=series.dtype)])
        else:
            return series

    def _exec_backtest_hulltac(self, model):
        # pct_change().shift(-1)
        open_prices = self.torch.tensor(self.dataset.data['Open'].values, dtype=self.torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)

        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = self.torch.multiply(position, open_returns)
        strategy_std = self.torch.std(strategy_returns).clamp(min=1e-12)
        
        one = self.torch.tensor(1.0, device=self.device, dtype=strategy_returns.dtype)
        strategy_returns_cumprod = self.torch.cumprod(self.torch.add(one, strategy_returns), dim=0)
        
        inv_N = self.torch.tensor(1.0 / len(strategy_returns), device=self.device, dtype=strategy_returns.dtype)
        strategy_geometric_mean_return = self.torch.subtract(
            self.torch.pow(strategy_returns_cumprod[-1].clamp(min=1e-12), inv_N), one
        )

        market_std = self.torch.std(open_returns).clamp(min=1e-12)

        one = self.torch.tensor(1.0, device=self.device, dtype=strategy_returns.dtype)
        market_returns_cumprod = self.torch.cumprod(self.torch.add(one, open_returns), dim=0)

        inv_N = self.torch.tensor(1.0 / len(open_returns), device=self.device, dtype=open_returns.dtype)
        market_geometric_mean_return = self.torch.subtract(
            self.torch.pow(market_returns_cumprod[-1].clamp(min=1e-12), inv_N), one
        )

        strategy_sharpe = strategy_geometric_mean_return / strategy_std

        excess_volatility = self.torch.divide(strategy_std, market_std).clamp(max=1.2)
        volatility_penalty = 1 + excess_volatility
        
        return_gap = self.torch.subtract(market_geometric_mean_return, strategy_geometric_mean_return).clamp(min=0.0)
        return_penalty = 1 + return_gap / 100

        loss = strategy_sharpe / (return_penalty * volatility_penalty)
        return loss
    
    def _exec_backtest(self, model):
        open_prices = self.torch.tensor(self.dataset.data['Open'].values, dtype=self.torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)

        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = self.torch.multiply(position, open_returns)
        strategy_std = self.torch.std(strategy_returns).clamp(min=1e-12)
        
        return strategy_returns.mean() / strategy_std

    def run_model(self, model):
        return self._exec_backtest(model)