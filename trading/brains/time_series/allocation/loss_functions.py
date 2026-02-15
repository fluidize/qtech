import torch
from torch import nn

class TorchBacktest:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dataset = None

    def load_dataset(self, dataset):
        "Expects a dataset with .X and .data"
        self.dataset = dataset
        return self.dataset
    
    def _model_wrapper(self, alloc_model, directional_model):
        X_tensor = self.dataset.X.to(self.device)

        directional_estimate = directional_model(X_tensor)
        predictions = alloc_model(X_tensor, directional_estimate)

        raw_signals_t = torch.zeros(len(self.dataset.data), dtype=torch.float32, device=self.device)
        valid_positions = torch.tensor(
            self.dataset.data.index.get_indexer(self.dataset.valid_indices),
            dtype=torch.long, device=self.device
        )
        raw_signals_t[valid_positions] = predictions
        return raw_signals_t

    def _torch_shift(self, series, shift: int = 1):
        if shift > 0:
            return torch.cat([torch.zeros(shift, device=self.device, dtype=series.dtype), series[:-shift]])
        elif shift < 0:
            return torch.cat([series[-shift:], torch.zeros(-shift, device=self.device, dtype=series.dtype)])
        else:
            return series

    def _exec_backtest_hulltac(self, alloc_model, directional_model):
        # pct_change().shift(-1)
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(alloc_model, directional_model).clamp(0.0, 1.0)

        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        strategy_std = torch.std(strategy_returns).clamp(min=1e-12)
        
        one = torch.tensor(1.0, device=self.device, dtype=strategy_returns.dtype)
        strategy_returns_cumprod = torch.cumprod(torch.add(one, strategy_returns), dim=0)
        
        inv_N = torch.tensor(1.0 / len(strategy_returns), device=self.device, dtype=strategy_returns.dtype)
        strategy_geometric_mean_return = torch.subtract(
            torch.pow(strategy_returns_cumprod[-1].clamp(min=1e-12), inv_N), one
        )

        market_std = torch.std(open_returns).clamp(min=1e-12)

        one = torch.tensor(1.0, device=self.device, dtype=strategy_returns.dtype)
        market_returns_cumprod = torch.cumprod(torch.add(one, open_returns), dim=0)

        inv_N = torch.tensor(1.0 / len(open_returns), device=self.device, dtype=open_returns.dtype)
        market_geometric_mean_return = torch.subtract(
            torch.pow(market_returns_cumprod[-1].clamp(min=1e-12), inv_N), one
        )

        strategy_sharpe = strategy_geometric_mean_return / strategy_std

        excess_volatility = torch.divide(strategy_std, market_std).clamp(max=1.2)
        volatility_penalty = 1 + excess_volatility
        
        return_gap = torch.subtract(market_geometric_mean_return, strategy_geometric_mean_return).clamp(min=0.0)
        return_penalty = 1 + return_gap / 100

        loss = strategy_sharpe / (return_penalty * volatility_penalty)
        return loss
    
    def get_sharpe(self, alloc_model, directional_model):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(alloc_model, directional_model).clamp(0.0, 1.0)

        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        strategy_std = torch.std(strategy_returns).clamp(min=1e-12)

        sharpe = strategy_returns.mean() / strategy_std

        return sharpe

class SharpeLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, alloc_model, directional_model, dataset):
        tb = TorchBacktest(device=self.device)
        tb.load_dataset(dataset)
        loss = tb.get_sharpe(alloc_model, directional_model)
        return -loss

class IntervalLoss(nn.Module):
    def __init__(self, width_weight: float = 0.1, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.width_weight = width_weight

    def forward(self, y, target):
        target = target.squeeze().float()
        lower_bound = y[:, 0]
        upper_bound = y[:, 1]

        outside_lower = (lower_bound - target).clamp(min=0)
        outside_upper = (target - upper_bound).clamp(min=0)
        coverage_penalty = (outside_lower + outside_upper).pow(2)

        width_penalty = (upper_bound - lower_bound).clamp(min=1e-6)

        loss = coverage_penalty.mean() + self.width_weight * width_penalty.mean()
        return loss