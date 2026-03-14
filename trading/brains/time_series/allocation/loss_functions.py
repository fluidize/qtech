import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from sklearn.utils.class_weight import compute_class_weight

class TorchBacktest:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dataset = None

    def load_dataset(self, dataset):
        "Expects a dataset with .X and .data"
        self.dataset = dataset
        return self.dataset

    def _model_wrapper(self, model):
        X_tensor = self.dataset.X.to(self.device)
        predictions = model(X_tensor)

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

    def get_hulltac(self, model):
        # pct_change().shift(-1)
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)
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
    
    def get_sharpe(self, model):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        strategy_std = torch.std(strategy_returns).clamp(min=1e-12)

        sharpe = strategy_returns.mean() / strategy_std

        return sharpe
    
    def get_total_return(self, model):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)  
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0 

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)

        total_return = (1 + strategy_returns).prod() - 1
        return total_return

    def get_excess_return(self, model):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)
        strategy_returns = torch.multiply(position, open_returns)

        strategy_total = (1 + strategy_returns).prod() - 1
        benchmark_total = (1 + open_returns).prod() - 1
        return strategy_total - benchmark_total

    def get_max_drawdown(self, model):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = self._model_wrapper(model).clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        equity = torch.cumprod(torch.add(torch.tensor(1.0, device=self.device, dtype=strategy_returns.dtype), strategy_returns), dim=0)
        running_max = torch.cummax(equity, dim=0).values
        drawdown = (running_max - equity) / running_max.clamp(min=1e-12)
        return drawdown.max()

class SharpeLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, model, dataset):
        tb = TorchBacktest(device=self.device)
        tb.load_dataset(dataset)
        loss = tb.get_sharpe(model)
        return -loss

class TRLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, model, dataset):
        tb = TorchBacktest(device=self.device)
        tb.load_dataset(dataset)
        loss = tb.get_total_return(model)
        return -loss


class ExcessReturnLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, model, dataset):
        tb = TorchBacktest(device=self.device)
        tb.load_dataset(dataset)
        excess = tb.get_excess_return(model)
        return -excess

class HullTacLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, model, dataset):
        tb = TorchBacktest(device=self.device)
        tb.load_dataset(dataset)
        loss = tb.get_hulltac(model)
        return -loss

class CombinedAllocLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def forward(self, model, dataset):
        tb = TorchBacktest(device=self.device)
        tb.load_dataset(dataset)
        sharpe = tb.get_sharpe(model)
        max_dd = tb.get_max_drawdown(model)
        loss = sharpe * (1 + max_dd).clamp(min=0.0)
        return loss

class NegativeLogLikelihoodLoss(nn.Module):
    def forward(self, y, target):
        mean = y[:, 0]
        std = y[:, 1].clamp(min=0.01)
        nll = -Normal(mean, std).log_prob(target)
        return nll.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, target: torch.Tensor):
        super().__init__()
        self.num_classes = num_classes
        weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=target.cpu().numpy().astype(int))
        weights = np.nan_to_num(weights, posinf=1.0).astype(np.float32)
        self.register_buffer('weight', torch.tensor(weights))

    def forward(self, y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(y, target.long(), weight=self.weight)