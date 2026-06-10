import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import PriceDataset

class TorchBacktest:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dataset = None

    def load_dataset(self, dataset: PriceDataset):
        "Expects a dataset with .X and .data"
        self.dataset = dataset
        return self.dataset

    def _torch_shift(self, series, shift: int = 1):
        if shift > 0:
            return torch.cat([torch.zeros(shift, device=self.device, dtype=series.dtype), series[:-shift]])
        elif shift < 0:
            return torch.cat([series[-shift:], torch.zeros(-shift, device=self.device, dtype=series.dtype)])
        else:
            return series

    def get_hulltac(self, raw_signals):
        # pct_change().shift(-1)
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals.clamp(0.0, 1.0)
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
    
    def get_sharpe(self, raw_signals):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals.clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        strategy_std = torch.std(strategy_returns).clamp(min=1e-12)

        sharpe = strategy_returns.mean() / strategy_std

        return sharpe

    def get_information_ratio(self, raw_signals):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals.clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        active_returns = strategy_returns - open_returns
        tracking_error = torch.std(active_returns).clamp(min=1e-12)
        return active_returns.mean() / tracking_error

    def get_total_return(self, raw_signals):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals.clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)

        total_return = (1 + strategy_returns).prod() - 1
        return total_return

    def get_excess_return(self, raw_signals):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals.clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)
        strategy_returns = torch.multiply(position, open_returns)

        strategy_total = (1 + strategy_returns).prod() - 1
        benchmark_total = (1 + open_returns).prod() - 1
        return strategy_total - benchmark_total

    def get_max_drawdown(self, raw_signals):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals.clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        equity = torch.cumprod(torch.add(torch.tensor(1.0, device=self.device, dtype=strategy_returns.dtype), strategy_returns), dim=0)
        running_max = torch.cummax(equity, dim=0).values
        drawdown = (running_max - equity) / running_max.clamp(min=1e-12)
        return drawdown.max()

    def get_sortino(self, raw_signals, target_return=0.0):
        open_prices = torch.tensor(self.dataset.data['Open'].values, dtype=torch.float32, device=self.device)
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0

        raw_signals_t = raw_signals
        position = self._torch_shift(raw_signals_t, 1)

        strategy_returns = torch.multiply(position, open_returns)
        excess_returns = strategy_returns - target_return
        downside_returns = torch.clamp(excess_returns, max=0.0)
        downside_risk = torch.std(downside_returns).clamp(min=1e-6)
        return -excess_returns.mean() / downside_risk

    def get_turnover(self, raw_signals):
        position_changes = torch.diff(raw_signals, dim=0)
        turnover = torch.abs(position_changes).mean()
        return turnover

    def get_pos_penalty(self, raw_signals, penalty_factor=0.1):
        near_zero_penalty = torch.mean(torch.exp(-torch.abs(raw_signals) * 5))
        return penalty_factor * near_zero_penalty

class SharpeLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tb = TorchBacktest(device=device)

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)
        loss = self.tb.get_sharpe(raw_signals)
        return -loss

class InformationRatioLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tb = TorchBacktest(device=device)

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)
        ir = self.tb.get_information_ratio(raw_signals)
        return -ir

class TotalReturnLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tb = TorchBacktest(device=device)

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)
        loss = self.tb.get_total_return(raw_signals)
        return -loss

class ExcessReturnLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tb = TorchBacktest(device=device)

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)
        excess = self.tb.get_excess_return(raw_signals)
        return -excess

class HullTacLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tb = TorchBacktest(device=device)

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)
        loss = self.tb.get_hulltac(raw_signals)
        return -loss

class TurnoverLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tb = TorchBacktest(device=device)

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)

        position_changes = torch.diff(raw_signals, dim=0)
        turnover = torch.abs(position_changes).mean()

        return turnover

class AllocationLoss(nn.Module):
    def __init__(self, device: str = "cuda", sortino_weight=1.0, turnover_weight=0.01, undertrading_weight=0.1):
        super().__init__()
        self.tb = TorchBacktest(device=device)
        self.sortino_weight = sortino_weight
        self.turnover_weight = turnover_weight
        self.undertrading_weight = undertrading_weight

    def forward(self, raw_signals, dataset):
        self.tb.load_dataset(dataset)
        sortino_value = self.tb.get_sortino(raw_signals)
        turnover_value = self.tb.get_turnover(raw_signals)
        undertrading_value = self.tb.get_pos_penalty(raw_signals)
        return (self.sortino_weight * sortino_value +
                self.turnover_weight * turnover_value +
                self.undertrading_weight * undertrading_value)