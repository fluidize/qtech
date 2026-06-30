import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import PriceDataset

def model_to_signals(model, dataset, device: str = "cuda", batch_size: int = 32, eval_mode: bool = True):
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False,
    )
    
    num_sequences = len(dataset)
    predictions = torch.zeros(num_sequences, dtype=torch.float32, device=device)

    with (torch.enable_grad() if not eval_mode else torch.no_grad()):
        for batch_X, batch_indices in dataloader:
            batch_X = batch_X.to(device)
            batch_predictions = model.get_action(batch_X)
            predictions[batch_indices] = batch_predictions.float()

    raw_signals_t = torch.zeros(len(dataset.main_data), dtype=torch.float32, device=device)
    valid_positions = torch.tensor(
        dataset.main_data.index.get_indexer(dataset.valid_indices),
        dtype=torch.long, device=device
    )
    raw_signals_t[valid_positions] = predictions

    return raw_signals_t

class TorchBacktest:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dataset = None

    def load_dataset(self, dataset: PriceDataset):
        "Expects a dataset with .features and .main_data"
        self.dataset = dataset
        return self.dataset

    def _torch_shift(self, series, shift: int = 1):
        if shift > 0:
            return torch.cat([torch.zeros(shift, device=self.device, dtype=series.dtype), series[:-shift]])
        elif shift < 0:
            return torch.cat([series[-shift:], torch.zeros(-shift, device=self.device, dtype=series.dtype)])
        return series

    def _get_open_prices(self):
        return torch.tensor(
            self.dataset.main_data['Open'].values,
            dtype=torch.float32,
            device=self.device,
        )

    def _get_open_returns(self):
        open_prices = self._get_open_prices()
        open_next = self._torch_shift(open_prices, -1)
        open_returns = (open_next - open_prices) / open_prices.clamp(min=1e-12)
        open_returns[-1] = 0.0
        return open_returns

    def _get_strategy_returns(self, raw_signals, clamp_signals: bool = True):
        open_returns = self._get_open_returns()
        if clamp_signals:
            raw_signals = raw_signals.clamp(0.0, 1.0)
        position = self._torch_shift(raw_signals, 1)
        return torch.multiply(position, open_returns), open_returns

    def _cumulative_return(self, returns):
        return (1 + returns).prod() - 1

    def _geometric_mean_return(self, returns):
        one = torch.tensor(1.0, device=self.device, dtype=returns.dtype)
        cumprod = torch.cumprod(one + returns, dim=0)
        inv_N = torch.tensor(1.0 / len(returns), device=self.device, dtype=returns.dtype)
        return torch.pow(cumprod[-1].clamp(min=1e-12), inv_N) - one

    def _equity_curve(self, returns):
        one = torch.tensor(1.0, device=self.device, dtype=returns.dtype)
        return torch.cumprod(one + returns, dim=0)

    def get_hulltac(self, raw_signals):
        strategy_returns, open_returns = self._get_strategy_returns(raw_signals)
        strategy_std = torch.std(strategy_returns).clamp(min=1e-12)
        strategy_geometric_mean_return = self._geometric_mean_return(strategy_returns)

        market_std = torch.std(open_returns).clamp(min=1e-12)
        market_geometric_mean_return = self._geometric_mean_return(open_returns)

        strategy_sharpe = strategy_geometric_mean_return / strategy_std
        excess_volatility = torch.div(strategy_std, market_std).clamp(max=1.2)
        volatility_penalty = 1 + excess_volatility

        return_gap = (market_geometric_mean_return - strategy_geometric_mean_return).clamp(min=0.0)
        return_penalty = 1 + return_gap / 100

        return strategy_sharpe / (return_penalty * volatility_penalty)

    def get_sharpe(self, raw_signals):
        strategy_returns, _ = self._get_strategy_returns(raw_signals)
        strategy_std = torch.std(strategy_returns).clamp(min=1e-12)
        return strategy_returns.mean() / strategy_std

    def get_information_ratio(self, raw_signals):
        strategy_returns, open_returns = self._get_strategy_returns(raw_signals)
        active_returns = strategy_returns - open_returns
        tracking_error = torch.std(active_returns).clamp(min=1e-12)
        return active_returns.mean() / tracking_error

    def get_total_return(self, raw_signals):
        strategy_returns, _ = self._get_strategy_returns(raw_signals)
        return self._cumulative_return(strategy_returns)

    def get_excess_return(self, raw_signals):
        strategy_returns, open_returns = self._get_strategy_returns(raw_signals)
        return self._cumulative_return(strategy_returns) - self._cumulative_return(open_returns)

    def get_max_drawdown(self, raw_signals):
        strategy_returns, _ = self._get_strategy_returns(raw_signals)
        equity = self._equity_curve(strategy_returns)
        running_max = torch.cummax(equity, dim=0).values
        drawdown = (running_max - equity) / running_max.clamp(min=1e-12)
        return drawdown.max()

    def get_sortino(self, raw_signals, target_return=0.0):
        strategy_returns, _ = self._get_strategy_returns(raw_signals, clamp_signals=False)
        excess_returns = strategy_returns - target_return
        downside_returns = torch.clamp(excess_returns, max=0.0)
        downside_risk = torch.std(downside_returns).clamp(min=1e-6)
        return -excess_returns.mean() / downside_risk

    def get_turnover(self, raw_signals):
        return torch.abs(torch.diff(raw_signals, dim=0)).mean()

    def get_pos_penalty(self, raw_signals, penalty_factor=0.1):
        return penalty_factor * torch.mean(torch.exp(-torch.abs(raw_signals) * 5))

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
    def __init__(self, device: str = "cuda", sortino_weight=1, turnover_weight=0.05, undertrading_weight=0.01):
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