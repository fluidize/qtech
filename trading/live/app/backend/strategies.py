import pandas as pd
import torch
from pathlib import Path

import trading.technical_analysis as ta

ALLOCATOR_FEATURES = 10
ALLOCATOR_DEVICE = "cpu"
ALLOCATOR_MODEL_PATH = "/home/fluidize/Documents/qtech/trading/brains/allocation/allocator.pth"


def ema_cross(
    data: pd.DataFrame, fast_period: int = 20, slow_period: int = 50
) -> pd.Series:
    close = data["Close"]
    fast = ta.ema(close, timeperiod=fast_period)
    slow = ta.ema(close, timeperiod=slow_period)

    signals = pd.Series(0, index=data.index, dtype=float)
    signals[fast > slow] = 1.0
    signals[fast < slow] = -1.0
    return signals


def mean_reversion(
    data: pd.DataFrame, window: int = 20, z_threshold: float = 1.5
) -> pd.Series:
    close = data["Close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std

    signals = pd.Series(0, index=data.index, dtype=float)
    signals[z > z_threshold] = -1.0
    signals[z < -z_threshold] = 1.0
    return signals


def supertrend(
    data: pd.DataFrame, period: int = 10, multiplier: int = 3
) -> pd.Series:
    direction = ta.supertrend_direction(
        data["High"], data["Low"], data["Close"], period=period, multiplier=multiplier
    )
    return direction.fillna(0.0)


def allocator(
    data: pd.DataFrame,
    seq_len: int = 8,
    batch_size: int = 4096,
    model_path: str = ALLOCATOR_MODEL_PATH,
) -> pd.Series:
    from trading.brains.allocation.models.deep_models import AllocatorPolicy
    from trading.brains.allocation.datasets import PriceDataset

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"allocator weights not found at {path}")

    model = AllocatorPolicy(in_channels=ALLOCATOR_FEATURES, in_width=seq_len)
    model.load_state_dict(torch.load(path, map_location=ALLOCATOR_DEVICE))
    model.eval()

    dataset = PriceDataset(data, add_ticker="", seq_len=seq_len)
    preds = torch.zeros(len(dataset), dtype=torch.float32, device=ALLOCATOR_DEVICE)
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset.X[i : i + batch_size].to(ALLOCATOR_DEVICE)
            preds[i : i + batch_size] = model.get_action(batch)

    raw = torch.zeros(len(data), dtype=torch.float32, device=ALLOCATOR_DEVICE)
    rows = torch.tensor(
        data.index.get_indexer(dataset.valid_indices),
        dtype=torch.long,
        device=ALLOCATOR_DEVICE,
    )
    raw[rows] = preds
    return pd.Series(raw.cpu().numpy(), index=data.index).clip(0.0, 1.0)
