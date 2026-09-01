from pathlib import Path

import pandas as pd
import torch

from trading.brains.allocation.models.deep_models import AllocatorPolicy
from trading.brains.allocation.datasets import PriceDataset

# directory scanned for *.pth allocator weights, next to the live app package
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_FEATURES = 10
MODEL_DEVICE = "cpu"


def _run_model(data, pth, seq_len, batch_size) -> pd.Series:
    model = AllocatorPolicy(in_channels=MODEL_FEATURES, in_width=seq_len)
    model.load_state_dict(torch.load(pth, map_location=MODEL_DEVICE))
    model.eval()

    dataset = PriceDataset(data, add_ticker="", seq_len=seq_len)
    preds = torch.zeros(len(dataset), dtype=torch.float32, device=MODEL_DEVICE)
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset.X[i : i + batch_size].to(MODEL_DEVICE)
            preds[i : i + batch_size] = model.get_action(batch)

    raw = torch.zeros(len(data), dtype=torch.float32, device=MODEL_DEVICE)
    rows = torch.tensor(
        data.index.get_indexer(dataset.valid_indices),
        dtype=torch.long,
        device=MODEL_DEVICE,
    )
    raw[rows] = preds
    return pd.Series(raw.cpu().numpy(), index=data.index).clip(0.0, 1.0)


def make_model_strategy(pth: Path):
    pth = Path(pth)

    def model_strategy(data, seq_len=8, batch_size=4096) -> pd.Series:
        if not pth.exists():
            raise FileNotFoundError(f"model weights not found at {pth}")
        return _run_model(data, pth, seq_len=seq_len, batch_size=batch_size)

    return model_strategy


def get_model_strategies() -> dict:
    """Autodetect *.pth allocator weights in MODELS_DIR as named strategies."""
    if not MODELS_DIR.exists():
        return {}
    strategies = {}
    for pth in sorted(MODELS_DIR.glob("*.pth")):
        name = pth.stem
        strategies[name] = make_model_strategy(pth)
    return strategies
