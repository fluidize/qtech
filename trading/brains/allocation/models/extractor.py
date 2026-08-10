import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WINDOW = 4


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-8))


def identity(x: torch.Tensor):
    return x


def same_pad_1d(x: torch.Tensor, window: int):
    total_pad = window - 1
    left = total_pad // 2
    right = total_pad - left
    return F.pad(x, (left, right))


def sma(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    x_pad = same_pad_1d(x, window)
    return F.avg_pool1d(x_pad, kernel_size=window, stride=1, padding=0)


def rolling_std(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    mean = sma(x, window=window)
    mean_sq = sma(x.square(), window=window)
    variance = mean_sq - mean.square()
    return torch.sqrt(torch.clamp(variance, min=0.0))


def rolling_zscore(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    mean = sma(x, window=window)
    std = rolling_std(x, window=window)
    return (x - mean) / (std + 1e-8)


def rolling_max(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    x_pad = same_pad_1d(x, window)
    return F.max_pool1d(x_pad, kernel_size=window, stride=1, padding=0)


def rolling_min(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    x_pad = same_pad_1d(x, window)
    return -F.max_pool1d(-x_pad, kernel_size=window, stride=1, padding=0)


def rolling_median(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    x_pad = same_pad_1d(x, window)
    batch_size, channels, seq_len = x.shape
    windows = F.unfold(
        x_pad.unsqueeze(2), kernel_size=(1, window), padding=0, stride=(1, 1)
    )
    windows = windows.view(batch_size, channels, window, seq_len)
    return torch.median(windows, dim=2).values


def rolling_range(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    return rolling_max(x, window=window) - rolling_min(x, window=window)


def rolling_position(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    rng = rolling_range(x, window=window)
    return (x - rolling_min(x, window=window)) / (rng + 1e-8)


def detrended(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    return x - sma(x, window=window)


def rolling_shannon_entropy(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    x_pad = same_pad_1d(x, window)
    batch_size, channels, seq_len = x.shape
    x_unfold = x_pad.unsqueeze(2)
    windows = F.unfold(x_unfold, kernel_size=(1, window), padding=0, stride=(1, 1))
    windows = windows.view(batch_size, channels, window, seq_len)
    windows = windows.permute(0, 1, 3, 2)

    probs = torch.softmax(windows, dim=-1)
    probs = torch.clamp(probs, min=1e-8)
    entropy = -(probs * _safe_log(probs)).sum(dim=-1)
    return entropy


def rolling_linear_regression_slope(x: torch.Tensor, window: int = DEFAULT_WINDOW):
    seq = torch.arange(x.size(-1), device=x.device, dtype=x.dtype).view(1, 1, -1)

    if window <= 1:
        return torch.zeros_like(x)

    window = min(window, x.size(-1))
    window_sizes = (
        torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
        .clamp(max=window)
        .view(1, 1, -1)
    )
    sum_x = window_sizes * (window_sizes - 1) / 2.0
    sum_x2 = window_sizes * (window_sizes - 1) * (2 * window_sizes - 1) / 6.0

    sum_y = torch.cumsum(x, dim=-1)
    sum_xy = torch.cumsum(x * seq, dim=-1)

    prefix_y = torch.zeros_like(sum_y)
    prefix_xy = torch.zeros_like(sum_xy)
    if window > 1:
        prefix_y[..., window:] = sum_y[..., :-window]
        prefix_xy[..., window:] = sum_xy[..., :-window]

    sum_y_window = sum_y - prefix_y
    sum_xy_window = sum_xy - prefix_xy

    denom = window_sizes * sum_x2 - sum_x.square()
    numer = window_sizes * sum_xy_window - sum_x * sum_y_window

    return torch.where(
        (window_sizes >= 2) & (denom > 0),
        numer / denom,
        torch.zeros_like(numer),
    )


def velocity(x: torch.Tensor):
    return torch.cat([torch.zeros_like(x[:, :, :1]), torch.diff(x, dim=2)], dim=2)


def acceleration(x: torch.Tensor):
    return torch.cat(
        [torch.zeros_like(x[:, :, :1]), torch.diff(velocity(x), dim=2)], dim=2
    )


def log_diff(x: torch.Tensor):
    ratios = x[:, :, 1:] / torch.clamp(x[:, :, :-1], min=1e-8)
    safe_ratios = torch.clamp(ratios, min=1e-8)
    return torch.cat(
        [torch.zeros_like(x[:, :, :1]), _safe_log(safe_ratios)],
        dim=2,
    )


def rfft(x: torch.Tensor):
    return torch.fft.fft(x, axis=2).real


def ifft(x: torch.Tensor):
    return torch.fft.ifft(x, axis=2).imag


def kalman_filter(
    x: torch.Tensor, process_noise: float = 1e-2, measurement_noise: float = 1e-1
):
    filtered = torch.zeros_like(x)
    state = x[..., :1]
    covariance = torch.ones_like(state)

    for i in range(x.size(-1)):
        measurement = x[..., i : i + 1]
        prediction = state
        prediction_covariance = covariance + process_noise

        kalman_gain = prediction_covariance / (
            prediction_covariance + measurement_noise
        )
        state = prediction + kalman_gain * (measurement - prediction)
        covariance = (1 - kalman_gain) * prediction_covariance
        filtered[..., i : i + 1] = state

    return filtered


ALL_TRANSFORMS = [
    identity,
    sma,
    # rolling_std,
    # rolling_zscore,
    rolling_max,
    rolling_min,
    rolling_median,
    rolling_range,
    rolling_position,
    detrended,
    # rolling_shannon_entropy,
    # rolling_linear_regression_slope,
    velocity,
    acceleration,
    log_diff,
    rfft,
    ifft,
    # kalman_filter,
]


class SequentialTransformLayer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outputs: int = 1,
        seq_len: int | None = None,
        transforms: list[callable] = ALL_TRANSFORMS,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.transforms = transforms
        self.weights = nn.Parameter(
            torch.randn(1, num_outputs, num_features * len(transforms), 1)
        )
        self.bias = nn.Parameter(torch.zeros(1, num_outputs, 1))
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor):
        # force fp32 in autocast
        x = x.float()
        # (B, C, W)

        transformed = [f(x) for f in self.transforms] # [M * (B, C, W)]
        z = torch.cat(transformed, dim=1) # (B, C*M, W)
        
        z = torch.matmul(self.weights.squeeze(0).squeeze(-1), z) # (O, C*M) @ (B, C*M, W) -> (B, O, W)
        # (B, O, W)

        # equivalent to:
        # z = z.unsqueeze(1)
        # # (B, 1, C*M, W)
        # z = self.weights * z
        # # (1, O, C*M, 1) * (B, 1, C*M, W) -> (B, O, C*M, W)
        # z = z.sum(dim=2)
        # # (B, O, W)  -- weighted features summed over C*M

        z = z + self.bias
        # (B, O, W)

        z = self.norm(z)
        # (B, O, W)

        return z


if __name__ == "__main__":
    import pandas as pd

    from trading.brains.allocation.datasets import PriceDataset
    from trading.model_tools import ta_transform, fetch_data

    sample_data = fetch_data(
        symbols=["SOL-USDT", "BTC-USDT"],
        days=30,
        interval="5m",
        age_days=0,
        data_source="binance",
        cache_expiry_hours=24,
        verbose=True,
    )

    transformed = ta_transform(sample_data, add_ticker="BTC-USDT")
    dataset = PriceDataset(sample_data, add_ticker="BTC-USDT", seq_len=16)
    batch = dataset[0][0].unsqueeze(0)

    layer = SequentialTransformLayer(
        num_features=batch.shape[1], num_outputs=3, transforms=ALL_TRANSFORMS
    )
    output = layer(batch)

    print("transformed_features", transformed.shape)
    print("dataset_shape", batch.shape)
    print("layer_output_shape", output.shape)
    print(output)
