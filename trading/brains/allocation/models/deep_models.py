import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import SequentialTransformLayer

DROPOUT = 1 / 8


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


class FeatureGate(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.gate_weights = nn.Parameter(
            torch.ones(num_features) * 4
        )  # init to sigmoid ~1

    def forward(self, x):
        # x is (B, C, W)
        gate = torch.sigmoid(self.gate_weights)  # (C,)

        gate = gate.view(1, -1, 1)  # (1, C, 1)

        return x * gate


class AttentionPooling(nn.Module):
    def __init__(self, input_channels=32, reduction_dim=32):
        super().__init__()

        self.reduction_dim = reduction_dim
        self.channel_reducer = nn.Conv1d(input_channels, self.reduction_dim, 1)
        self.attention = nn.Sequential(
            nn.Conv1d(self.reduction_dim, self.reduction_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(self.reduction_dim, 1, kernel_size=1),
        )

        self.output_dim = self.reduction_dim

    def forward(self, x):
        x = self.channel_reducer(x)  # (B, reduction_dim, W)
        scores = self.attention(x).squeeze(1)  # (B, W)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)  # (B, 1, W)
        pooled = torch.sum(x * weights, dim=-1)  # (B, reduction_dim)
        return pooled


class ConvolutionEncoder(nn.Module):
    def __init__(
        self,
        channels,
        width,
        hidden_channel_size=32,
        hidden_linear_size=32,
        out_size=32,
    ):
        super().__init__()

        self.channels = channels
        self.width = width
        self.hidden_channel_size = hidden_channel_size
        self.hidden_linear_size = hidden_linear_size
        self.out_size = out_size

        # Feature gate for input channels
        self.feature_gate = FeatureGate(channels)

        self.msp = AttentionPooling(
            input_channels=hidden_channel_size, reduction_dim=hidden_channel_size
        )

        self.convolver = nn.Sequential(
            nn.Conv1d(channels, hidden_channel_size, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channel_size,
                hidden_channel_size,
                kernel_size=3,
                padding=1,
                groups=hidden_channel_size,
            ),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size, hidden_channel_size * 2, kernel_size=1),
            nn.GroupNorm(8, hidden_channel_size * 2),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channel_size * 2,
                hidden_channel_size * 2,
                kernel_size=3,
                padding=1,
                groups=hidden_channel_size * 2,
            ),
            nn.GroupNorm(8, hidden_channel_size * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size * 2, hidden_channel_size, kernel_size=1),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channel_size, hidden_channel_size, kernel_size=3, padding=1
            ),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            # nn.Linear(hidden_channel_size * self.width, hidden_linear_size),
            nn.Linear(self.msp.output_dim, hidden_linear_size),
            nn.LayerNorm(hidden_linear_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_linear_size, hidden_linear_size),
            nn.LayerNorm(hidden_linear_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_linear_size, hidden_linear_size),
            nn.LayerNorm(hidden_linear_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_linear_size, self.out_size),
        )

    def forward(self, x):
        # Apply feature gate
        x = self.feature_gate(x)

        x = self.convolver(x)
        x = self.msp(x)
        embedding = self.encoder(x)
        return embedding


class LSTMEncoder(nn.Module):
    def __init__(self, channels, width, hidden_size=32, out_size=32):
        super().__init__()

        self.channels = channels
        self.width = width
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.feature_gate = FeatureGate(channels)

        self.lstm = nn.LSTM(channels, hidden_size, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * width, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, out_size),
        )

    def forward(self, x):
        x = self.feature_gate(x)

        x = x.transpose(1, 2)  # (B, W, C)
        x, _ = self.lstm(x)
        x = x.flatten(1, 2)  # (B, hidden_size * W)
        embedding = self.encoder(x)
        return embedding


class Booster(nn.Module):
    def __init__(self, lstm_embedding_size, conv_embedding_size, out_size=1):
        super().__init__()

        self.lstm_embedding_size = lstm_embedding_size
        self.conv_embedding_size = conv_embedding_size
        self.out_size = out_size

        self.encoder = nn.Sequential(
            nn.Linear(lstm_embedding_size + conv_embedding_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, out_size),
        )

    def forward(self, conv_embedding, lstm_embedding):
        x = torch.cat([conv_embedding, lstm_embedding], dim=1)
        output = self.encoder(x)
        return output


class AllocatorPolicy(nn.Module):
    def __init__(
        self,

        in_channels, #num features for STL
        in_width, #lookback len for STL + conv + lstm
        out_channels=2, #STL output features

        conv_hidden_channel_size=32,
        conv_hidden_linear_size=32,
        conv_out_size=32,
        lstm_hidden_size=32,
        lstm_out_size=32,
    ):

        super().__init__()

        self.stu = nn.Sequential(
            SequentialTransformLayer(
                num_features=in_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
            SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
                SequentialTransformLayer(
                num_features=out_channels,
                num_outputs=out_channels,
            ),
        )

        self.conv = ConvolutionEncoder(
            out_channels,
            in_width,
            hidden_channel_size=conv_hidden_channel_size,
            hidden_linear_size=conv_hidden_linear_size,
            out_size=conv_out_size,
        )
        self.lstm = LSTMEncoder(
            out_channels, in_width, hidden_size=lstm_hidden_size, out_size=lstm_out_size
        )

        self.mean_booster = Booster(lstm_out_size, conv_out_size, out_size=1)
        self.log_std_booster = Booster(lstm_out_size, conv_out_size, out_size=1)

        self.out_size = self.conv.out_size + self.lstm.out_size

        self.dist_encoder = nn.Sequential(
            nn.Linear(self.out_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.mean_head = nn.Linear(64, 1)
        self.log_std_head = nn.Linear(64, 1)

        self.apply(initialize_weights)

    def get_action(self, obs, epoch=None, total_epochs=None):
        mean, log_std = self.forward(obs)
        if self.training:
            log_std = torch.clamp(log_std, -2, 1)
            std = torch.exp(log_std)

            if epoch is not None and total_epochs is not None:
                # Decay from 1.0 to 0.1 over training
                exploration_scale = 1.0 - 0.9 * (epoch / total_epochs)
                exploration_scale = max(0.1, exploration_scale)
                std = std * exploration_scale

            epsilon = torch.randn_like(mean)
            action = mean + std * epsilon
        else:
            action = mean

        ### tanh squish
        return F.tanh(action).squeeze(-1)

    def forward(self, x):
        x = self.stu(x)

        x1 = self.conv(x)
        x2 = self.lstm(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.dist_encoder(x)

        mean = self.mean_head(x) + self.mean_booster(x1, x2)
        log_std = self.log_std_head(x) + self.log_std_booster(x1, x2)

        return mean, log_std
