import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from trading.model_tools import ta_transform


class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, add_ticker: str, seq_len: int = 10):
        self.seq_len = seq_len
        self.data = data

        self.X = ta_transform(self.data, add_ticker=add_ticker)
        self.valid_indices = self.X.index[self.seq_len - 1 :]

        feat_arr = self.X.values.astype(np.float32)

        sequences = []
        for i in range(len(feat_arr) - self.seq_len + 1):
            sequences.append(feat_arr[i : i + self.seq_len])

        self.X = torch.from_numpy(np.array(sequences))
        # Transpose to (num_sequences, num_features, seq_len)
        self.X = self.X.transpose(1, 2)

        # (num_sequences, num_features, seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], idx
