import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLR(nn.Module): #multiple linear regression
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogMLR(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        x = torch.log(torch.clamp(x, min=1e-8))
        return self.linear(x)


class ExponentialMLR(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        x = torch.exp(x)
        return self.linear(x)


class PolynomialMLR(nn.Module):
    def __init__(self, num_features, degree=2):
        super().__init__()
        self.num_features = num_features
        self.degree = degree
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        x = torch.pow(x, self.degree)
        return self.linear(x)


class SqrtMLR(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        x = torch.sqrt(x)
        return self.linear(x)


class LogisticUnit(nn.Module):
    def __init__(self, num_features, model_type='linear'):
        super().__init__()
        self.num_features = num_features
        self.model_type = model_type

        if model_type == 'linear':
            self.mlr = MLR(num_features)
        elif model_type == 'log':
            self.mlr = LogMLR(num_features)
        elif model_type == 'exp':
            self.mlr = ExponentialMLR(num_features)
        elif model_type == 'poly':
            self.mlr = PolynomialMLR(num_features)
        elif model_type == 'sqrt':
            self.mlr = SqrtMLR(num_features)

    def forward(self, x):
        x = x.squeeze(-1)  # rm the sequence dimension of 1
        return torch.sigmoid(self.mlr(x))

    def get_action(self, x, epoch=None, total_epochs=None):
        return self.forward(x).squeeze(-1)
