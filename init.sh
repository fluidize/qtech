#!/bin/bash
source .venv/bin/activate

echo "Installing dependencies..."

pip install -e trading \
    numpy pandas matplotlib seaborn plotly \
    scikit-learn scipy hmmlearn optuna optunahub lightgbm \
    yfinance requests aiohttp websockets \
    beautifulsoup4 selenium \
    rich tqdm textual \
    discord-webhook \
    torch torchvision torchinfo \
    numba base58 solders \
    textual

echo "All dependencies installed!"

