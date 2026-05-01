#!/bin/bash
source .venv/bin/activate

install_dependencies() {
    echo "Installing dependencies..."

    pip install -e . \
        numpy pandas matplotlib seaborn plotly \
        scikit-learn scipy hmmlearn optuna optunahub lightgbm \
        yfinance requests aiohttp websockets \
        beautifulsoup4 selenium \
        rich tqdm textual \
        discord-webhook \
        torch torchvision\
        numba

    echo "All dependencies installed!"
}

install_dependencies
