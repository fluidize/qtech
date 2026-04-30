#!/bin/bash
source .venv/bin/activate

install_dependencies() {
    echo "Installing dependencies..."

    pip install numpy pandas matplotlib seaborn plotly
    
    pip install scikit-learn scipy hmmlearn optuna optunahub lightgbm
    
    pip install yfinance requests aiohttp websockets
    
    pip install beautifulsoup4 selenium
    
    pip install rich tqdm textual
    
    pip install discord-webhook
    
    echo "All dependencies installed!"
}

install_dependencies
