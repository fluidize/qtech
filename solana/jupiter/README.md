# Jupiter Trading API

## Overview
The Jupiter Trading API is designed to facilitate token swaps on the Solana blockchain. It provides a set of functionalities for executing trades, managing wallets, and interacting with the Jupiter API.

## Components

### 1. **`main.py`**
The main entry point of the application, which initializes the `JupiterTrading` class with the user's public and private keys. This class manages trading operations and interacts with the Jupiter API.

### 2. **`jupiter_api.py`**
This module contains the `JupiterAPI` class, which provides methods for:
- **Placing Orders**: Execute token swaps by specifying input and output mints, the amount to trade, and wallet credentials.
- **Fetching Wallet Information**: Retrieve balances of tokens in a specified wallet.
- **Getting Token Prices**: Fetch current prices of specified tokens.
- **Calculating Wallet Value**: Calculate the total value of a wallet based on current token prices.

### 3. **`jupiter_server.js`**
This JavaScript file runs a Node.js server that interacts with the Solana blockchain. Key functionalities include:
- **Signing Transactions**: The `signTransaction` function signs transactions using the wallet's private key.
- **Executing Swaps**: The `executeSwap` function handles the logic for swapping tokens, including fetching order details and executing the transaction on the blockchain.
- **API Endpoints**: Exposes endpoints for placing swaps and checking the server's health.

### 4. **Token Class (`token`)**
The `token` class in `jupiter_api.py` stores constant values for commonly used tokens on the Solana blockchain, such as:
- **SOL**: The native token of the Solana blockchain.
- **WBTC**: Wrapped Bitcoin.
- **USDC**: A stablecoin pegged to the US dollar.