# Jupiter Swap API

A REST API server for executing token swaps on Jupiter DEX.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file (optional):
```
PORT=3000
```

3. Start the server:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

## API Endpoints

### POST /api/swap

Execute a token swap on Jupiter DEX.

**Request Body:**
```json
{
    "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  // USDC
    "outputMint": "So11111111111111111111111111111111111111112",  // SOL
    "amount": 5,  // Amount to swap
    "walletPrivateKey": "your_private_key",
    "walletAddress": "your_wallet_address"
}
```

**Response:**
```json
{
    "success": true,
    "signature": "transaction_signature",
    "transactionUrl": "https://solscan.io/tx/...",
    // Additional Jupiter API response data
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy"
}
```

## Security Notes

- The API includes rate limiting (100 requests per 15 minutes per IP)
- Private keys should be handled securely and never stored on the server
- Consider implementing additional authentication for production use

## Common Token Mints

- SOL: `So11111111111111111111111111111111111111112`
- USDC: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` 