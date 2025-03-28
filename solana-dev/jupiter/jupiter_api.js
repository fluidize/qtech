import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { Transaction, Message, PublicKey, Keypair, VersionedTransaction } from '@solana/web3.js';
import bs58 from 'bs58';
import fetch from 'node-fetch';

const app = express();

const port = process.argv[2] || process.env.PORT || 8080;

if (isNaN(port) || port < 0 || port > 65535) {
    console.error('Invalid port number. Please provide a number between 0 and 65535');
    process.exit(1);
}

// Middleware
app.use(express.json());
app.use(cors());

class JupiterSwap {
    constructor(walletPrivateKey, walletAddress) {
        this.walletPrivateKey = walletPrivateKey;
        this.walletAddress = walletAddress;
        this.signerKeypair = Keypair.fromSecretKey(bs58.decode(walletPrivateKey));
    }

    async signTransaction(transactionBase64) {
        try {
            const transaction = VersionedTransaction.deserialize(Buffer.from(transactionBase64, 'base64'));
            transaction.sign([this.signerKeypair]);
            return Buffer.from(transaction.serialize()).toString('base64');
        } catch (error) {
            console.error('Error signing transaction:', error);
            throw error;
        }
    }

    async getOrder(inputMint, outputMint, amount) {
        const tokenInfoResponse = await (
            await fetch(`https://api.jup.ag/tokens/v1/token/${inputMint}`)
        ).json();

        const decimals = tokenInfoResponse.decimals;
        const url = `https://api.jup.ag/ultra/v1/order?inputMint=${inputMint}&outputMint=${outputMint}&amount=${amount*Math.pow(10,decimals)}&taker=${this.walletAddress}`;
        const response = await fetch(url);
        return await response.json();
    }

    async executeSwap(inputMint, outputMint, amount) {
        try {
            console.log('Using signer public key:', this.signerKeypair.publicKey.toString());
            
            const orderResponse = await this.getOrder(inputMint, outputMint, amount);
            const signedTransactionBase64 = await this.signTransaction(orderResponse.transaction);
            
            const executeResponse = await (
                await fetch('https://api.jup.ag/ultra/v1/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        signedTransaction: signedTransactionBase64,
                        requestId: orderResponse.requestId,
                    }),
                })
            ).json();

            if (executeResponse.signature) {
                return {
                    success: true,
                    signature: executeResponse.signature,
                    transactionUrl: `https://solscan.io/tx/${executeResponse.signature}`,
                    ...executeResponse
                };
            } else {
                throw new Error('Swap failed');
            }
        } catch (error) {
            console.error('Failed to process transaction:', error);
            throw error;
        }
    }
}

app.post('/swap', async (req, res) => {
    try {
        const { inputMint, outputMint, amount, walletAddress, walletPrivateKey } = req.body;

        if (!inputMint || !outputMint || !amount || !walletPrivateKey || !walletAddress) {
            return res.status(400).json({
                success: false,
                error: 'Missing required parameters'
            });
        }

        if (isNaN(amount) || amount <= 0) {
            return res.status(400).json({
                success: false,
                error: 'Amount must be a positive number'
            });
        }

        const jupiterSwap = new JupiterSwap(walletPrivateKey, walletAddress);
        const result = await jupiterSwap.executeSwap(inputMint, outputMint, amount);
        
        res.json(result);
    } catch (error) {
        console.error('Swap API error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error'
        });
    }
});

app.get('/wallet/:wallet_addr', async (req, res) => {
    const { wallet_addr } = req.params;
    res.send(`${wallet_addr}`)
});

app.get('/', (req, res) => {
    res.json({ status: 'healthy' });
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
}); 