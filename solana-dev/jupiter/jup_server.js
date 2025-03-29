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

async function signTransaction(transactionBase64, walletPrivateKey) {
    try {
        const signerKeypair = Keypair.fromSecretKey(bs58.decode(walletPrivateKey));
        const transaction = VersionedTransaction.deserialize(Buffer.from(transactionBase64, 'base64'));
        transaction.sign([signerKeypair]);
        return Buffer.from(transaction.serialize()).toString('base64');
    } catch (error) {
        console.error('Error signing transaction:', error);
        throw error;
    }
}

async function getOrder(inputMint, outputMint, amount, walletAddress) {
    const inputMintInfo = await (
        await fetch(`https://api.jup.ag/tokens/v1/token/${inputMint}`)
    ).json();
    const inputDecimal = inputMintInfo.decimals;

    const outputMintInfo = await (
        await fetch(`https://api.jup.ag/tokens/v1/token/${outputMint}`)
    ).json();
    const outputDecimal = outputMintInfo.decimals;

    const url = `https://api.jup.ag/ultra/v1/order?inputMint=${inputMint}&outputMint=${outputMint}&amount=${amount*Math.pow(10,inputMintInfo.decimals)}&taker=${walletAddress}`;
    const response = await fetch(url);
    const responseData = await response.json();

    console.log(`Swapping: ${Number(responseData.inAmount)/ Math.pow(10, inputDecimal)} ${inputMintInfo.symbol} to ${Number(responseData.outAmount)/ Math.pow(10, outputDecimal)} ${outputMintInfo.symbol}`);
    return responseData;
}

async function executeSwap(inputMint, outputMint, amount, walletPrivateKey, walletAddress) {
    try {
        const signerKeypair = Keypair.fromSecretKey(bs58.decode(walletPrivateKey));
        console.log('Using signer public key:', signerKeypair.publicKey.toString());
        
        const orderResponse = await getOrder(inputMint, outputMint, amount, walletAddress);
        const signedTransactionBase64 = await signTransaction(orderResponse.transaction, walletPrivateKey);
        
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

        const result = await executeSwap(inputMint, outputMint, amount, walletPrivateKey, walletAddress);
        res.json(result);
    } catch (error) {
        console.error('Swap API error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error'
        });
    }
});

app.get('/', (req, res) => {
    res.json({ status: 'healthy' });
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});