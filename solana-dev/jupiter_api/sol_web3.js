import { Transaction, Message, PublicKey, Keypair, VersionedTransaction } from '@solana/web3.js';
import bs58 from 'bs58';
import fetch from 'node-fetch';
import promptSync from 'prompt-sync';


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
        // get decimals here
        const tokenInfoResponse = await (
            await fetch('https://api.jup.ag/tokens/v1/token/JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN')
        ).json();

        const decimals = tokenInfoResponse.decimals
        console.log(decimals)

        const url = `https://api.jup.ag/ultra/v1/order?inputMint=${inputMint}&outputMint=${outputMint}&amount=${amount*Math.pow(10,decimals)}&taker=${this.walletAddress}`;
        const response = await fetch(url);
        return await response.json();
    }

    async executeSwap(inputMint, outputMint, amount) {
        try {
            console.log('Using signer public key:', this.signerKeypair.publicKey.toString());
            
            console.log('Fetching transaction from Jupiter...');
            const orderResponse = await this.getOrder(inputMint, outputMint, amount);
            console.log('Order response:', orderResponse);

            console.log('Signing transaction...');
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

            console.log('Transaction Sent!');

            if (executeResponse.signature) {
                console.log('Swap successful:', JSON.stringify(executeResponse, null, 2));
                console.log(`https://solscan.io/tx/${executeResponse.signature}`);
                return executeResponse;
            } else {
                console.error('Swap failed:', JSON.stringify(executeResponse, null, 2));
                throw new Error('Swap failed');
            }
        } catch (error) {
            console.error('Failed to process transaction:', error);
            throw error;
        }
    }
}

// sol So11111111111111111111111111111111111111112
// usdc EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v

const main = async () => {
    const prompt = promptSync();
    const WALLET_INFO = prompt("WALLET INFO (private,public): ").split(",")

    const WALLET_PRIVATE_KEY = WALLET_INFO[0];
    const WALLET_ADDRESS = WALLET_INFO[1];

    const jupiterSwap = new JupiterSwap(WALLET_PRIVATE_KEY, WALLET_ADDRESS);
    
    try {
        await jupiterSwap.executeSwap("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "So11111111111111111111111111111111111111112", 5);
    } catch (error) {
        console.error('Main execution failed:', error);
    }
};

main();