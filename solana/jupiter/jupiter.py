import os
import requests
import base58
import base64

from solders.solders import Keypair, VersionedTransaction

from typing import Dict, List

def get_token_info(contract_address: str) -> Dict:
    url = f"https://lite-api.jup.ag/tokens/v1/token/{contract_address}"
    response = requests.get(url)
    return response.json()

class JupiterWalletHandler:
    def __init__(self, private_key: str):
        self.private_key = private_key
        self.wallet = Keypair.from_bytes(base58.b58decode(private_key))
    
    def get_order(self, input_mint: str, output_mint: str, input_amount: int) -> Dict:
        """
        Get an order from Jupiter with ultra API. Automatically scales UI to raw.
        """

        input_mint_info = get_token_info(input_mint)
        input_decimals = input_mint_info['decimals']

        raw_input_amount = int(input_amount * (10 ** input_decimals))

        order_params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": raw_input_amount,
            "taker": str(self.wallet.pubkey()),
        }

        response = requests.get("https://lite-api.jup.ag/ultra/v1/order", params=order_params)

        if response.status_code != 200:
            print(f"Error fetching order: {response.json()}")
            return None
        else:
            return response.json()

if __name__ == "__main__":
    wallet = JupiterWalletHandler(input("Enter private key: "))
    print(wallet.get_order("So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 0.5))