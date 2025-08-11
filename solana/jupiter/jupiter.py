import os
import requests
import base58
import base64
from solders.solders import Keypair, VersionedTransaction

from typing import Dict, List, Optional, Tuple, Union
from rich import print
from time import sleep

class Token:
    SOL = "So11111111111111111111111111111111111111112"
    USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    wBTC = "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh"
    cbBTC = "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij"
    wETH = "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"
    JitoSOL = "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn"

def get_token_info(contract_address: str) -> Dict:
    url = f"https://lite-api.jup.ag/tokens/v1/token/{contract_address}"
    response = requests.get(url)
    return response.json()

def get_usd_price(contract_address: str) -> Optional[float]:
    try:
        url = f"https://lite-api.jup.ag/price/v2?ids={contract_address}"
        response = requests.get(url)
        return float(response.json()['data'][contract_address]['price'])
    except Exception as e:
        print(f"Error getting USD price: {e}")
        return None

class JupiterWalletHandler:
    def __init__(self, private_key: str):
        self.private_key = private_key
        self.wallet = Keypair.from_bytes(base58.b58decode(private_key))
        self.wallet_address = self.wallet.pubkey()
    
    def get_order(self, input_mint: str, output_mint: str, input_amount: float, retry: bool = True, retry_limit: int = 5) -> Optional[Tuple[float, float, float, float, float, float, str]]:
        """
        Get an order from Jupiter with ultra API. Automatically scales UI to raw.
        Returns a tuple of (in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx)
        """
        try:
            for i in range(retry_limit if retry else 1):
                input_mint_info = get_token_info(input_mint)
                input_decimals = input_mint_info['decimals']

                raw_input_amount = int(input_amount * (10 ** input_decimals))

                order_params = {
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": raw_input_amount,
                    "taker": str(self.wallet_address),
                }

                response = requests.get("https://lite-api.jup.ag/ultra/v1/order", params=order_params)
                response_json = response.json()
                
                if response.status_code != 200:
                    print(f"Error fetching order {response.status_code} : {response.json()}")
                    sleep(0.5)
                    continue
                else:
                    in_usd = float(response_json['inUsdValue'])
                    out_usd = float(response_json['outUsdValue'])
                    slippage_bps = float(response_json['slippageBps'])
                    fee_bps = float(response_json['feeBps'])
                    price_impact_pct = float(response_json['priceImpactPct']) #positive pi is good
                    price_impact_usd = float(response_json['priceImpact'])
                    unsigned_tx = str(response_json['transaction'])
                    request_id = response_json['requestId']
                    
                    return in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx, request_id
        except Exception as e:
            print(f"Error in get_order: {e}")
            return None
        
    def execute_order(self, unsigned_tx: str, request_id: str):
        tx_bytes = base64.b64decode(unsigned_tx)
        raw_tx = VersionedTransaction.from_bytes(tx_bytes)

        account_keys = raw_tx.message.account_keys
        wallet_index = account_keys.index(self.wallet_address)

        signers = list(raw_tx.signatures)
        signers[wallet_index] = self.wallet

        signed_transaction = VersionedTransaction(raw_tx.message, signers)
        serialized_signed_transaction = base64.b64encode(bytes(signed_transaction)).decode("utf-8")

        execute_request = {
            "signedTransaction": serialized_signed_transaction,
            "requestId": request_id,
        }

        execute_response = requests.post(
            "https://lite-api.jup.ag/ultra/v1/execute", json=execute_request
        )

        if execute_response.status_code == 200:
            error_data = execute_response.json()
            signature = error_data["signature"]
            return signature
        else:
            if error_data["status"] != "Success":
                error_code = error_data["code"]
                error_message = error_data["error"]

                print(f"Transaction failed! Signature: {signature}")
                print(f"Custom Program Error Code: {error_code}")
                print(f"Message: {error_message}")
                print(f"View transaction on Solscan: https://solscan.io/tx/{signature}")
                return None
            
    def order_and_execute(self, input_mint: str, output_mint: str, input_amount: float, retry: bool = True, retry_limit: int = 5) -> Optional[Tuple[float, float, float, float, float, float, str]]:
        in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx, request_id = self.get_order(input_mint, output_mint, input_amount, retry, retry_limit)
        signature = self.execute_order(unsigned_tx, request_id)
        return signature

    def get_wallet_balances(self, wallet_address):
        response = requests.get(f"https://lite-api.jup.ag/ultra/v1/balances/{wallet_address}")
        return response.json()

    def get_wallet_token_amount(self, token_address):
        if token_address == "So11111111111111111111111111111111111111112":
            token_address = "SOL"
        wallet_info = self.get_wallet_balances(self.wallet_address)
        return wallet_info[token_address]["uiAmount"]
            

if __name__ == "__main__":
    jupiter = JupiterWalletHandler("")