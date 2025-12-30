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

def get_token_info(contract_address: str, api_key: Optional[str] = None) -> Dict:
    url = f"https://api.jup.ag/ultra/v1/search?query={contract_address}"
    headers = {'x-api-key' : api_key}
    response = requests.get(url, headers=headers)
    return response.json()[0]

def get_usd_price(contract_address: str, api_key: Optional[str] = None) -> Optional[float]:
    try:
        url = f"https://api.jup.ag/price/v2?ids={contract_address}"
        headers = {'x-api-key' : api_key}
        response = requests.get(url, headers=headers)
        return float(response.json()['data'][contract_address]['price'])
    except Exception as e:
        print(f"Error getting USD price: {e}")
        return None

class JupiterWalletHandler:
    def __init__(self, private_key: str, api_key: Optional[str] = None):
        self.private_key = private_key
        self.api_key = api_key
        self.wallet = Keypair.from_bytes(base58.b58decode(private_key))
        self.wallet_address = self.wallet.pubkey()
    
    def get_order(self, input_mint: str, output_mint: str, input_amount_decimals: float, retry: bool = True, retry_limit: int = 5) -> Optional[Tuple[float, float, float, float, float, float, str]]:
        """
        Input amount is in decimals of the input mint
        Get an order from Jupiter with ultra API. Automatically scales UI to raw.
        Returns a tuple of (in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx)
        """
        try:
            headers = {'x-api-key': self.api_key}
            
            for i in range(retry_limit if retry else 1):
                input_mint_info = get_token_info(input_mint, self.api_key)
                input_decimals = input_mint_info['decimals']
                output_mint_info = get_token_info(output_mint, self.api_key)
                output_decimals = output_mint_info['decimals']

                raw_input_amount = int(input_amount_decimals * (10 ** input_decimals))

                order_params = {
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": raw_input_amount,
                    "taker": str(self.wallet_address),
                }

                response = requests.get("https://api.jup.ag/ultra/v1/order", params=order_params, headers=headers)
                response_json = response.json()
                
                if response.status_code != 200:
                    print(f"Error fetching order {response.status_code} : {response.json()}")
                    sleep(0.5)
                    continue
                else:
                    response_info = {
                        "input_mint": str(response_json['inputMint']),
                        "output_mint": str(response_json['outputMint']),
                        "in_amount": float(response_json['inAmount']),
                        "out_amount": float(response_json['outAmount']),
                        "in_amount_decimals": float(response_json['inAmount']) / (10 ** input_decimals),
                        "out_amount_decimals": float(response_json['outAmount']) / (10 ** output_decimals),
                        "in_usd": float(response_json['inUsdValue']),
                        "out_usd": float(response_json['outUsdValue']),
                        "slippage_bps": float(response_json['slippageBps']),
                        "fee_bps": float(response_json['feeBps']),
                        "price_impact_pct": float(response_json['priceImpactPct']),
                        "price_impact_usd": float(response_json['priceImpact']),
                        "unsigned_tx": str(response_json['transaction']),
                        "request_id": str(response_json['requestId'])
                    }
                    
                    return response_info
        except Exception as e:
            print(f"Error in get_order: {e}")
            raise e
        
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

        headers = {'x-api-key': self.api_key}
        
        execute_response = requests.post(
            "https://api.jup.ag/ultra/v1/execute", json=execute_request, headers=headers
        )

        if execute_response.status_code == 200:
            response_data = execute_response.json()
            signature = response_data["signature"]
            return signature
        else:
            error_data = execute_response.json()
            if error_data.get("status") != "Success":
                error_code = error_data.get("code", "Unknown")
                error_message = error_data.get("error", "Unknown error")
                signature = error_data.get("signature", "N/A")

                print(f"Transaction failed! Signature: {signature}")
                print(f"Custom Program Error Code: {error_code}")
                print(f"Message: {error_message}")
                print(f"View transaction on Solscan: https://solscan.io/tx/{signature}")
            return None
            
    def order_and_execute(self, input_mint: str, output_mint: str, input_amount_decimals: float, retry: bool = True, retry_limit: int = 5) -> Optional[Tuple[float, float, float, float, float, float, str]]:
        """
        Returns order response info and signature
        """
        response_info = self.get_order(input_mint, output_mint, input_amount_decimals, retry, retry_limit)
        unsigned_tx = response_info['unsigned_tx']
        request_id = response_info['request_id']
        signature = self.execute_order(unsigned_tx, request_id)
        return response_info, signature

    def get_wallet_balances(self, wallet_address):
        headers = {'x-api-key': self.api_key}
        response = requests.get(f"https://api.jup.ag/ultra/v1/balances/{wallet_address}", headers=headers)
        return response.json()

    def get_wallet_token_amount(self, token_address):
        if token_address == "So11111111111111111111111111111111111111112":
            token_address = "SOL"
        wallet_info = self.get_wallet_balances(self.wallet_address)
        return wallet_info[token_address]["uiAmount"]
            

if __name__ == "__main__":
    import time

    start = time.time()
    jupiter = JupiterWalletHandler("2BmZhw6gq2VyyvQNhzbXSPp1riXVDQqfiBNPeALf54gsZ9Wh4bLzQrzbysRUgxZVmi862VcXTwFvcAnfC1KYwWsz", "48c75b32-8a38-4c69-b425-24953191bcaa")
    print(jupiter.get_order(Token.USDC, Token.SOL, 100.0, retry=True))
    end = time.time()
    print(end-start)
