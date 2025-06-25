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
                    "taker": str(self.wallet.pubkey()),
                }

                response = requests.get("https://lite-api.jup.ag/ultra/v1/order", params=order_params)
                response_json = response.json()
                
                if response.status_code != 200:
                    print(f"Error fetching order {response.status_code} : {response.json()}")
                    sleep(1.5)
                    continue
                else:
                    in_usd = float(response_json['inUsdValue'])
                    out_usd = float(response_json['outUsdValue'])
                    slippage_bps = float(response_json['slippageBps'])
                    fee_bps = float(response_json['feeBps'])
                    price_impact_pct = float(response_json['priceImpactPct']) #positive pi is good
                    price_impact_usd = float(response_json['priceImpact'])
                    unsigned_tx = str(response_json['transaction'])
                    
                    return in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx
        except Exception as e:
            print(f"Error in get_order: {e}")
            return None
            

if __name__ == "__main__":
    jupiter = JupiterWalletHandler("2BmZhw6gq2VyyvQNhzbXSPp1riXVDQqfiBNPeALf54gsZ9Wh4bLzQrzbysRUgxZVmi862VcXTwFvcAnfC1KYwWsz") #placeholder
    for i in range(10):
        result = jupiter.get_order(Token.SOL, Token.USDC, 1.0)
        print(result)
        if result:
            in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx = result
            print(f"In USD: {type(in_usd)} {in_usd}")
            print(f"Out USD: {type(out_usd)} {out_usd}")
            print(f"Slippage BPS: {type(slippage_bps)} {slippage_bps}")
            print(f"Fee BPS: {type(fee_bps)} {fee_bps}")
            print(f"Price Impact PCT: {type(price_impact_pct)} {price_impact_pct}")
            print(f"Price Impact USD: {type(price_impact_usd)} {price_impact_usd}")
            print(f"Unsigned TX: {type(unsigned_tx)} {unsigned_tx}")
        else:
            print("Failed to get order")