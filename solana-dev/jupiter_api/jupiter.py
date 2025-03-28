import requests
import json
import time
from rich import print

wallet_addr = "3X2LFoTQecbpqCR7G5tL1kczqBKurjKPHhKSZrJ4wgWc"

inputmnt = "So11111111111111111111111111111111111111112"
outputmnt = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

amount = 100000000

class JupiterAPI:
    def __init__(self, wallet_address):
        self.base_url = "https://api.jup.ag/ultra/v1/"
        self.wallet_address = wallet_address

    def get_order(self, input_mint: str, output_mint: str, amount: int) -> dict:
        extension = f"order?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&taker={wallet_addr}"
        url = f'{self.base_url}{extension}'
        response = requests.get(url)
        return response.json()

    def execute_order(self, order_data: dict) -> dict:
        transaction_base64 = order_data['transaction']


        url = f'{self.base_url}/execute'
        response = requests.post(url, json=order_data)
        return response.json()

jupiter = JupiterAPI(wallet_addr)
jupiter.execute_order(jupiter.get_order(inputmnt, outputmnt, 100000000))