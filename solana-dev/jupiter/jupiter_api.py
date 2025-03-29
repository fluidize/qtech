import requests
import subprocess
import asyncio
import time
import os
from rich import print

os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
class token: #store CAs for easy access
    SOL = 'So11111111111111111111111111111111111111112'
    WBTC = '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh'
    USDC = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'

class JupiterAPI:
    def __init__(self, wallet_address, private_key):
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.buy_with = token.USDC

        self._start_api() #use node.js for solana/web3.js implementations; for other functions just use request.get/post

    def _run_node(self, relative_file_path, port):
        print(f'Starting {relative_file_path}')
        process = subprocess.Popen(['node', relative_file_path, str(port)])
        time.sleep(1)
        print(f"[green]NODE STARTED[/green]")

    def _start_api(self):
        api_name = 'jup_server.js'
        port = 8080
        self._run_node(api_name, port) #api on localhost:8080
        self.base_url = f'http://localhost:{port}'

    def place_order(self, input_mint, output_mint, amount, public_key, private_key):
        path = "/swap"
        #amount gets converted to base units in API
        data = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "walletAddress": public_key,
            "walletPrivateKey": private_key
        }
        request = requests.post(self.base_url+path, json=data)
        return request

    def get_wallet(self, wallet_address, ):
        response = requests.get(f"https://api.jup.ag/ultra/v1/balances/{wallet_address}")
        return response.json()
    
    def get_price(self, contract_address):
        if isinstance(contract_address, list):
            contract_address = ",".join(contract_address)

        response = requests.get(f"https://api.jup.ag/price/v2?ids={contract_address}").json() #price for ca in usdc
        output_dict = {}
        for key in response["data"]:
            if response["data"][key] is None:
                output_dict[key] = 0
            else:
                output_dict[key] = float(response["data"][key]["price"])
        return output_dict
    
    def get_wallet_value(self, wallet_address):
        wallet_info = self.get_wallet(wallet_address)
        if 'SOL' in wallet_info:
            wallet_info[token.SOL] = wallet_info.pop('SOL')

        net_worth = 0
        
        for ca in wallet_info:
            decimal_amount = wallet_info[ca]['uiAmount']
            token_worth = list(self.get_price(ca).values())[0]
            net_worth += decimal_amount * token_worth
        
        return net_worth


    
    def buy_max(self, output_mint): # Buy maximum SOL with all USDC
        wallet_info = self.get_wallet(self.wallet_address)
        usdc_token_amount = wallet_info[token.USDC]["uiAmount"]
        
        order = self.place_order(self.buy_with, output_mint, usdc_token_amount, self.wallet_address, self.private_key)
        return order.json()

    def sell_max(self, input_mint, output_mint=token.USDC): # Sell all SOL for USDC
        wallet_info = self.get_wallet(self.wallet_address)
        token_amount = wallet_info[input_mint]["uiAmount"]
        
        order = self.place_order(input_mint, output_mint, token_amount, self.wallet_address, self.private_key)
        return order.json()

    def buy(self, output_mint, amount): # Buy x SOL with USDC
        order = self.place_order(self.buy_with, output_mint, amount, self.wallet_address, self.private_key)
        return order.json()

    def sell(self, input_mint, amount, output_mint=token.USDC): # Sell x SOL for USDC
        order = self.place_order(input_mint, output_mint, amount, self.wallet_address, self.private_key)
        return order.json()

if __name__ == "__main__":
    jupiter = JupiterAPI("a","b")
    print(jupiter.get_wallet_value("GSpGregkksyFFgWaQ48CkSi4bVKwC1uEj6T6Mp6nvgQh"))


# Example usage:
# jupiter.buy_max(token.SOL)  # Buy max SOL with all USDC
# jupiter.sell_max(token.SOL)  # Sell all SOL for USDC
# jupiter.buy(token.SOL, 1.5)  # Buy 1.5 SOL with USDC
# jupiter.sell(token.SOL, 1.5)  # Sell 1.5 SOL for USDC
