import requests
import subprocess
import asyncio
import os
from rich import print

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_node(relative_file_path, port):
    print(f'Starting {relative_file_path}')
    process = subprocess.Popen(['node', relative_file_path, str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    

class JupiterTrading:
    def __init__(self, wallet_address, private):
        self.wallet_address = wallet_address
        self.private = private

        self.tokens = { #default token ca
            'SOL' : 'So11111111111111111111111111111111111111112',
            'USDC' : 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
        }

        self._start_api()

    def _start_api(self):
        api_name = 'jupiter_api.js'
        port = 8080
        run_node(api_name, port) #api on localhost:8080
        self.base_url = f'http://localhost:{port}'
        print("[green]API STARTED[/green]")

    def place_order(self, input_mint, output_mint, amount, public_key, private_key):
        #amount gets converted to units in API
        data = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "walletAddress": public_key,
            "walletPrivateKey": private_key
        }
        requests.post()

public, private = input('Public,Private: ').strip().split(',')
jupiter = JupiterTrading(public, private)