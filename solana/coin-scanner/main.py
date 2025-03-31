import os
import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import SQUARE
import datetime

rich_console = Console()

class CoinData:
    def __init__(self):
        options = Options()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")  # run in background
        options.add_argument("--log-level=3")  # suppress unnecessary logs
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # initialize WebDriver once for the class
        rich_console.print("[bold green]Initializing selenium webscraper...[/bold green]", end="")
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', { get: () => false });")

    def close_driver(self):
        self.driver.quit()
    
    def _time_ago(self, dt):
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - dt
        
        # Convert to total seconds
        total_seconds = delta.total_seconds()
        
        # Handle different time units
        if total_seconds < 60:
            return f"{int(total_seconds)}s ago"
        elif total_seconds < 3600:
            return f"{int(total_seconds // 60)}m ago"
        elif total_seconds < 86400:
            return f"{int(total_seconds // 3600)}h ago"
        elif total_seconds < 2592000:
            return f"{int(total_seconds // 86400)}d ago"
        elif total_seconds < 31536000:
            return f"{int(total_seconds // 2592000)}mo ago"
        else:
            return f"{int(total_seconds // 31536000)}y ago"

    def _webscrape(self, token_address):
        rich_console.print(f"[cyan]Scanning {token_address}...[/cyan]", end="\r")
        self.driver.get(f"https://jup.ag/trenches/{token_address}")
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, ".relative.inline-flex.items-center.rounded-sm.text-sm.font-medium.tabular-nums.leading-none")
            elements = [element.text for element in elements] #mkt cap, 24hvol, fully diluted value, liquidity, holders
    
            price = WebDriverWait(self.driver, 2).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".relative.inline-flex.items-center.rounded-sm.text-lg.font-medium.leading-none"))
            ).text.replace("\n", "") #subscript newline
        
            return (price, elements[0], elements[1], elements[2],elements[3],elements[4])
        except:
            return ("NA", "NA", "NA", "NA","NA","NA")


    def _get_new_tokens(self,limit):
        url = "https://api.jup.ag/tokens/v1/new"
        params = {'limit':limit}
        headers = {'Accept': 'application/json'}
        response = requests.request("GET", url, headers=headers, params=params)
        raw = response.json()
        tokens = {}
        #UNUSED DATA
        #"known_markets": token["known_markets"]
        #"metadata_updated_at": datetime.datetime.utcfromtimestamp(token["metadata_updated_at"]).strftime('%Y-%m-%d %H:%M:%S')
        count = 0
        for token in raw:
            rich_console.print(f"[cyan]Scanning {token['mint']}... ({count})[/cyan]", end="\r")
            elements = self._webscrape(token['mint'])
            tokens[token["mint"]] = {
                #FROM API
                "symbol": token["symbol"],
                "name": token["name"],
                "created_at": datetime.datetime.fromtimestamp(int(token["created_at"]), datetime.UTC),
                "mint_authority": "✔" if token.get("mint_authority") else "❌",
                "freeze_authority": "✔" if token.get("freeze_authority") else "❌",
                "decimals": token["decimals"],
                "logo_uri": token.get("logo_uri"),
                #FROM SELENIUM
                "price": elements[0],
                "market_cap": elements[1],
                "daily_volume": elements[2],
                "fdv": elements[3],
                "liquidity": elements[4],
                "holders": elements[5]
            }
            count += 1
        return tokens

    def scan_auto(self, count=10):
        rich_console.print(f"[bold bright_yellow]Scanning {count} tokens.[/bold bright_yellow]")
        table_data = self._get_new_tokens(count)
        # Format data for display - flatten
        table_data = [
            #symbol, name, address, timeago, price, mktcap, 24vol, fdv, liq, mintA, freezeA
            (data['symbol'], data['name'], key, self._time_ago(data['created_at']),data['price'], data['market_cap'], data['daily_volume'], data['fdv'], data['liquidity'], data['holders'], data['mint_authority'], data['freeze_authority'])
            for key, data in table_data.items()
        ]
        return table_data

    def _scan_single_token(self, token_address):
        url = f"https://api.jup.ag/tokens/v1/token/{token_address}"
        headers = {'Accept': 'application/json'}
        response = requests.request("GET", url, headers=headers)
        raw = response.json()
        elements = self._webscrape(token_address)
        token = {
            #FROM API
            "symbol": raw["symbol"],
            "name": raw["name"],
            "created_at": datetime.datetime.fromisoformat(raw["created_at"]),
            "mint_authority": "✔" if raw.get("mint_authority") else "❌",
            "freeze_authority": "✔" if raw.get("freeze_authority") else "❌",
            "decimals": raw["decimals"],
            "logo_uri": raw.get("logo_uri"),
            #FROM SELENIUM
            "price": elements[0],
            "market_cap": elements[1],
            "daily_volume": elements[2],
            "fdv": elements[3],
            "liquidity": elements[4],
            "holders": elements[5]
        }
        return token
    
    def scan(self, token_address):
        table_data = self._scan_single_token(token_address)

        table_data = (table_data['symbol'], table_data['name'], token_address, self._time_ago(table_data['created_at']),table_data['price'], table_data['market_cap'], table_data['daily_volume'], table_data['fdv'], table_data['liquidity'], table_data['holders'], table_data['mint_authority'], table_data['freeze_authority'])
        return table_data

    # def _scan_single_token_jup(self, token_address): #OUTDATED; WEBSITE UPDATED; better things than webscraping
    #     check_url = f"https://jup.ag/tokens/{token_address}"
    #     try:
    #         self.driver.get(check_url)  # Load URL
            
    #         # Wait for elements to load
    #         token_name = WebDriverWait(self.driver, 5).until(
    #             EC.visibility_of_element_located((By.CSS_SELECTOR, ".flex.items-center.gap-1\\.5.text-xl.font-semibold"))
    #         ).text

    #         organic_score = WebDriverWait(self.driver, 5).until(
    #             EC.visibility_of_element_located((By.CSS_SELECTOR, ".flex.items-center.rounded-r.px-2.text-sm.font-medium.text-black"))
    #         ).text

    #         price = WebDriverWait(self.driver, 5).until(
    #             EC.visibility_of_element_located((By.CSS_SELECTOR, ".flex.items-center.text-lg.font-semibold"))
    #         ).text.replace("\n", "").replace("\t", "").replace("    ", "") #subscript does newline for some reason

    #         elements = self.driver.find_elements(By.CSS_SELECTOR, ".flex.items-center.text-sm.font-semibold") #mkt cap , liquidity, 24h vol, holders
    #         elements = [element.text for element in elements if element.text]
    #         if elements[3] == "Instant": #liquidity box is gone
    #             elements[3] = elements[2] #shift
    #             elements[2] = elements[1]
    #             elements[1] = "[bold red]None[/bold red]"

    #         #token name, address, score, price, mkt cap , liquidity, 24h vol, holders
    #         return (token_name, token_address, organic_score, price, elements[0], elements[1], elements[2], elements[3]) #[link=] for hypertext

    #     except Exception as e:
    #     rich_console.print(f"[bold red]Error scanning {token_address}[/bold red]")
    #     print(e)
    #     return None