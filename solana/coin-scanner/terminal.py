from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import SQUARE
import sys

import webbrowser

rich_console = Console()

from main import CoinData

class CommandUI:
    def __init__(self):
        self.CoinData = CoinData()
        self.commands = {
            "help": self.show_help,

            "scan": self.scan, #alias
            "sc": self.scan,

            "scan-auto": self.scan_auto,
            "sca": self.scan_auto,

            "rugcheck": self.rugcheck,
            "rc": self.rugcheck,

            "exit": self.exit_app
        }

    def show_help(self):
        help_text = Text("Available Commands:", style="bold green")
        help_text.append("\n  - help - Show this help message")
        help_text.append("\n  - scan-auto <count>- Scan coins latest coins from pump.fun. Default=10 | sca ")
        help_text.append("\n  - scan <token> - Scan a specific token address. | sc")
        help_text.append("\n  - rugcheck <token> - Generate a link to rugcheck.xyz. | rc")
        help_text.append("\n  - exit - Exit the app")
        rich_console.print(Panel(help_text))
    
    def _hypertext(self, text, link, style):
        return f"[{style}][link={link}]{text}[/{style}]"

    def scan_auto(self, count=10):
        table = Table(title="[bold cyan]Memecoins[/bold cyan]", header_style="bold white", box=SQUARE)
        table.add_column("Symbol", justify="center", style="bold white")
        table.add_column("Name", justify="center")
        table.add_column("Address + Rugcheck.xyz Link")
        table.add_column("Age", justify="center")
        table.add_column("Price", justify="center")
        table.add_column("Mkt Cap", justify="center")
        table.add_column("24H Volume", justify="center")
        table.add_column("FDV", justify="center")
        table.add_column("Liquidity", justify="center")
        table.add_column("Holders", justify="center")
        table.add_column("Mint", justify="center")
        table.add_column("Freeze", justify="center")
        table.add_column("TRADE", justify="center")

        table_data = self.CoinData.scan_auto(count)
        #symbol, name, address, timeago, price, mktcap, 24vol, fdv, liq, holders, mintA, freezeA
        for data in table_data:
            table.add_row(data[0],data[1],self._hypertext(data[2],f"https://rugcheck.xyz/tokens/{data[2]}", "bold bright_blue"),data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10], data[11], self._hypertext(data[0], f"https://gmgn.ai/sol/token/{data[2]}", "underline bright_green"))

        rich_console.print(table)

    def scan(self,token):
        # token = Prompt.ask("Enter token address")
        data = self.CoinData.scan(token)

        if not data:
            rich_console.print("[bold red]Failed to retrieve token data.[/bold red]")
            return

        table = Table(title="[bold cyan]Memecoins[/bold cyan]", header_style="bold white", box=SQUARE)
        table.add_row(data[0],data[1],self._hypertext(data[2],f"https://rugcheck.xyz/tokens/{data[2]}", "bold bright_blue"),data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10], data[11], self._hypertext(data[0], f"https://gmgn.ai/sol/token/{data[2]}", "underline bright_green"))

        rich_console.print(table)

    def rugcheck(self, token):
        rich_console.print(self._hypertext(f"https://rugcheck.xyz/tokens/{token}",f"https://rugcheck.xyz/tokens/{token}","bold underlined bright_blue"))
        webbrowser.open(f"https://rugcheck.xyz/tokens/{token}", autoraise=False)

    def _get_score_style(self, score):
        score = float(score)
        if score >= 80.00:
            return "bold bright_green"
        elif score > 0.00:
            return "bold bright_yellow"
        else:
            return "bold bright_red"

    def exit_app(self):
        self.CoinData.close()  # Ensure WebDriver is properly closed
        rich_console.print("Exiting the app...", style="bold red")
        sys.exit()

    def run(self):
        while True:
            command_input = Prompt.ask("\nEnter a command", default="help")
            command_parts = command_input.split(" ")

            command_name = command_parts[0]
            command_args = command_parts[1:]

            if command_name in self.commands:
                try:
                    self.commands[command_name](*command_args)
                except TypeError as e:
                    rich_console.print(f"[bold red]Error: Incorrect parameters ({e})[/bold red]")
            else:
                rich_console.print(f"[bold red]Invalid command:[/bold red] {command_name}. Type [bold green]help[/bold green] for available commands.")

if __name__ == "__main__":
    ui = CommandUI()
    ui.run()