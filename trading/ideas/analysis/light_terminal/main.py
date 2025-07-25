#!/usr/bin/env python3
"""
Bloomberg-style Financial Terminal
Built with Textual for TUI interface
"""

import yfinance as yf
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Button, Static
from textual.containers import Horizontal, Vertical

class Terminal(App):
    CSS = """
        #divider {
            width: 1;
            background: $surface;
            height: 100%;
            min-height: 1;
        }
        """
    def on_mount(self) -> None:
        self.theme = "tokyo-night"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="Enter ticker symbol (e.g. AAPL)", id="ticker_input")
                yield Button("Get Info", id="get_info_btn")
            yield Static("", id="divider")
            with Vertical(id="right"):
                yield Static("Enter a ticker and press the button.", id="output")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "get_info_btn":
            input_widget = self.query_one("#ticker_input", Input)
            ticker = input_widget.value.strip().upper()
            output = self.query_one("#output", Static)
            if not ticker:
                output.update("Please enter a ticker symbol.")
                return
            output.update("Loading...")
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info or 'longName' not in info:
                    output.update(f"No data found for {ticker}.")
                    return
                name = info.get('longName', 'N/A')
                price = info.get('currentPrice', 'N/A')
                output.update(f"{ticker} - {name}\nCurrent Price: {price}")
            except Exception as e:
                output.update(f"Error: {e}")

if __name__ == "__main__":
    app = Terminal()
    app.run()
