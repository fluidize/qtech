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

        .input {
            margin: 1 2;
        }
        
        #info_title {
            border: round $primary;
            background: $primary;
            color: $text;
            text-align: center;
            margin: 1 2;
        }

        .info {
            margin: 1 2;
        }
        """
    def on_mount(self) -> None:
        self.theme = "tokyo-night"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="Enter ticker symbol (e.g. AAPL)", id="ticker_input", classes="input")
                yield Button("Get Info", id="get_info_btn", classes="input")
            yield Static("", id="divider")
            with Vertical(id="right"):
                yield Static("Enter a ticker.", id="info_title", classes="info")
                yield Static("Current Price: ", id="price", classes="info")
                yield Static("52-Week High: ", id="high", classes="info")
                yield Static("52-Week Low: ", id="low", classes="info")
                yield Static("52-Week Change: ", id="change", classes="info")
                yield Static("Market Cap: ", id="market_cap", classes="info")
                yield Static("P/E Ratio: ", id="pe_ratio", classes="info")
                yield Static("Dividend Yield: ", id="dividend_yield", classes="info")
                yield Static("Volume: ", id="volume", classes="info")
                yield Static("Average Volume: ", id="average_volume", classes="info")
                yield Static("50-Day Moving Average: ", id="fifty_day_moving_average", classes="info")
                yield Static("200-Day Moving Average: ", id="two_hundred_day_moving_average", classes="info")
                yield Static("Earnings Date: ", id="earnings_date", classes="info")
                yield Static("Dividend Date: ", id="dividend_date", classes="info")
                yield Static("Dividend Amount: ", id="dividend_amount", classes="info")
        yield Footer()

    def update_info(self) -> None:
        input_widget = self.query_one("#ticker_input", Input)
        ticker = input_widget.value.strip().upper()

        title = self.query_one("#info_title", Static)
        price = self.query_one("#price", Static)
        high = self.query_one("#high", Static)
        low = self.query_one("#low", Static)
        change = self.query_one("#change", Static)
        market_cap = self.query_one("#market_cap", Static)
        pe_ratio = self.query_one("#pe_ratio", Static)
        dividend_yield = self.query_one("#dividend_yield", Static)
        volume = self.query_one("#volume", Static)
        average_volume = self.query_one("#average_volume", Static)
        fifty_day_moving_average = self.query_one("#fifty_day_moving_average", Static)
        two_hundred_day_moving_average = self.query_one("#two_hundred_day_moving_average", Static)
        earnings_date = self.query_one("#earnings_date", Static)
        dividend_date = self.query_one("#dividend_date", Static)
        dividend_amount = self.query_one("#dividend_amount", Static)

        if not ticker:
            title.update("Please enter a ticker symbol.")
            return
        title.update("Loading...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or 'longName' not in info:
                title.update(f"No data found for {ticker}.")
                return
            #get and update the info
            name = info.get('longName', 'N/A')
            current_price = info.get('currentPrice', 'N/A')
            high_value = info.get('fiftyTwoWeekHigh', 'N/A')
            low_value = info.get('fiftyTwoWeekLow', 'N/A')
            change_value = info.get('fiftyTwoWeekChange', 'N/A')
            market_cap_value = info.get('marketCap', 'N/A')
            pe_ratio_value = info.get('trailingPE', 'N/A')
            dividend_yield_value = info.get('dividendYield', 'N/A')
            volume_value = info.get('volume', 'N/A')
            average_volume_value = info.get('averageVolume', 'N/A')
            fifty_day_moving_average_value = info.get('fiftyDayAverage', 'N/A')
            two_hundred_day_moving_average_value = info.get('twoHundredDayAverage', 'N/A')
            earnings_date_value = info.get('earningsDate', 'N/A')
            dividend_date_value = info.get('dividendDate', 'N/A')
            dividend_amount_value = info.get('dividendAmount', 'N/A')
            title.update(f"{ticker} - {name}")
            price.update(f"Current Price: {current_price}")
            high.update(f"52-Week High: {high_value}")
            low.update(f"52-Week Low: {low_value}")
            change.update(f"52-Week Change: {change_value}")
            market_cap.update(f"Market Cap: {market_cap_value}")
            pe_ratio.update(f"P/E Ratio: {pe_ratio_value}")
            dividend_yield.update(f"Dividend Yield: {dividend_yield_value}")
            volume.update(f"Volume: {volume_value}")
            average_volume.update(f"Average Volume: {average_volume_value}")
            fifty_day_moving_average.update(f"50-Day Moving Average: {fifty_day_moving_average_value}")
            two_hundred_day_moving_average.update(f"200-Day Moving Average: {two_hundred_day_moving_average_value}")
            earnings_date.update(f"Earnings Date: {earnings_date_value}")
            dividend_date.update(f"Dividend Date: {dividend_date_value}")
            dividend_amount.update(f"Dividend Amount: {dividend_amount_value}")
        except Exception as e:
            title.update(f"Error: {e}")
            raise e
        

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "get_info_btn":
            self.update_info()


if __name__ == "__main__":
    app = Terminal()
    app.run()
