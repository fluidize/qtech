#!/usr/bin/env python3
"""
Bloomberg-style Financial Terminal
Built with Textual for TUI interface
"""

from fsspec.spec import threading
import yfinance as yf
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, Header, Footer, Input, Button, Static
from textual.containers import Horizontal, Vertical
import mplfinance as mpf
import matplotlib.pyplot as plt

class Terminal(App):
    CSS = """
        .input {
            margin: 1 1;
        }

        #ticker_input {
            width: 85%;            
        }

        #get_info_btn {
            width: 14%;
            align: center middle;
        }

        #get_chart_btn {
            width: 100%;
            align: center middle;
        }

        #info_collapsible {
            margin: 1 2;
            border: $primary;
        }

        #left {
            align: center middle;
        }

        #divider {
            width: 1;
            background: $surface;
            height: 100%;
            min-height: 1;
        }
        
        #info_title {
            border: round $primary;
            background: $surface;
            color: $text;
            text-align: center;
            margin: 1 2;
        }

        .info {
            margin: 1 2;
            text-style: bold;
        }
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticker = "N/A"

    def on_mount(self) -> None:
        self.theme = "tokyo-night"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                with Horizontal():
                    yield Input(placeholder="Enter ticker symbol (e.g. AAPL)", id="ticker_input", classes="input")
                    yield Button("Get Info", id="get_info_btn", classes="input")
                yield Button("Get Chart", id="get_chart_btn", classes="input")
                
                with Collapsible(title=f"{self.ticker} Company Report", collapsed=True, collapsed_symbol=">>>", expanded_symbol="vvv", id="info_collapsible"):
                    yield Static("Address: ", id="address", classes="info")
                    yield Static("City: ", id="city", classes="info")
                    yield Static("State: ", id="state", classes="info")
                    yield Static("Zipcode: ", id="zip", classes="info")
                    yield Static("Country: ", id="country", classes="info")
                    yield Static("Phone: ", id="phone", classes="info")
                    yield Static("Website: ", id="website", classes="info")
                    yield Static("Industry: ", id="industry", classes="info")
                    yield Static("Sector: ", id="sector", classes="info")
                    yield Static("CEO: ", id="ceo", classes="info")
                    yield Static("Employees: ", id="employees", classes="info")
                    yield Static("Description: ", id="description", classes="info")

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
    
    def _set_ticker(self) -> None:
        input_widget = self.query_one("#ticker_input", Input)
        self.ticker = input_widget.value.strip().upper()

        if not self.ticker:
            title = self.query_one("#info_title", Static)
            title.update("Please enter a ticker symbol.")
            return

    def update_other_info(self, info: dict) -> None:
        info_collapsible = self.query_one("#info_collapsible", Collapsible)
        info_collapsible.title = f"{self.ticker} Company Report"
        address = self.query_one("#address", Static)
        city = self.query_one("#city", Static)
        state = self.query_one("#state", Static)
        zipcode = self.query_one("#zip", Static)
        country = self.query_one("#country", Static)
        phone = self.query_one("#phone", Static)
        website = self.query_one("#website", Static)
        industry = self.query_one("#industry", Static)
        sector = self.query_one("#sector", Static)
        description = self.query_one("#description", Static)
        ceo = self.query_one("#ceo", Static)
        employees = self.query_one("#employees", Static)

        address.update(f"Address: {info.get('address1', 'N/A')}")
        city.update(f"City: {info.get('city', 'N/A')}")
        state.update(f"State: {info.get('state', 'N/A')}")
        zipcode.update(f"Zipcode: {info.get('zip', 'N/A')}")
        country.update(f"Country: {info.get('country', 'N/A')}")
        phone.update(f"Phone: {info.get('phone', 'N/A')}")
        website.update(f"Website: {info.get('website', 'N/A')}")
        industry.update(f"Industry: {info.get('industry', 'N/A')}")
        sector.update(f"Sector: {info.get('sector', 'N/A')}")
        description.update(f"Description: {info.get('longBusinessSummary', 'N/A')}")
        try:
            ceo.update(f"CEO: {info.get('companyOfficers', 'N/A')[0]['name']}")
        except:
            ceo.update(f"CEO: N/A")
        employees.update(f"Employees: {info.get('fullTimeEmployees', 'N/A')}")


    def update_info(self) -> None:
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

        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            self.update_other_info(info)
            if not info or 'longName' not in info:
                title.update(f"No data found for {self.ticker}.")
                return
            #get and update the info
            name = info.get('longName', 'N/A')
            current_price = info.get('currentPrice', 'N/A')
            high_value = info.get('fiftyTwoWeekHigh', 'N/A')
            low_value = info.get('fiftyTwoWeekLow', 'N/A')
            change_value = info.get('52WeekChange', 'N/A')
            market_cap_value = info.get('marketCap', 'N/A')
            pe_ratio_value = info.get('trailingPE', 'N/A')
            dividend_yield_value = info.get('dividendYield', 'N/A')
            volume_value = info.get('volume', 'N/A')
            average_volume_value = info.get('averageVolume', 'N/A')
            fifty_day_moving_average_value = info.get('fiftyDayAverage', 'N/A')
            two_hundred_day_moving_average_value = info.get('twoHundredDayAverage', 'N/A')
            earnings_date_value = info.get('earningsDate', 'N/A')
            dividend_date_value = info.get('dividendDate', 'N/A')
            dividend_amount_value = info.get('dividendRate', 'N/A')
            title.update(f"{self.ticker} - {name}")
            price.update(f"Current Price: {current_price}")
            high.update(f"52-Week High: {high_value}")
            low.update(f"52-Week Low: {low_value}")
            change.update(f"52-Week Change: {change_value}%")
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

    def show_chart(self) -> None:
        title = self.query_one("#info_title", Static)
        try:
            candlesticks = yf.Ticker(self.ticker).history(period="1y", interval="1d")
            def plot_chart():
                mpf.plot(
                    candlesticks,
                    type='candle',
                    style='charles',
                    figsize=(10, 6),
                    volume=True,
                    ylabel='Price',
                    ylabel_lower='Volume',
                    title=f"{self.ticker} Chart",
                    scale_padding={'left': 1, 'right': 1, 'top': 1, 'bottom': 1}
                )
            threading.Thread(target=plot_chart).start()
        except Exception as e:
            title = self.query_one("#info_title", Static)
            title.update(f"Error: {e}")
            return
        

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "get_info_btn":
            self._set_ticker()
            self.update_info()
        elif event.button.id == "get_chart_btn":
            self._set_ticker()
            self.show_chart()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "ticker_input":
            self._set_ticker()
            self.update_info()


if __name__ == "__main__":
    app = Terminal()
    app.run()
