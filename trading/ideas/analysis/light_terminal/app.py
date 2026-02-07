import numpy as np
import yfinance as yf
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, Header, Footer, Input, Button, Static
from textual.containers import Horizontal, HorizontalGroup, Vertical, VerticalScroll
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from trading.options.curve.main import get_option_chain, filter_otm, get_option_iv, days_to_expiry

def show_volatility_surface(ticker: str) -> None:
    ticker_obj = yf.Ticker(ticker)
    x_calls = np.array([])
    y_calls = np.array([])
    z_calls = np.array([])
    x_puts = np.array([])
    y_puts = np.array([])
    z_puts = np.array([])
    spot = None

    for expiration in ticker_obj.options:
        chain, spot, expiration = get_option_chain(ticker, expiration)
        otm_calls, otm_puts = filter_otm(chain, spot)
        otm_calls_iv = get_option_iv(otm_calls)
        otm_puts_iv = get_option_iv(otm_puts)
        otm_calls_strike = otm_calls["strike"]
        otm_puts_strike = otm_puts["strike"]
        days_exp = days_to_expiry(expiration)

        x_calls = np.append(x_calls, otm_calls_strike)
        y_calls = np.append(y_calls, np.full(len(otm_calls_strike), days_exp))
        z_calls = np.append(z_calls, otm_calls_iv)
        x_puts = np.append(x_puts, otm_puts_strike)
        y_puts = np.append(y_puts, np.full(len(otm_puts_strike), days_exp))
        z_puts = np.append(z_puts, otm_puts_iv)

    if x_calls.size == 0 and x_puts.size == 0:
        raise ValueError(f"No option data for {ticker}")

    x_all = np.concatenate([x_calls, x_puts])
    y_all = np.concatenate([y_calls, y_puts])
    z_all = np.concatenate([z_calls, z_puts])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_calls, y_calls, z_calls, c="green", marker="o", label="OTM Calls", s=10, alpha=0.5)
    ax.scatter(x_puts, y_puts, z_puts, c="red", marker="o", label="OTM Puts", s=10, alpha=0.5)

    xi = np.linspace(x_all.min(), x_all.max(), 50)
    yi = np.linspace(y_all.min(), y_all.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi_grid = griddata((x_all, y_all), z_all, (xi_grid, yi_grid), method="linear")
    ax.plot_surface(xi_grid, yi_grid, zi_grid, cmap="viridis", alpha=0.5)

    if spot is not None:
        x_plane = np.array([spot, spot])
        y_plane = np.array([0, y_all.max()])
        ax.plot(x_plane, y_plane, np.zeros_like(x_plane), color="black", label="Spot")

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"Volatility Surface for {ticker}")
    ax.legend()
    plt.tight_layout()
    plt.show()

def show_volatility_curve(ticker: str) -> None:
    chain, spot, expiration = get_option_chain(ticker)
    otm_calls, otm_puts = filter_otm(chain, spot)

    otm_calls_iv = get_option_iv(otm_calls)
    otm_puts_iv = get_option_iv(otm_puts)

    otm_calls_strike = otm_calls['strike']
    otm_puts_strike = otm_puts['strike']

    plt.figure(figsize=(10, 6))

    plt.plot(otm_calls_strike, otm_calls_iv, label='OTM Calls', color='green', marker='o')
    plt.plot(otm_puts_strike, otm_puts_iv, label='OTM Puts', color='red', marker='o')
    plt.axvline(spot, color='black', linestyle='--', label='Spot Price')

    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Volatility Curve for {ticker} (Exp: {expiration})')
    plt.legend()
    plt.tight_layout()
    plt.show()

class Terminal(App):
    CSS = """
        .input {
            margin: 1 2;
        }

        #ticker_input {
            width: 85%;
        }

        #get_info_btn {
            width: 15%;
            align: center middle;
        }

        #info_collapsible {
            margin: 1 2 1 2;
            margin-top: 0;
        }

        #report_scroll {
            height: 20;
            max-height: 20;
            overflow-y: auto;
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
        self.theme = "nord"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                with Horizontal():
                    yield Input(placeholder="Enter ticker symbol (e.g. AAPL)", id="ticker_input", classes="input")
                    yield Button("Load Info", id="get_info_btn", classes="input")
                with HorizontalGroup(id="buttons_row"):
                    yield Button("Price Chart", id="get_chart_btn", classes="input")
                    yield Button("Volatility Surface", id="volatility_surface_btn", classes="input")
                    yield Button("Volatility Curve", id="volatility_curve_btn", classes="input")
                with Collapsible(title=f"{self.ticker} Report", collapsed=True, collapsed_symbol="→", expanded_symbol="↘", id="info_collapsible"):
                    with VerticalScroll(id="report_scroll"):
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
                yield Static("Dividend Yield (%): ", id="dividend_yield", classes="info")
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
        info_collapsible.title = f"{self.ticker} Report"
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
            current_price = info.get('regularMarketPrice', 'N/A')
            high_value = info.get('fiftyTwoWeekHigh', 'N/A')
            low_value = info.get('fiftyTwoWeekLow', 'N/A')
            change_value = info.get('52WeekChange', 'N/A')
            if change_value == "N/A":
                history = stock.history(period="1y", interval="1d")
                change_value = (history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0]

            market_cap_value = info.get('marketCap', 'N/A')
            if market_cap_value != "N/A":
                if market_cap_value / 1_000_000_000_000 > 1:
                    market_cap_value = f"{market_cap_value / 1_000_000_000_000:.2f}T"
                elif market_cap_value / 1_000_000_000 > 1:
                    market_cap_value = f"{market_cap_value / 1_000_000_000:.2f}B"
                elif market_cap_value / 1_000_000 > 1:
                    market_cap_value = f"{market_cap_value / 1_000_000:.2f}M"
                else:
                    market_cap_value = f"{market_cap_value:.2f / 1_000}K"
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
            try:
                change.update(f"52-Week Change: {int(change_value*100)}%")
            except:
                change.update(f"52-Week Change: N/A")
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
            # Get stock data
            stock = yf.Ticker(self.ticker)
            data = stock.history(period="1y", interval="1d")
            
            # Create subplots for price and volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{self.ticker} Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ), row=1, col=1)
            
            # Add volume bars
            colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{self.ticker} Stock Chart (1 Year)',
                yaxis_title='Price',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            fig.show()
        except Exception as e:
            title.update(f"Error: {e}")
            return

    def show_volatility_surface(self) -> None:
        title = self.query_one("#info_title", Static)
        try:
            show_volatility_surface(self.ticker)
        except Exception as e:
            title.update(f"Error: {e}")
            return
    
    def show_volatility_curve(self) -> None:
        title = self.query_one("#info_title", Static)
        try:
            show_volatility_curve(self.ticker)
        except Exception as e:
            title.update(f"Error: {e}")
            return
        

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "get_info_btn":
            self._set_ticker()
            self.update_info()
        elif event.button.id == "get_chart_btn":
            self._set_ticker()
            self.show_chart()
            self.update_info()
        elif event.button.id == "volatility_surface_btn":
            self._set_ticker()
            self.show_volatility_surface()
            self.update_info()
        elif event.button.id == "volatility_curve_btn":
            self._set_ticker()
            self.show_volatility_curve()
            self.update_info()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "ticker_input":
            self._set_ticker()
            self.update_info()


if __name__ == "__main__":
    app = Terminal()
    app.run()

    
