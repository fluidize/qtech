import json
from datetime import datetime
from urllib.request import Request, urlopen

import numpy as np
from rich.text import Text
import yfinance as yf
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Button, Label, ListItem, ListView, Static
from textual.containers import Horizontal, HorizontalGroup, Vertical, VerticalScroll
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from trading.options.curve.main import get_option_chain, filter_otm, get_option_iv, days_to_expiry


TICKERTICK_BASE = "https://api.tickertick.com/feed"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"}


def _fuzzy_match(text: str, pattern: str) -> bool:
    pattern = pattern.lower()
    text = text.lower()
    i = 0
    for c in text:
        if i < len(pattern) and c == pattern[i]:
            i += 1
    return i == len(pattern)


def fetch_rss_articles(ticker: str) -> list[dict]:
    try:
        q = f"z:{ticker.lower()}&(T:earning?T:sec)"
        url = f"{TICKERTICK_BASE}?q={q}&n=200"
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        return [
            {
                "title": s.get("title", ""),
                "summary": s.get("description", ""),
                "link": s.get("url", ""),
            }
            for s in data.get("stories", [])
        ]
    except Exception:
        return []


def format_num(value, prefix: str = "", suffix: str = "", compact: bool = True, decimals: int = 2) -> str:
    if value is None or value == "N/A":
        return "N/A"
    if isinstance(value, str):
        return value
    if compact:
        if value >= 1e12:
            return f"{prefix}{value/1e12:.2f}T{suffix}"
        if value >= 1e9:
            return f"{prefix}{value/1e9:.2f}B{suffix}"
        if value >= 1e6:
            return f"{prefix}{value/1e6:.2f}M{suffix}"
        if value >= 1e3:
            return f"{prefix}{value/1e3:.2f}K{suffix}"
    return f"{prefix}{format(value, f'.{decimals}f')}{suffix}"


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

        #report_title {
            text-style: bold;
            border: round $primary;
            background: $surface;
            color: $text;
            text-align: center;
            margin-bottom: 1;
        }

        #report_scroll {
            height: 16;
            min-height: 16;
            overflow-y: auto;
        }

        #left, #right {
            height: 100%;
        }

        #left_top {
            height: 1fr;
            min-height: 8;
        }

        #right_top {
            height: 1fr;
            min-height: 8;
        }

        #ticker_info_title {
            text-style: bold;
            border: round $primary;
            background: $surface;
            color: $text;
            text-align: center;
            margin: 1 2;
        }

        .ticker_info {
            margin: 0 2;
        }

        #report_scroll {
            margin: 1 2;
            margin-bottom: 0;
            background: $surface;
        }

        .report_info {
            margin: 1 2;
            text-style: bold;
        }

        #news_summary_scroll {
            max-height: 30%;
            border: round $primary;
        }

        .news_summary {
            margin: 1 2;
        }

        .divider_h {
            height: 1;
            min-height: 1;
            background: $surface;
            width: 100%;
        }

        .divider_v {
            width: 1;
            min-width: 1;
            background: $surface;
            height: 100%;
        }

        #news_search {
            margin: 1 2;
        }

        #news_list {
            background: $background;
        }
        #news_list > ListItem {
            margin: 1 2;
        }
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticker = "N/A"
        self.news_articles = []
        self.filtered_articles = []

    def on_mount(self) -> None:
        self.theme = "solarized-light"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                with Vertical(id="left_top"):
                    yield Input(placeholder="Enter ticker symbol (e.g. AAPL)", id="ticker_input", classes="input")
                    with HorizontalGroup(id="buttons_row"):
                        yield Button("Price Chart", id="get_chart_btn", classes="input")
                        yield Button("Volatility Surface", id="volatility_surface_btn", classes="input")
                        yield Button("Volatility Curve", id="volatility_curve_btn", classes="input")
                    with VerticalScroll(id="report_scroll"):
                        yield Static(f"{self.ticker} Report", id="report_title")
                        yield Static("Address: ", id="address", classes="report_info")
                        yield Static("City: ", id="city", classes="report_info")
                        yield Static("State: ", id="state", classes="report_info")
                        yield Static("Zipcode: ", id="zip", classes="report_info")
                        yield Static("Country: ", id="country", classes="report_info")
                        yield Static("Phone: ", id="phone", classes="report_info")
                        yield Static("Website: ", id="website", classes="report_info")
                        yield Static("Industry: ", id="industry", classes="report_info")
                        yield Static("Sector: ", id="sector", classes="report_info")
                        yield Static("CEO: ", id="ceo", classes="report_info")
                        yield Static("Employees: ", id="employees", classes="report_info")
                        yield Static("Description: ", id="description", classes="report_info")
                with VerticalScroll(id="news_summary_scroll"):
                    yield Static("Select an article for summary", id="news_summary", classes="news_summary")

            yield Static("", classes="divider_v")

            with Vertical(id="right"):
                with VerticalScroll(id="right_top"):
                    yield Static("Enter a ticker.", id="ticker_info_title", classes="ticker_info")
                    yield Static("Current Price: ", id="price", classes="ticker_info")
                    yield Static("52-Week High: ", id="high", classes="ticker_info")
                    yield Static("52-Week Low: ", id="low", classes="ticker_info")
                    yield Static("52-Week Change: ", id="change", classes="ticker_info")
                    yield Static("Market Cap: ", id="market_cap", classes="ticker_info")
                    yield Static("P/E Ratio: ", id="pe_ratio", classes="ticker_info")
                    yield Static("Dividend Yield (%): ", id="dividend_yield", classes="ticker_info")
                    yield Static("Volume: ", id="volume", classes="ticker_info")
                    yield Static("Average Volume: ", id="average_volume", classes="ticker_info")
                    yield Static("50-Day Moving Average: ", id="fifty_day_moving_average", classes="ticker_info")
                    yield Static("200-Day Moving Average: ", id="two_hundred_day_moving_average", classes="ticker_info")
                    yield Static("Earnings Date: ", id="earnings_date", classes="ticker_info")
                    yield Static("Dividend Date: ", id="dividend_date", classes="ticker_info")
                    yield Static("Dividend Amount: ", id="dividend_amount", classes="ticker_info")
                    yield Static("365d Sharpe: ", id="sharpe_365d", classes="ticker_info")
                yield Static("", classes="divider_h")
                yield Input(placeholder="Search news...", id="news_search", classes="input")
                yield ListView(id="news_list")
        yield Footer()
    
    def _set_ticker(self) -> None:
        input_widget = self.query_one("#ticker_input", Input)
        self.ticker = input_widget.value.strip().upper()

        if not self.ticker:
            title = self.query_one("#ticker_info_title", Static)
            title.update("Please enter a ticker symbol.")
            return

    def update_other_info(self, info: dict) -> None:
        self.query_one("#report_title", Static).update(f"{self.ticker} Report")
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
        title = self.query_one("#ticker_info_title", Static)
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
        sharpe_365d = self.query_one("#sharpe_365d", Static)

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

            market_cap_value = format_num(info.get('marketCap', 'N/A'), prefix="$")
            pe_ratio_value = info.get('trailingPE', 'N/A')
            dividend_yield_value = info.get('dividendYield', 'N/A')
            volume_value = info.get('volume', 'N/A')
            average_volume_value = info.get('averageVolume', 'N/A')
            volume_display = format_num(volume_value * current_price if volume_value != 'N/A' and current_price != 'N/A' else 'N/A', prefix="$")
            avg_vol_display = format_num(average_volume_value, prefix="shares ", decimals=0)
            fifty_day_moving_average_value = info.get('fiftyDayAverage', 'N/A')
            two_hundred_day_moving_average_value = info.get('twoHundredDayAverage', 'N/A')
            earnings_date_value = info.get('earningsDate', 'N/A')
            dividend_date_value = info.get('dividendDate', 'N/A')
            dividend_amount_value = info.get('dividendRate', 'N/A')

            title.update(f"{self.ticker} - {name}")
            price.update(f"Current Price: {format_num(current_price, prefix='$', compact=False)}")

            if (high_value - current_price) / current_price > 0.5:
                tag = "green"
            elif (current_price - low_value) / low_value < -0.5:
                tag = "red"
            else:
                tag = "orange"
            high.update(f"[{tag}]52-Week High: {format_num(high_value, prefix='$', compact=False)}[/{tag}]")

            if (current_price - low_value) / low_value > 0.5:
                tag = "red"
            elif (current_price - low_value) / low_value < -0.5:
                tag = "green"
            else:
                tag = "orange"
            low.update(f"[{tag}]52-Week Low: {format_num(low_value, prefix='$', compact=False)}[/{tag}]")

            change_str = format_num(change_value * 100 if change_value not in ('N/A', None) else 'N/A', suffix='%', compact=False, decimals=0)
            tag = "green" if change_value > 0 else "red"
            change.update(f"[{tag}]52-Week Change: {change_str}[/{tag}]")

            market_cap.update(f"Market Cap: {market_cap_value}")
            pe_ratio.update(f"P/E Ratio: {pe_ratio_value}")
            dividend_yield.update(f"Dividend Yield: {format_num(dividend_yield_value if dividend_yield_value not in ('N/A', None) else 'N/A', suffix='%', compact=False)}")
            volume.update(f"Volume: {volume_display}")
            average_volume.update(f"Average Volume: {avg_vol_display}")
            fifty_day_moving_average.update(f"50-Day Moving Average: {format_num(fifty_day_moving_average_value, prefix='$', compact=False)}")
            two_hundred_day_moving_average.update(f"200-Day Moving Average: {format_num(two_hundred_day_moving_average_value, prefix='$', compact=False)}")
            earnings_date.update(f"Earnings Date: {earnings_date_value}")
            dividend_date_display = "N/A"
            if dividend_date_value not in ("N/A", None):
                months = {
                    1: "Jan",
                    2: "Feb",
                    3: "Mar",
                    4: "Apr",
                    5: "May",
                    6: "Jun",
                    7: "Jul",
                    8: "Aug",
                    9: "Sep",
                    10: "Oct",
                    11: "Nov",
                    12: "Dec"
                }
                dividend_date_display = datetime.fromtimestamp(dividend_date_value).strftime(f"{datetime.fromtimestamp(dividend_date_value).day} {months[datetime.fromtimestamp(dividend_date_value).month]} {datetime.fromtimestamp(dividend_date_value).year}")
            dividend_date.update(f"Dividend Date: {dividend_date_display}")
            dividend_amount.update(f"Dividend Amount: {format_num(dividend_amount_value, prefix='$', compact=False)}")

            history = stock.history(period="1y", interval="1d", auto_adjust=True)
            if len(history) > 1:
                returns = history['Close'].pct_change().dropna()
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                sharpe_365d.update(f"365d Sharpe: {sharpe:.4f}")
            else:
                sharpe_365d.update(f"365d Sharpe: N/A")
        except Exception as e:
            title.update(f"Error: {e}")
        self._update_news()

    def _update_news(self) -> None:
        if not self.ticker or self.ticker == "N/A":
            return
        self.news_articles = fetch_rss_articles(self.ticker)
        news_summary = self.query_one("#news_summary", Static)
        news_summary.update("Select an article for summary")
        self._filter_news_list()

    def _filter_news_list(self) -> None:
        news_articles = getattr(self, "news_articles", None)
        if news_articles is None:
            return
        search_input = self.query_one("#news_search", Input)
        search = search_input.value.strip().lower()
        if search:
            self.filtered_articles = [a for a in news_articles if _fuzzy_match(a.get("title", ""), search)]
        else:
            self.filtered_articles = list(news_articles)
        news_list = self.query_one("#news_list", ListView)
        news_list.clear()
        for a in self.filtered_articles:
            news_list.append(ListItem(Label(a["title"], classes="headline")))


    def on_list_view_selected(self, message: ListView.Selected) -> None:
        if message.list_view.id != "news_list":
            return
        if not getattr(self, "filtered_articles", None):
            return
        idx = message.index
        if 0 <= idx < len(self.filtered_articles):
            article = self.filtered_articles[idx]
            summary = article.get("summary", "")
            link = article.get("link", "")
            content = Text()
            if summary:
                content.append(summary)
            content.append("\n\n")
            if link:
                content.append(link, style="bold link " + link)
            self.query_one("#news_summary", Static).update(content)

    def show_chart(self) -> None:
        title = self.query_one("#ticker_info_title", Static)
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(period="1y", interval="1d", auto_adjust=True)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{self.ticker} Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ), row=1, col=1)
            
            colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)
            
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
        title = self.query_one("#ticker_info_title", Static)
        try:
            show_volatility_surface(self.ticker)
        except Exception as e:
            title.update(f"Error: {e}")
            return
    
    def show_volatility_curve(self) -> None:
        title = self.query_one("#ticker_info_title", Static)
        try:
            show_volatility_curve(self.ticker)
        except Exception as e:
            title.update(f"Error: {e}")
            return
        

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "get_chart_btn":
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
        elif event.input.id == "news_search":
            self._filter_news_list()


if __name__ == "__main__":
    app = Terminal()
    app.run()

    
