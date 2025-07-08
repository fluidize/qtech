#!/usr/bin/env python3
"""
Bloomberg-style Financial Terminal
Built with Textual for TUI interface
"""

import asyncio
from datetime import datetime
from typing import Optional

import yfinance as yf
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)
from textual.reactive import reactive
from textual.message import Message
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console


class TickerInfo(Static):
    """Widget to display ticker information"""
    
    def __init__(self, renderable="", **kwargs):
        super().__init__(renderable, **kwargs)
        self.ticker_data = None
    
    def update_ticker(self, ticker: str):
        """Update the display with new ticker information"""
        try:
            # Fetch ticker data
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            history = stock.history(period="1d")
            
            if history.empty or not info:
                self.update("❌ Ticker not found or no data available")
                return
            
            # Get current price
            current_price = history['Close'].iloc[-1] if not history.empty else info.get('currentPrice', 'N/A')
            
            # Create formatted display text
            company_name = info.get('longName', 'N/A')
            
            # Format price
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = str(current_price)
            
            # Build the display text
            display_text = f"""📈 {ticker.upper()} - {company_name}

💰 FINANCIAL METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current Price:      {price_str}
Market Cap:         {self._format_large_number(info.get('marketCap', 'N/A'))}
P/E Ratio:          {f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A'}
Forward P/E:        {f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A'}
EPS (TTM):          {f"${info.get('trailingEps', 'N/A'):.2f}" if info.get('trailingEps') else 'N/A'}
Dividend Yield:     {f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A'}

📊 TRADING DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

52W High:           {f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A'}
52W Low:            {f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A'}
Volume:             {self._format_large_number(info.get('volume', 'N/A'))}
Avg Volume:         {self._format_large_number(info.get('averageVolume', 'N/A'))}
Beta:               {f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A'}

🏢 COMPANY INFO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sector:             {info.get('sector', 'N/A')}
Industry:           {info.get('industry', 'N/A')}
Employees:          {self._format_large_number(info.get('fullTimeEmployees', 'N/A'))}
Website:            {info.get('website', 'N/A')}

📈 ADDITIONAL DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Previous Close:     ${info.get('previousClose', 'N/A')}
Day Range:          ${info.get('dayLow', 'N/A')} - ${info.get('dayHigh', 'N/A')}
Last Updated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            self.update(display_text)
            
        except Exception as e:
            self.update(f"❌ Error fetching data for {ticker}: {str(e)}")
    
    def _format_large_number(self, num):
        """Format large numbers with K, M, B suffixes"""
        if not isinstance(num, (int, float)) or num == 'N/A':
            return 'N/A'
        
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        elif num >= 1e3:
            return f"${num/1e3:.2f}K"
        else:
            return f"${num:.2f}"


class SearchWidget(Container):
    """Search widget for ticker input"""
    
    class TickerSubmitted(Message):
        """Message sent when ticker is submitted"""
        
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker
            super().__init__()
    
    def compose(self) -> ComposeResult:
        yield Label("Enter Ticker Symbol:", classes="search-label")
        yield Input(placeholder="e.g., AAPL, MSFT, TSLA...", id="ticker-input")
        yield Button("Search", variant="primary", id="search-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle search button press"""
        if event.button.id == "search-btn":
            ticker_input = self.query_one("#ticker-input", Input)
            if ticker_input.value.strip():
                self.post_message(self.TickerSubmitted(ticker_input.value.strip()))
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input field"""
        if event.input.id == "ticker-input" and event.input.value.strip():
            self.post_message(self.TickerSubmitted(event.input.value.strip()))


class FinancialTerminal(App):
    """Main Terminal Application"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    .search-container {
        height: auto;
        padding: 1;
        background: $primary-background;
        border: solid $primary;
    }
    
    .search-label {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .info-container {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $secondary;
        margin-top: 1;
    }
    
    Input {
        margin-bottom: 1;
    }
    
    Button {
        margin-left: 1;
    }
    
    #ticker-input {
        background: $surface;
        border: solid $accent;
    }
    
    #search-btn {
        background: $accent;
        color: $text;
    }
    """
    
    TITLE = "Financial Terminal - Bloomberg Style"
    SUB_TITLE = "Real-time Stock Information"
    
    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header()
        with Container(classes="search-container"):
            yield SearchWidget()
        with Container(classes="info-container"):
            yield TickerInfo("💡 Enter a ticker symbol above to get started", id="ticker-info")
        yield Footer()
    
    def on_search_widget_ticker_submitted(self, message: SearchWidget.TickerSubmitted) -> None:
        """Handle ticker submission"""
        ticker_info = self.query_one("#ticker-info", TickerInfo)
        ticker_info.update("🔄 Loading...")
        
        # Update the display with ticker information
        ticker_info.update_ticker(message.ticker)
        
        # Clear the input
        ticker_input = self.query_one("#ticker-input", Input)
        ticker_input.value = ""
    
    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


def main():
    """Run the application"""
    app = FinancialTerminal()
    app.run()


if __name__ == "__main__":
    main()
