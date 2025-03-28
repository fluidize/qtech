import dash
from dash import dcc, html
import plotly.graph_objs as go
import random
import pandas as pd
import time

app = dash.Dash(__name__)

# Sample OHLC data
ohlc_data = []

def fetch_data():
    timestamp = pd.to_datetime('now')
    open_price = random.uniform(100, 110)
    close_price = open_price + random.uniform(-2, 2)
    high = max(open_price, close_price) + random.uniform(0, 2)
    low = min(open_price, close_price) - random.uniform(0, 2)
    ohlc_data.append(dict(
        x=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close_price
    ))
    
    if len(ohlc_data) > 50:
        ohlc_data.pop(0)

def create_ohlc_figure():
    #make candles closer to each other
    return {
        'data': [go.Candlestick(
            x=[d['x'] for d in ohlc_data],
            open=[d['open'] for d in ohlc_data],
            high=[d['high'] for d in ohlc_data],
            low=[d['low'] for d in ohlc_data],
            close=[d['close'] for d in ohlc_data],
            increasing_line_color='green',
            decreasing_line_color='red',
            increasing_fillcolor='green',
            decreasing_fillcolor='red'
        )],
        'layout': go.Layout(
            title="Live Candlestick Chart",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                tickformat='%H:%M:%S',
                dtick=1000  # 1 second intervals    
            ),
            yaxis=dict(
                tickformat='.2f'
            ),
            template='plotly_dark'
        )
    }

app.layout = html.Div([
    dcc.Graph(id='ohlc-chart', animate=True),
])

@app.callback(
    dash.dependencies.Output('ohlc-chart', 'figure'),
    dash.dependencies.Input('ohlc-chart', 'relayoutData')
)
def update_ohlc(_):
    fetch_data()
    return create_ohlc_figure()

if __name__ == '__main__':
    app.run(debug=True)
