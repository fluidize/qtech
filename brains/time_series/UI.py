import streamlit as st
import io
import os
import re

import plotly.graph_objects as go

from rnn_model import TimeSeriesPredictor, ModelTesting

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.sidebar.header('Adjust Train Parameters') #TRAINING

epochs = st.sidebar.number_input('Epochs', min_value=1, max_value=1000, value=5)
rnn_width = st.sidebar.number_input('RNN Width', min_value=1, max_value=8192, value=128)
dense_width = st.sidebar.number_input('Dense Width', min_value=1, max_value=8192, value=128)

ticker = st.sidebar.selectbox('Ticker', options=['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'])
chunks = st.sidebar.number_input('Chunks', min_value=1, max_value=100, value=5)
interval = st.sidebar.selectbox('Interval', options=['1m', '2m', '5m', '15m', '30m', '1h', '1d'], index=2)
age_days = st.sidebar.number_input('Age Days', min_value=1, max_value=365, value=10)

save_model = st.sidebar.checkbox('Save Model as .keras file')

if st.sidebar.button('Run Training'):
    with st.spinner("Training model...", show_time=True):
        model = TimeSeriesPredictor(epochs, rnn_width, dense_width, ticker, chunks, interval, age_days)
        model_data, plotly_figure = model.run(save=save_model)
        st.success('Model training complete!')
        st.plotly_chart(plotly_figure)

# Regex search for .keras files in the CWD
directory_files = os.listdir(os.getcwd())
keras_files = [f for f in directory_files if re.search(r'\.keras$', f)]

test_model = st.sidebar.selectbox('Select Model', keras_files)

st.sidebar.header('Adjust Test Parameters') #TESTING

if st.sidebar.button('Test Model'):
    with st.spinner("Testing model...", show_time=True):
        test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
        test_client._load_model(model_name=test_model)
        plotly_figure = test_client.run()

        st.plotly_chart(plotly_figure)
        
        summary_str = io.StringIO()
        test_client.model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        summary_str = summary_str.getvalue()

        st.markdown("<h3 style='color: #9b59b6; font-weight: bold;'>Model Summary</h3>", unsafe_allow_html=True)
        summary_lines = summary_str.split('\n')
        for line in summary_lines:
            if any(char.isalnum() for char in line.strip()):
                st.write(line)
