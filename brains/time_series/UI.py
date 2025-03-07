import streamlit as st
import json

import plotly.graph_objects as go

from train_rnn import TimeSeriesPredictor

# Function to run training with given parameters
def run_train_rnn(rnn_width, dense_width):
    model = TimeSeriesPredictor(rnn_width=rnn_width, dense_width=dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10)
    st.markdown(f'<p style="color:green;">Training with RNN Width: {rnn_width}, Dense Width: {dense_width}</p', unsafe_allow_html=True)

    return model #return history

# Streamlit app
st.title('RNN Training App')

# Sidebar for parameters
st.sidebar.header('Adjust Parameters')
rnn_width = st.sidebar.slider('RNN Width', min_value=1, max_value=2048, value=128)
dense_width = st.sidebar.slider('Dense Width', min_value=1, max_value=2048, value=128)

# Button to run training
if st.sidebar.button('Run Training'):
    st.write('Running training...')
    result = run_train_rnn(rnn_width, dense_width)
    st.write('Training completed!')

    # Plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(result['loss']))), y=result['loss'], mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(result['mean_squared_error']))), y=result['mean_squared_error'], mode='lines', name='Mean Squared Error'))
    st.plotly_chart(fig)