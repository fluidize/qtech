import streamlit as st
import json

import plotly.graph_objects as go

from train_rnn import TimeSeriesPredictor

# Function to run training with given parameters
def run_train_rnn(rnn_width, dense_width):
    st.markdown(f'<p style="color:slateblue; font-weight:bold;">Training with RNN Width: {rnn_width}, Dense Width: {dense_width}</p>', unsafe_allow_html=True)
    model = TimeSeriesPredictor(rnn_width=rnn_width, dense_width=dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10)
    st.write(f'<p style="color:lime; font-weight:bold;">Training completed!</p>', unsafe_allow_html=True)

    return model #return history

# Streamlit app
st.title('RNN Training App')

# Sidebar for parameters
st.sidebar.header('Adjust Train Parameters')
rnn_width = st.sidebar.slider('RNN Width', min_value=1, max_value=2048, value=1)
dense_width = st.sidebar.slider('Dense Width', min_value=1, max_value=2048, value=1)

# Button to run training
if st.sidebar.button('Run Training'):
    result = run_train_rnn(rnn_width, dense_width)
    # Plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(result['loss']))), y=result['loss'], mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(result['mean_squared_error']))), y=result['mean_squared_error'], mode='lines', name='Mean Squared Error'))
    st.plotly_chart(fig)

if st.sidebar.button('Test Model'):
    