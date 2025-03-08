import streamlit as st
import json

import plotly.graph_objects as go

from rnn_model import TimeSeriesPredictor, ModelTesting

# Function to run training with given parameters
def run_train_rnn(rnn_width, dense_width):
    st.markdown(f'<p style="color:slateblue; font-weight:bold;">Training with RNN Width: {rnn_width}, Dense Width: {dense_width}</p>', unsafe_allow_html=True)
    model = TimeSeriesPredictor(rnn_width=rnn_width, dense_width=dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10)
    st.write(f'<p style="color:lime; font-weight:bold;">Training completed!</p>', unsafe_allow_html=True)

    return model #return history

st.sidebar.header('Adjust Train Parameters') #TRAINING
epochs = st.sidebar.number_input('Epochs', min_value=1, max_value=1000, value=5)
rnn_width = st.sidebar.number_input('RNN Width', min_value=1, max_value=2048, value=128)
dense_width = st.sidebar.number_input('Dense Width', min_value=1, max_value=2048, value=128)
save_model = st.sidebar.checkbox('Save Model as .keras file')

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
    with st.spinner("Testing model...", show_time=True):
        test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
        test_client._load_model(model_name=test_model)
        plotly_figure = test_client.run()

        layer_details = test_client._get_summary(test_client.model)

        st.plotly_chart(plotly_figure)
        
        summary_str = io.StringIO()
        test_client.model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        summary_str = summary_str.getvalue()

        st.markdown("<h3 style='color: #9b59b6; font-weight: bold;'>Model Summary</h3>", unsafe_allow_html=True)
        summary_lines = summary_str.split('\n')
        for line in summary_lines:
            if any(char.isalnum() for char in line.strip()):
                st.write(line)
