import wiener_process as wiener
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def monte_carlo_simulation(n_steps, n_paths, time_unit=1):
    paths = []
    
    variance = time_unit / n_steps
    stddev = np.sqrt(variance)

    for i in range(n_paths):
        paths.append(wiener.wiener_process(n_steps, time_unit=time_unit, live_plot=False))
    
    paths_array = np.array(paths)
    
    df = pd.DataFrame(paths_array.T) 

    mean_path = df.mean(axis=1)
    stddev_path = df.std(axis=1)

    fig = go.Figure()
    for i in range(df.shape[1]):
        fig.add_trace(go.Scatter(x=df.index, y=df[i], mode='lines', name=f'Path {i+1}'))

    # Add shaded area for standard deviation
    fig.add_trace(go.Scatter(
        x=df.index,
        y=mean_path + stddev_path,
        mode='lines',
        line=dict(color='lightblue', dash='dash'),
        name='Mean + Std Dev',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=mean_path - stddev_path,
        mode='lines',
        line=dict(color='lightblue', dash='dash'),
        fill='tonexty',  # Fill area between the two lines
        name='Variance Area',
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title='Shaded Area Representing Variance Around Mean Path',
        xaxis_title='Steps',
        yaxis_title='Wiener Process Value'
    )

    fig.show()

if __name__ == "__main__":
    paths = monte_carlo_simulation(1000, 100, time_unit=1)
