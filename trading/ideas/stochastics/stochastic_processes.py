import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tqdm import tqdm
from plotly.subplots import make_subplots
import scipy.stats as stats
import matplotlib.pyplot as plt

def wiener_process(n_steps, time_unit=1):
    w = np.zeros(n_steps * time_unit)
    delta_t = time_unit / n_steps

    for i in range(1, n_steps * time_unit):
        delta_w = np.random.normal(0, np.sqrt(delta_t))
        w[i] = w[i-1] + delta_w
    
    return w

def geometric_brownian_motion(S0, drift, volatility, n_steps, time_unit=1):
    w = wiener_process(n_steps, time_unit=time_unit, gbm=True)
    
    S = np.zeros(n_steps * time_unit)
    S[0] = S0
    
    delta_t = time_unit / n_steps
    for i in range(1, n_steps * time_unit):
        S[i] = S[i-1] * np.exp((drift - 0.5 * volatility**2) * delta_t + volatility * np.sqrt(delta_t) * (w[i] - w[i-1]))
    
    return S

def monte_carlo_simulation(n_steps, n_paths, time_unit=1):
    # if process is log-normal, taking a logarithm reverses the exponential transformation, turning it into a normal distribution
    paths = []
    
    variance = time_unit / n_steps
    stddev = np.sqrt(variance)

    for i in tqdm(range(n_paths), desc="Simulating paths"):
        paths.append(wiener_process(n_steps, time_unit=time_unit, gbm=True))
    
    paths_array = np.array(paths)
    
    df = pd.DataFrame(paths_array.T) 
    return df

def plot_paths(df):
    mean_path = df.mean(axis=1)
    stddev_path = df.std(axis=1)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Simulated Paths", "Distributions"))

    for i in range(df.shape[1]):
        fig.add_trace(go.Scatter(x=df.index, y=df[i], mode='lines', name=f'Path {i+1}'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=mean_path, mode='lines', name='Mean Path', line=dict(color='blue', width=2)), row=1, col=1)

    final_values = df.iloc[-1]
    fig.add_trace(go.Histogram(x=final_values, name='Final Values', opacity=0.75, nbinsx=1000), row=2, col=1)

    fig.update_layout(
        title=f'Monte Carlo Simulation | {df.shape[1]} paths | {stddev_path.iloc[0]:.2f} Standard Deviation',
        xaxis_title='Steps',
        yaxis_title='Process Value',
        template='plotly_dark'
    )

    fig.show()

    plt.figure(figsize=(8, 6))
    stats.probplot(np.log(final_values), dist="norm", plot=plt)
    plt.title('Q-Q Plot of Log-Transformed Final Values')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    S0 = 100        # Initial stock price
    mu = 0.05       # Drift (5% expected return)
    sigma = 0.2     # Volatility (20%)
    T = 1           # Time period (1 year)
    dt = 1/252      # Time step (daily for 252 trading days)
    N = int(T / dt) # Number of time steps

    # Simulate stock price path
    simulated_prices = geometric_brownian_motion(S0, mu, sigma, N, time_unit=1)

    import matplotlib.pyplot as plt

    # Plot the simulated stock price
    plt.figure(figsize=(10,6))
    plt.plot(simulated_prices)
    plt.title('Simulated Stock Price (Geometric Brownian Motion)')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.show()