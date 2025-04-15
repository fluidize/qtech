import numpy as np
import plotly.graph_objects as go

def wiener_process(n_steps, time_unit=1, live_plot=False):
    y = []
    w = 0

    frames = []  # List to hold frames

    for i in range(n_steps):
        delta_t = time_unit / n_steps
        # in a wiener process, the variance of the increment delta_w is delta_t. in order to get the stddev, we sqrt
        # the variance of each step is: Var[ΔW]=Δt
        delta_w = np.random.normal(0, np.sqrt(delta_t)) #numpy requires stddev, not variance
        w += delta_w
        y.append(w)

        if live_plot:
            frames.append(go.Frame(
                data=[go.Scatter(x=list(range(len(y))), y=y, mode='lines')],
                name=str(i)
            ))

    if live_plot:
        fig = go.Figure(
            data=[go.Scatter(x=list(range(len(y))), y=y, mode='lines')],
            frames=frames
        )
        fig.update_layout(
            title='Wiener Process',
            xaxis_title='Steps',
            yaxis_title='Wiener Process Value',
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'showactive': False,
                'type': 'buttons',
                'xanchor': 'right',
                'yanchor': 'bottom'
            }],
        )

        fig.show()
    return y

if __name__ == "__main__":
    w = wiener_process(1000, time_unit=10, live_plot=True)
