import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from spitfire.chemistry.flamelet import Flamelet
from spitfire.chemistry.mechanism import ChemicalMechanismSpec

m = ChemicalMechanismSpec(cantera_xml='heptane-liu-hewson-chen-pitsch-highT.xml', group_name='gas')
pressure = 101325.
air = m.stream(stp_air=True)
air.TP = 298., pressure
fuel = m.stream('TPY', (488., pressure, 'NXC7H16:1'))

flamelet_specs = {'mech_spec': m, 'pressure': pressure, 'oxy_stream': air, 'fuel_stream': fuel}


def get_data(cpoint, ccoeff, npts):
    z, dz = Flamelet._clustered_grid(npts, cpoint, ccoeff)
    dz = np.hstack([dz, 1. - z[-2]])
    return z, 1. / dz


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([dcc.Graph(id='graph-dz', className='four columns'),
              dcc.Graph(id='graph-T', className='four columns')]),
    html.Div([html.Div([html.Label('Cluster point'),
                        html.Div(id='cluster-point-text'),
                        dcc.Slider(
                            id='cluster-point-slider',
                            min=0.01,
                            max=0.99,
                            value=0.06,
                            step=0.01,
                            className='four columns'
                        ), ]),
              html.Div([html.Label('Cluster intensity'),
                        html.Div(id='cluster-intensity-text'),
                        dcc.Slider(
                            id='cluster-coeff-slider',
                            min=0.1,
                            max=10.,
                            value=4,
                            step=0.3,
                            className='four columns'
                        )]),
              html.Div([html.Label('Number of nodes'),
                        html.Div(id='npts-text'),
                        dcc.Slider(
                            id='npts-slider',
                            min=32,
                            max=192,
                            value=72,
                            step=8,
                            className='four columns'
                        )])]),
])


@app.callback(
    [Output('graph-dz', 'figure'),
     Output('graph-T', 'figure')],
    [Input('cluster-point-slider', 'value'),
     Input('cluster-coeff-slider', 'value'),
     Input('npts-slider', 'value')])
def update_figure(cpoint, ccoeff, npts):
    z, inv_dz = get_data(cpoint, ccoeff, npts)
    traces_dz = [dict(
        x=z,
        y=inv_dz,
        mode='lines+markers',
        opacity=0.7,
        line=dict(width=4, color='Black'),
        marker={
            'symbol': 'square',
            'size': 8,
            'color': 'RoyalBlue',
            'line': dict(width=1, color='Black')
        },
    )]
    fs = dict(flamelet_specs)
    fs.update(dict({'grid_points': npts, 'grid_cluster_intensity': ccoeff, 'grid_cluster_point': cpoint,
                    'initial_condition': 'equilibrium'}))
    f = Flamelet(**fs)
    T_eq = f.initial_temperature
    fs.update(dict({'initial_condition': 'Burke-Schumann'}))
    f = Flamelet(**fs)
    T_bs = f.initial_temperature
    traces_T = [dict(
        name='Equilibrium',
        x=z,
        y=T_eq,
        mode='lines+markers',
        opacity=0.7,
        line=dict(width=4, color='Black'),
        marker={
            'symbol': 'square',
            'size': 8,
            'color': 'RoyalBlue',
            'line': dict(width=1, color='Black')
        },
    ),
        dict(
            name='Burke-Schumann',
            x=z,
            y=T_bs,
            mode='lines+markers',
            opacity=0.7,
            line=dict(width=4, color='Black'),
            marker={
                'symbol': 'square',
                'size': 8,
                'color': 'Salmon',
                'line': dict(width=1, color='Black')
            },
        )
    ]

    return {
               'data': traces_dz,
               'layout': dict(
                   xaxis={'type': 'linear', 'title': 'mixture fraction', 'range': [0, 1]},
                   yaxis={'type': 'log', 'title': 'equivalent uniform grid points', 'range': [0, 4]},
                   margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                   legend={'x': 0, 'y': 1},
                   hovermode='closest',
                   transition={'duration': 1e-3},
               )
           }, \
           {
               'data': traces_T,
               'layout': dict(
                   xaxis={'type': 'linear', 'title': 'mixture fraction', 'range': [0, 1]},
                   yaxis={'type': 'linear', 'title': 'T (K)', 'range': [298, 3000]},
                   margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                   legend={'x': 0, 'y': 1},
                   hovermode='closest',
                   transition={'duration': 1e-3},
               )
           }


@app.callback(
    [Output('cluster-point-text', 'children'),
     Output('cluster-intensity-text', 'children'),
     Output('npts-text', 'children')],
    [Input('cluster-point-slider', 'value'),
     Input('cluster-coeff-slider', 'value'),
     Input('npts-slider', 'value')])
def update_figure(cpoint, ccoeff, npts):
    return f'{cpoint:.2f}', f'{ccoeff:.2f}', f'{npts}'


if __name__ == '__main__':
    app.run_server(debug=True)

