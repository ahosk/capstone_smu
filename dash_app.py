import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash()

df_pred = pd.read_pickle(r'../Dashboard/dash_data_pred.pkl.gzip',compression='gzip')
df_true = pd.read_pickle(r'../Dashboard/dash_data_true.pkl.gzip',compression='gzip')



def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='S&P 500 Market Cap'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),  
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([ 
        dcc.Graph(id='tip-graph'),
        html.Label([
            "colorscale",
            dcc.Dropdown(
                id='colorscale-dropdown', clearable=False,
                value='bluyl', options=[
                    {'label': c, 'value': c}
                    for c in px.colors.named_colorscales()
                ])
        ]),
    ])
])

@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def display_table(dropdown_value):
    if dropdown_value is None:
        return generate_table(df_true)

    dff = df_true[df_true.ticker.str.contains('|'.join(dropdown_value))]
    return generate_table(dff)

if __name__ == '__main__':
    app.run_server(debug=True)