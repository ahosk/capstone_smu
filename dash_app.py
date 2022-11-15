import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from stockstats import StockDataFrame as Sdf
import dash_bootstrap_components as dbc
from dash import dash_table as dt
import yahoo_fin.stock_info as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd


# defining style color
colors = {"background": "#000000", "text": "#696969"}

df_true = pd.read_pickle(r'../Dashboard/dash_data_true.pkl.gzip',compression='gzip')
df_true['ticker'] = df_true['ticker'].str.replace(r'-US','')
ticker_list = df_true.ticker.unique()

df_pred = pd.read_pickle(r'../Dashboard/dash_data_pred.pkl.gzip',compression='gzip')
df_pred['ticker'] = df_pred['ticker'].str.replace(r'-US','')
df_pred = df_pred.rename(columns={"1":"POM_Probability"})

external_stylesheets = [dbc.themes.DARKLY]


# adding css
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.Div(
            [  # header Div
                dbc.Row(
                    [
                        dbc.Col(
                            html.Header(
                                [
                                    html.H1(
                                        "Stock Prediction Dashbord",
                                        style={
                                            "textAlign": "center",
                                            "color": colors["text"],
                                        },
                                    )
                                ]
                            )
                        )
                    ]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [  # Dropdown Div
                dbc.Row(
                    [
                        dbc.Col(  # Tickers
                            dcc.Dropdown(
                                id="stock_name",
                                options=[
                                    {
                                        "label": str(ticker_list[i]),
                                        "value": str(ticker_list[i]),
                                    }
                                    for i in range(len(ticker_list))
                                ],
                                searchable=True,
                                placeholder = 'Enter Stock Ticker',
                                multi=False,
                                style={"color": '#696969'}
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(  # Graph type
                            dcc.Dropdown(
                                id="chart",
                                options=[
                                    {"label": "line", "value": "Line"},
                                    {"label": "candlestick",
                                        "value": "Candlestick"},
                                    {"label": "Simple moving average",
                                        "value": "SMA"},
                                    {
                                        "label": "Exponential moving average",
                                        "value": "EMA",
                                    },
                                    {"label": "OHLC", "value": "OHLC"},
                                ],
                                placeholder = 'Select Chart Type',
                                multi=False,
                                style={"color": "#696969"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="quarter",
                                options=[
                                    {"label": "Q4 2019", "value": "12/27/2019"},
                                    {"label": "Q1 2020", "value": "4/3/2020"},
                                    {"label": "Q2 2020", "value": "7/2/2020"},
                                    {'label': "Q3 2020", "value": "BLANK"},
                                    {'label': "Q4 2020", "value": "BLANK"},
                                    {'label': "Q1 2021", "value": "BLANK"},
                                    {'label': "Q2 2021", "value": "BLANK"},
                                    {'label': "Q3 2021", "value": "BLANK"},
                                    {'label': "Q4 2021", "value": "BLANK"},
                                ],
                                placeholder='Select Prediction Quarter(s)',
                                multi=False,
                                style={"color": "#696969"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="plot_value",
                                options=[
                                    {"label": "Market Cap", "value": "Mkt_Cap"},
                                    {'label': 'EPS GR 3M', 'value':'EPS_GR_3M'},
                                    {'label': 'REV GR 3M', 'value':'REV_GR_3M'},
                                    {'label': 'DY 3M Rev', 'value':'DY_3M_Rev'},
                                    {'label': 'EVS 3M Rev', 'value':'EVS_3M_Rev'},
                                    {'label': 'PB', 'value':'PB_3M_Rev_REL'},
                                ],
                                placeholder='Select a Predicted Value',
                                multi=False,
                                style={"color": "#696969"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="plot_value2",
                                options=[
                                    {"label": "Predicted Forward EPS 3M", "value": "Pred_FWD_EPS_3M_GR"},
                                    {'label': 'Predicted Forward Revenue 3M', 'value':'Pred_FWD_REV_3M_GR'},
                                    {'label': 'Predicted FPE 3M', 'value':'Pred_FPE_3M_REV'},
                                    {'label': 'Predicted EVS 3M', 'value':'Pred_EVS_3M_REV'},
                                    {'label': 'Predicted DY 3M', 'value':'Pred_DY_3M_REV'},
                                    {'label': 'Predicted PB 3M', 'value':'Pred_PB_3M_REV'},
                                ],
                                placeholder='Select a Predicted Value',
                                multi=False,
                                style={"color": "#696969"},
                            ),
                            width={"size": 2},
                        ),
                        dbc.Col(  # button
                            dbc.Button(
                                "Plot",
                                id="submit-button-state",
                                className="mr-1",
                                n_clicks=1,
                            ),
                            width={"size": 2},
                        ),
                    ], justify='center',
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="live price",
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                }
                            )
                        ), 
                        ]
                    ),
                    dbc.Col(
                            dt.DataTable(
                                id="predictions",
                                style_table={"width": "auto"},
                                style_cell={
                                    "white_space": "normal",
                                    "height": "auto",
                                    "backgroundColor": '#272727',
                                    "color": "white",
                                    "font_size": "12px",
                                    'textAlign':'left',
                                    'padding':'5px',
                                    'fontWeight':'bold'
                                },
                                style_data={"border": "white"},
                                style_header={
                                    "backgroundColor": colors['background'],
                                    "border": "#4d4d4d",
                                    'textAlign': 'left',
                                    'fontWeight':'bold',
                                    'font_size':'16px',
                                    'padding':'5px'
                                },
                                columns=[{'id':c,'name':c} for c in df_pred.columns.values],
                                data=df_pred.to_dict('records'),
                            )
                        ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id='true_pred_graph',
                                config={
                                    "displaylogo":False,
                                    "modeBarButtonsToRemove":["pan2d", "lasso2d"],
                                }
                            ),
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="graph",
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                },
                            ),
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dt.DataTable(
                                id="info",
                                style_table={"height": "auto"},
                                style_cell={
                                    "white_space": "normal",
                                    "height": "auto",
                                    "backgroundColor": colors["background"],
                                    "color": "white",
                                    "font_size": "16px",
                                },
                                style_data={"border": "#4d4d4d"},
                                style_header={
                                    "backgroundColor": colors["background"],
                                    "fontWeight": "bold",
                                    "border": "#4d4d4d",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "center"}
                                    for c in ["attribute", "value"]
                                ],
                            ),
                            width={"size": 6, "offset": 3},
                        )
                    ]
                ),
            ]
        ),
    ],
)

# Callback main graph
@app.callback(
    # output
    Output("true_pred_graph", "figure"),
    # input
    [Input("submit-button-state", "n_clicks"),
    # state
    State("stock_name", "value"), State("chart", "value"),State('plot_value','value'),State('plot_value2','value')],
)
def true_pred_generator(n_clicks,ticker,chart_name,data_plot,data_plot2):
    df = df_true[df_true['ticker']==ticker]
    df['datepart'] = pd.to_datetime(df['Date'])
    df['monthyear'] = df.datepart.astype('datetime64[M]') 
    df2 = df_pred[df_pred['ticker']==ticker]
    df2['datepart'] = pd.to_datetime(df2['Date'])
    df2['monthyear'] = df2.datepart.astype('datetime64[M]') 

    charts = ['Line','Candlestick','SMA','OHLC','EMA']
    if chart_name in charts:
        true_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df['monthyear'], y=df[data_plot],name=data_plot,line_color='green'
                    )
                ],
            layout={
                    "height": 1000,
                    "title": ticker + ': ' + data_plot,
                    "showlegend": True,
                    "plot_bgcolor": colors["background"],
                    "paper_bgcolor": colors['background'],
                    "font": {"color": colors["text"]},
                }
        )
        true_fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    activecolor="blue",
                    bgcolor=colors["background"],
                    buttons=list(
                        [
                            dict(count=7, label="10D",step="day", stepmode="backward"
                            ),
                            dict(
                                count=15, label="15D", step="day", stepmode="backward"
                            ),
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=3, label="3m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="1y", step="year",stepmode="backward"
                            ),
                            dict(count=5, label="5y", step="year",stepmode="backward"
                            ),
                            dict(count=1, label="YTD",step="year", stepmode="todate"
                            ),
                            dict(step="all"
                            ),
                        ]
                    ),
                ),
            )
        true_fig.add_trace(
                go.Scatter(
                        x=df2['monthyear'], y=df2[data_plot2],name=data_plot2,line_color='red'
                    )
            )
            
    return true_fig


@app.callback(
    # output
    [Output("graph", "figure"), Output("live price", "figure")],
    # input
    [Input("submit-button-state", "n_clicks")],
    # state
    [State("stock_name", "value"), State("chart", "value")],
)
def graph_generator(n_clicks, ticker, chart_name):
        if n_clicks >= 1:  # Checking for user to click submit button

            # loading data
            start_date = datetime.now().date() - timedelta(days=5 * 365)
            end_data = datetime.now().date()
            df = yf.get_data(
                ticker, start_date=start_date, end_date=end_data, interval="1d"
            )
            stock = Sdf(df)

            # selecting graph type

            # line plot
            if chart_name == "Line":
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=list(df.index), y=list(df.close), fill="tozeroy", name="close"
                        )
                    ],
                    layout={
                        "height": 1000,
                        "title": ticker + ': ' + chart_name,
                        "showlegend": True,
                        "plot_bgcolor": colors["background"],
                        "paper_bgcolor": colors["background"],
                        "font": {"color": colors["text"]},
                    },
                )

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        activecolor="blue",
                        bgcolor=colors["background"],
                        buttons=list(
                            [
                                dict(count=7, label="10D",
                                    step="day", stepmode="backward"),
                                dict(
                                    count=15, label="15D", step="day", stepmode="backward"
                                ),
                                dict(
                                    count=1, label="1m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=3, label="3m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=6, label="6m", step="month", stepmode="backward"
                                ),
                                dict(count=1, label="1y", step="year",
                                    stepmode="backward"),
                                dict(count=5, label="5y", step="year",
                                    stepmode="backward"),
                                dict(count=1, label="YTD",
                                    step="year", stepmode="todate"),
                                dict(step="all"),
                            ]
                        ),
                    ),
                )

            # Candelstick
            if chart_name == "Candlestick":
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=list(df.index),
                            open=list(df.open),
                            high=list(df.high),
                            low=list(df.low),
                            close=list(df.close),
                            name="Candlestick",
                        )
                    ],
                    layout={
                        "height": 1000,
                        "title": ticker + ': ' + chart_name,
                        "showlegend": True,
                        "plot_bgcolor": colors["background"],
                        "paper_bgcolor": colors["background"],
                        "font": {"color": colors["text"]},
                    },
                )

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        activecolor="blue",
                        bgcolor=colors["background"],
                        buttons=list(
                            [
                                dict(count=7, label="10D",step="day", stepmode="backward"
                                ),
                                dict(
                                    count=15, label="15D", step="day", stepmode="backward"
                                ),
                                dict(
                                    count=1, label="1m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=3, label="3m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=6, label="6m", step="month", stepmode="backward"
                                ),
                                dict(count=1, label="1y", step="year",stepmode="backward"
                                ),
                                dict(count=5, label="5y", step="year",stepmode="backward"
                                ),
                                dict(count=1, label="YTD",step="year", stepmode="todate"
                                ),
                                dict(step="all"
                                ),
                            ]
                        ),
                    ),
                )

            # simple oving average
            if chart_name == "SMA":
                close_ma_10 = df.close.rolling(10).mean()
                close_ma_15 = df.close.rolling(15).mean()
                close_ma_30 = df.close.rolling(30).mean()
                close_ma_100 = df.close.rolling(100).mean()
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=list(close_ma_10.index), y=list(close_ma_10), name="10 Days"
                        ),
                        go.Scatter(
                            x=list(close_ma_15.index), y=list(close_ma_15), name="15 Days"
                        ),
                        go.Scatter(
                            x=list(close_ma_30.index), y=list(close_ma_15), name="30 Days"
                        ),
                        go.Scatter(
                            x=list(close_ma_100.index), y=list(close_ma_15), name="100 Days"
                        ),
                    ],
                    layout={
                        "height": 1000,
                        "title": ticker + ': ' + chart_name,
                        "showlegend": True,
                        "plot_bgcolor": colors["background"],
                        "paper_bgcolor": colors["background"],
                        "font": {"color": colors["text"]},
                    },
                )

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        activecolor="blue",
                        bgcolor=colors["background"],
                        buttons=list(
                            [
                                dict(count=7, label="10D",
                                    step="day", stepmode="backward"),
                                dict(
                                    count=15, label="15D", step="day", stepmode="backward"
                                ),
                                dict(
                                    count=1, label="1m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=3, label="3m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=6, label="6m", step="month", stepmode="backward"
                                ),
                                dict(count=1, label="1y", step="year",
                                    stepmode="backward"),
                                dict(count=5, label="5y", step="year",
                                    stepmode="backward"),
                                dict(count=1, label="YTD",
                                    step="year", stepmode="todate"),
                                dict(step="all"),
                            ]
                        ),
                    ),
                )

            # Open_high_low_close
            if chart_name == "OHLC":
                fig = go.Figure(
                    data=[
                        go.Ohlc(
                            x=df.index,
                            open=df.open,
                            high=df.high,
                            low=df.low,
                            close=df.close,
                        )
                    ],
                    layout={
                        "height": 1000,
                        "title": ticker + ': ' + chart_name,
                        "showlegend": True,
                        "plot_bgcolor": colors["background"],
                        "paper_bgcolor": colors["background"],
                        "font": {"color": colors["text"]},
                    },
                )

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        activecolor="blue",
                        bgcolor=colors["background"],
                        buttons=list(
                            [
                                dict(count=7, label="10D",
                                    step="day", stepmode="backward"),
                                dict(
                                    count=15, label="15D", step="day", stepmode="backward"
                                ),
                                dict(
                                    count=1, label="1m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=3, label="3m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=6, label="6m", step="month", stepmode="backward"
                                ),
                                dict(count=1, label="1y", step="year",
                                    stepmode="backward"),
                                dict(count=5, label="5y", step="year",
                                    stepmode="backward"),
                                dict(count=1, label="YTD",
                                    step="year", stepmode="todate"),
                                dict(step="all"),
                            ]
                        ),
                    ),
                )

            # Exponential moving average
            if chart_name == "EMA":
                close_ema_10 = df.close.ewm(span=10).mean()
                close_ema_15 = df.close.ewm(span=15).mean()
                close_ema_30 = df.close.ewm(span=30).mean()
                close_ema_100 = df.close.ewm(span=100).mean()
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=list(close_ema_10.index), y=list(close_ema_10), name="10 Days"
                        ),
                        go.Scatter(
                            x=list(close_ema_15.index), y=list(close_ema_15), name="15 Days"
                        ),
                        go.Scatter(
                            x=list(close_ema_30.index), y=list(close_ema_30), name="30 Days"
                        ),
                        go.Scatter(
                            x=list(close_ema_100.index),
                            y=list(close_ema_100),
                            name="100 Days",
                        ),
                    ],
                    layout={
                        "height": 1000,
                        "title": ticker + ': ' + chart_name,
                        "showlegend": True,
                        "plot_bgcolor": colors["background"],
                        "paper_bgcolor": colors["background"],
                        "font": {"color": colors["text"]},
                    },
                )

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        activecolor="blue",
                        bgcolor=colors["background"],
                        buttons=list(
                            [
                                dict(count=7, label="10D",
                                    step="day", stepmode="backward"),
                                dict(
                                    count=15, label="15D", step="day", stepmode="backward"
                                ),
                                dict(
                                    count=1, label="1m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=3, label="3m", step="month", stepmode="backward"
                                ),
                                dict(
                                    count=6, label="6m", step="month", stepmode="backward"
                                ),
                                dict(count=1, label="1y", step="year",
                                    stepmode="backward"),
                                dict(count=5, label="5y", step="year",
                                    stepmode="backward"),
                                dict(count=1, label="YTD",
                                    step="year", stepmode="todate"),
                                dict(step="all"),
                            ]
                        ),
                    ),
                )

        end_data = datetime.now().date()
        start_date = datetime.now().date() - timedelta(days=30)
        res_df = yf.get_data(
            ticker, start_date=start_date, end_date=end_data, interval="1d"
        )
        price = yf.get_live_price(ticker)
        prev_close = res_df.close.iloc[0]

        live_price = go.Figure(
            data=[
                go.Indicator(
                    domain={"x": [0, 1], "y": [0, 1]},
                    value=price,
                    mode="number+delta",
                    title={"text": "Price"},
                    delta={"reference": prev_close},
                )
            ],
            layout={
                "height": 300,
                "showlegend": True,
                "plot_bgcolor": colors["background"],
                "paper_bgcolor": colors["background"],
                "font": {"color": colors["text"]}
            },
        )
        return fig, live_price


@app.callback(
    [Output("info", "columns"), Output("info", "data")],
    [Input("submit-button-state", "n_clicks")],
    [State("stock_name", "value")],
)
def quotes_generator(n_clicks, ticker):
    # info table
    current_stock = yf.get_quote_table(ticker, dict_result=False)
    columns = [{"name": i, "id": i} for i in current_stock.columns]
    t_data = current_stock.to_dict("records")
    return columns, t_data

@app.callback(
    Output('predictions','data'),
    Input("quarter", "value"),Input('stock_name','value'),
    State('quarter','value')
)
def get_predictions(n_clicks,ticker,quarter):
    df = df_pred[(df_pred['ticker'] == ticker ) & (df_pred['Date']==quarter)].to_dict('records')
    return df

if __name__ == "__main__":
    app.run_server(debug=True)