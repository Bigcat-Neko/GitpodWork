import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import requests
import pandas as pd

# Change API_BASE to your API endpoint
API_BASE = "http://127.0.0.1:8080"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("NEKO AI Trading Dashboard"),
    dbc.Row([
        dbc.Col(html.Div(id='active-trades'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='historical-trades'), width=12)
    ]),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)
])

@app.callback(
    [Output('active-trades', 'children'),
     Output('historical-trades', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch active trades
    try:
        active_response = requests.get(f"{API_BASE}/active_trades", timeout=5).json()
        if active_response:
            active_df = pd.DataFrame.from_dict(active_response, orient="index")
            active_table = dbc.Table.from_dataframe(active_df, striped=True, bordered=True, hover=True)
        else:
            active_table = "No active trades."
    except Exception as e:
        active_table = f"Error fetching active trades: {e}"
    
    # Fetch historical trades
    try:
        hist_response = requests.get(f"{API_BASE}/historical_trades", timeout=5).json()
        if hist_response and "historical_trades" in hist_response:
            hist_df = pd.DataFrame(hist_response["historical_trades"])
            hist_table = dbc.Table.from_dataframe(hist_df, striped=True, bordered=True, hover=True)
        else:
            hist_table = "No historical trades."
    except Exception as e:
        hist_table = f"Error fetching historical trades: {e}"

    return active_table, hist_table

if __name__ == '__main__':
    app.run_server(debug=True)
