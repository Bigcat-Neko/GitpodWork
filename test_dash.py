import dash
from dash import html

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div("Hello, Dash is working!")

if __name__ == '__main__':
    app.run_server(debug=True, host="127.0.0.1", port=8050)
