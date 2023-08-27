from dash.dependencies import Input, Output
from dash import html, dcc
import dash_bootstrap_components as dbc
import predictions
import range_based
import date_based
import upload_train
from init import app, server
import mysql_connect as mc

server = server

# Color
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = mc.df

# ---------------------- Callback start --------------------


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')])
def update_content(pathname):
    if pathname == "/factors":
        return range_based.layout

    elif pathname == '/factors_date':
        return date_based.layout

    elif pathname == '/predictions':
        return predictions.layout

    elif pathname == '/':
        return index_page

    elif pathname == '/retrain':
        return upload_train.layout


@app.callback(
    Output('navbar', 'children'),
    [Input('url', 'pathname')])
def update_content(pathname):
    if pathname == '/factors':
        return navbar
    elif pathname == '/factors_date':
        return navbar

    elif pathname == '/predictions':
        return navbar

    elif pathname == '/retrain':
        return navbar

    elif pathname == '/':
        return navbar


# ------------------- Callback end -----------------------

app.layout = html.Div(children=[
    html.Div(id='navbar'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Left card
left_card = html.Div([
    html.H3("Problem Statement", className="card-title"),
    html.P(
        "Wind energy plays an increasing role in the supply of energy world-wide."
        "The energy output of a wind farm is highly dependent on the wind conditions present at its site."
        "If the output can be predicted more accurately, energy suppliers can coordinate the collaborative "
        "production of different energy sources more efficiently to avoid costly overproduction."),
    html.P(
        "Wind power or wind energy is the use of wind to provide the mechanical power through wind turbines"
        "to turn electric generators and traditionally to do other work, like milling or pumping. Wind power "
        "is a sustainable and renewable energy, and has a much smaller impact on the environment compared to burning "
        "fossil fuels. Wind farms consist of many individual wind turbines, which are connected to the electric power "
        "transmission network. Onshore wind is an inexpensive source of electric power, competitive with or in many places "
        "cheaper than coal or gas plants. Onshore wind farms also have an impact on the landscape, as "
        "typically they need to be spread over more land than other power stations and need to be built in wild "
        "and rural areas, which can lead to industrialization of the countryside and habitat loss. Offshore "
        "wind is steadier and stronger than on land and offshore farms have less visual impact, but construction "
        "and maintenance costs are higher. Small onshore wind farms can feed some energy into the grid or provide "
        "electric power to isolated off-grid locations."),

    dbc.Button("Read More", color="primary", href="")
])

# Right card
right_card = html.Div(
    [
        html.H3("Alerts", className="card-title"),
        html.P("Wind Speed: {} m/s".format(round(df.iloc[5, 2], 3))),
        html.P("Wind Driection: {}°".format(round(df.iloc[5, 4], 3))),
        html.P("Power Output: {} kw".format(round(df.iloc[5, 1], 3))),

        html.H5("Status: "),
        dbc.Button("Active", color="success"),
        html.Br(),
        html.Br(),
        dbc.Button("Deatil Predictions", color="warning", href="/predictions")
    ]
)

# info
info = dbc.Row([
    dbc.Col(left_card, width=8),
    dbc.Col(width=1),
    dbc.Col(right_card, width=3)
])


# --------------------- Navbar ------------------------
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", active=True, href="/")),
        dbc.NavItem(dbc.NavLink("Predictions", href="/predictions")),
        dbc.NavItem(dbc.NavLink("Visualizations", href="/factors")),
        dbc.NavItem(dbc.NavLink("Retrain", href="/retrain")),
    ],
    brand="PEOWT",
    brand_href="/",
    color='dark',
    dark=True
)

# --------------------- Index page ---------------------
index_page = html.Div(
    style={'position': 'absolute', 'height': '100%'},
    children=[
        # background image
        html.Img(
            src="./assets/main.jpeg",
            style={'width': '100%', 'opacity': '0.6', 'object-fit': 'cover'},
        ),

        # logo
        html.H1(
            children="PEOWT",
            style={
                'color': colors['background'],
                'textAlign':'center', 'position':'absolute',
                'top':'35%',
                'left':'50%',
                'transform':'translate(-50%, -50%)',
                'fontSize': '4vw',
                'textDecoration':'underline',
            }
        ),

        # description
        html.H1(
            children='Predicting the energy output of wind turbine based on weather condition',
            style={
                'color': 'black',
                'textAlign': 'center',
                'position': 'absolute',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'fontSize': '2vw',
            }
        ),

        html.P(
            dbc.Button("Learn more", color="primary", href="#", className="lead",
                       style={
                           'color': 'black',
                           'textAlign': 'center',
                           'position': 'absolute',
                           'top': '60%',
                           'left': '50%',
                           'transform': 'translate(-50%, -50%)'
                       })
        ),

        html.Br(),
        html.Br(),
        html.Br(),

        html.Div(className="container", children=[
            info,
            html.Br(),
            html.Br(),
        ])
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
