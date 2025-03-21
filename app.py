from dash.dependencies import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc
from src import predictions, range_based, upload_train, date_based
from init import app, server
from src.mysql_connect import df

# Enhanced color palette
colors = {
    "primary": "#1E88E5",
    "secondary": "#00897B",
    "background": "#F5F7FA",
    "text": "#263238",
    "light_text": "#546E7A",
    "accent": "#FFC107",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "danger": "#F44336",
    "card_bg": "#FFFFFF",
}

# ---------------------- Callback start --------------------
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def update_content(pathname):
    if pathname == "/factors":
        return range_based.layout
    elif pathname == "/factors_date":
        return date_based.layout
    elif pathname == "/predictions":
        return predictions.layout
    elif pathname == "/":
        return index_page
    elif pathname == "/retrain":
        return upload_train.layout


@app.callback(Output("navbar", "children"), [Input("url", "pathname")])
def update_content(pathname):
    if pathname in ["/factors", "/factors_date", "/predictions", "/retrain", "/"]:
        return navbar
    return navbar


# ------------------- Callback end -----------------------

# Custom card style
card_style = {
    "backgroundColor": colors["card_bg"],
    "borderRadius": "8px",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "padding": "20px",
    "height": "100%",
}

# Left card with improved styling
left_card = html.Div(
    [
        html.H3(
            "Problem Statement",
            className="card-title mb-4",
            style={"color": colors["primary"]},
        ),
        html.P(
            "Wind energy plays an increasing role in the supply of energy world-wide. "
            "The energy output of a wind farm is highly dependent on the wind conditions present at its site. "
            "If the output can be predicted more accurately, energy suppliers can coordinate the collaborative "
            "production of different energy sources more efficiently to avoid costly overproduction.",
            className="mb-3",
            style={"lineHeight": "1.6", "color": colors["text"]},
        ),
        html.P(
            "Wind power or wind energy is the use of wind to provide mechanical power through wind turbines "
            "to turn electric generators and traditionally to do other work, like milling or pumping. Wind power "
            "is a sustainable and renewable energy, and has a much smaller impact on the environment compared to burning "
            "fossil fuels. Wind farms consist of many individual wind turbines, which are connected to the electric power "
            "transmission network.",
            className="mb-3",
            style={"lineHeight": "1.6", "color": colors["text"]},
        ),
        html.P(
            "Onshore wind is an inexpensive source of electric power, competitive with or in many places "
            "cheaper than coal or gas plants. Offshore wind is steadier and stronger than on land and offshore farms "
            "have less visual impact, but construction and maintenance costs are higher.",
            className="mb-4",
            style={"lineHeight": "1.6", "color": colors["text"]},
        ),
        dbc.Button(
            "Read More",
            color="primary",
            href="https://en.wikipedia.org/wiki/Wind_power",
            className="mt-2",
            style={
                "fontWeight": "500",
                "borderRadius": "4px",
                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                "transition": "all 0.3s ease",
            },
        ),
    ],
    style=card_style,
)

# Right card with improved styling
right_card = html.Div(
    [
        html.H3(
            "Turbine Status",
            className="card-title mb-4",
            style={"color": colors["primary"]},
        ),
        html.Div(
            [
                html.P(
                    "Wind Speed:",
                    className="font-weight-bold mb-1",
                    style={"color": colors["light_text"]},
                ),
                html.H5("{} m/s".format(round(df.iloc[5, 2], 3)), className="mb-3"),
            ]
        ),
        html.Div(
            [
                html.P(
                    "Wind Direction:",
                    className="font-weight-bold mb-1",
                    style={"color": colors["light_text"]},
                ),
                html.H5("{}°".format(round(df.iloc[5, 4], 3)), className="mb-3"),
            ]
        ),
        html.Div(
            [
                html.P(
                    "Power Output:",
                    className="font-weight-bold mb-1",
                    style={"color": colors["light_text"]},
                ),
                html.H5("{} kW".format(round(df.iloc[5, 1], 3)), className="mb-4"),
            ]
        ),
        html.Div(
            [
                html.H5("Status:", className="mb-2"),
                dbc.Button(
                    "Active",
                    color="success",
                    className="mr-3 mb-3",
                    style={
                        "fontWeight": "500",
                        "borderRadius": "4px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                        "transition": "all 0.3s ease",
                    },
                ),
                html.Div(className="mt-3"),
                dbc.Button(
                    "Detail Predictions",
                    color="warning",
                    href="/predictions",
                    className="mt-2",
                    style={
                        "fontWeight": "500",
                        "borderRadius": "4px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                        "transition": "all 0.3s ease",
                    },
                ),
            ]
        ),
    ],
    style=card_style,
)

# Improved info section
info = dbc.Row(
    [
        dbc.Col(left_card, md=8, className="mb-4"),
        dbc.Col(right_card, md=4, className="mb-4"),
    ],
    className="mt-5",
)

# --------------------- Improved Navbar ------------------------
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", active=True, href="/", className="px-3")),
        dbc.NavItem(dbc.NavLink("Predictions", href="/predictions", className="px-3")),
        dbc.NavItem(dbc.NavLink("Visualizations", href="/factors", className="px-3")),
        dbc.NavItem(dbc.NavLink("Retrain", href="/retrain", className="px-3")),
    ],
    brand="PEOWT",
    brand_href="/",
    color=colors["primary"],
    dark=True,
    className="",
    style={
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
    },
)

# --------------------- Improved Index page ---------------------
index_page = html.Div(
    [
        # Hero section with improved styling
        html.Div(
            style={
                "position": "relative",
                "height": "60vh",
                "overflow": "hidden",
                "marginBottom": "2rem",
            },
            children=[
                # Background image with improved styling
                html.Div(
                    style={
                        "backgroundImage": 'url("./assets/main.jpeg")',
                        "backgroundSize": "cover",
                        "backgroundPosition": "center",
                        "filter": "brightness(0.8)",
                        "height": "100%",
                        "width": "100%",
                    }
                ),
                # Overlay for better text readability
                html.Div(
                    style={
                        "position": "absolute",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0, 0, 0, 0.4)",
                    }
                ),
                # Content container
                html.Div(
                    style={
                        "position": "absolute",
                        "top": "50%",
                        "left": "50%",
                        "transform": "translate(-50%, -50%)",
                        "textAlign": "center",
                        "width": "80%",
                        "maxWidth": "1000px",
                        "color": "white",
                    },
                    children=[
                        # Logo
                        html.H1(
                            "PEOWT",
                            style={
                                "fontSize": "4rem",
                                "fontWeight": "700",
                                "marginBottom": "1rem",
                                "textShadow": "0 2px 4px rgba(0, 0, 0, 0.5)",
                            },
                        ),
                        # Description
                        html.H2(
                            "Predicting the energy output of wind turbines based on weather conditions",
                            style={
                                "fontSize": "1.5rem",
                                "fontWeight": "400",
                                "marginBottom": "2rem",
                                "textShadow": "0 2px 4px rgba(0, 0, 0, 0.5)",
                            },
                        ),
                        # Call to action button
                        dbc.Button(
                            "Learn More",
                            color="primary",
                            href="#info-section",
                            size="lg",
                            className="mr-2",
                            style={
                                "fontWeight": "500",
                                "borderRadius": "4px",
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
                                "padding": "0.75rem 1.5rem",
                                "transition": "all 0.3s ease",
                            },
                        ),
                        dbc.Button(
                            "View Predictions",
                            color="secondary",
                            href="/predictions",
                            size="lg",
                            outline=True,
                            className="ml-2",
                            style={
                                "fontWeight": "500",
                                "borderRadius": "4px",
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
                                "padding": "0.75rem 1.5rem",
                                "backgroundColor": "rgba(255, 255, 255, 0.1)",
                                "borderColor": "white",
                                "color": "white",
                                "transition": "all 0.3s ease",
                            },
                        ),
                    ],
                ),
            ],
        ),
        # Info section
        html.Div(
            id="info-section",
            className="container py-5",
            children=[
                html.H2(
                    "Wind Energy Prediction",
                    className="text-center mb-5",
                    style={"color": colors["primary"]},
                ),
                info,
                # Added features section
                html.Div(className="mt-5 pt-3"),
                html.H2(
                    "Key Features",
                    className="text-center mb-5",
                    style={"color": colors["primary"]},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        className="text-center mb-3",
                                        children=[
                                            html.I(
                                                className="fas fa-chart-line fa-3x",
                                                style={"color": colors["primary"]},
                                            )
                                        ],
                                    ),
                                    html.H4(
                                        "Real-time Predictions",
                                        className="text-center mb-3",
                                    ),
                                    html.P(
                                        "Get accurate power output predictions based on current and forecasted weather conditions.",
                                        className="text-center",
                                        style={"color": colors["light_text"]},
                                    ),
                                ],
                                style=card_style,
                            ),
                            md=4,
                            className="mb-4",
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        className="text-center mb-3",
                                        children=[
                                            html.I(
                                                className="fas fa-wind fa-3x",
                                                style={"color": colors["primary"]},
                                            )
                                        ],
                                    ),
                                    html.H4(
                                        "Wind Data Analysis",
                                        className="text-center mb-3",
                                    ),
                                    html.P(
                                        "Visualize relationships between wind parameters and energy production for optimal planning.",
                                        className="text-center",
                                        style={"color": colors["light_text"]},
                                    ),
                                ],
                                style=card_style,
                            ),
                            md=4,
                            className="mb-4",
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        className="text-center mb-3",
                                        children=[
                                            html.I(
                                                className="fas fa-cogs fa-3x",
                                                style={"color": colors["primary"]},
                                            )
                                        ],
                                    ),
                                    html.H4(
                                        "Model Retraining", className="text-center mb-3"
                                    ),
                                    html.P(
                                        "Continuously improve prediction accuracy by retraining models with new data inputs.",
                                        className="text-center",
                                        style={"color": colors["light_text"]},
                                    ),
                                ],
                                style=card_style,
                            ),
                            md=4,
                            className="mb-4",
                        ),
                    ]
                ),
            ],
        ),
        # Footer
        html.Footer(
            className="mt-5 py-4",
            style={"backgroundColor": colors["primary"], "color": "white"},
            children=[
                html.Div(
                    className="container text-center",
                    children=[
                        html.P(
                            "© 2025 PEOWT - Predicting Energy Output of Wind Turbines",
                            className="mb-2",
                        ),
                        html.P(
                            "Sustainable energy prediction for a greener future",
                            style={"fontSize": "0.9rem"},
                        ),
                    ],
                )
            ],
        ),
    ]
)

# Main app layout
app.layout = html.Div(
    style={"backgroundColor": colors["background"], "minHeight": "100vh"},
    children=[
        # Include FontAwesome for icons
        html.Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
        ),
        html.Div(id="navbar"),
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ],
)


if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, dev_tools_props_check=False)
