from dash import html, dcc
import dash
import numpy as np
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime as dt
from init import app
from src import mysql_connect

colors = {"background": "#111111", "text": "#7FDBFF"}


# Filter data based on two dates with proper date handling
def filter_data_based_on_dates(date1, date2, df):
    """Filter dataframe between two dates (inclusive), handling various date formats and errors."""
    try:
        start_date = pd.to_datetime(date1)
        end_date = pd.to_datetime(date2)

        # Find a suitable date column (more robust)
        date_cols = [
            col
            for col in df.columns
            if pd.api.types.is_datetime64_dtype(
                pd.to_datetime(df[col], errors="coerce")
            )
            or pd.api.types.is_string_dtype(df[col])
        ]
        if not date_cols:
            print("Error: No suitable date column found in the DataFrame.")
            return pd.DataFrame()  # Return empty DataFrame if no date column is found

        date_column = date_cols[0]  # Use the first suitable column

        # Convert to datetime if necessary
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(
                df[date_column], infer_datetime_format=True, errors="coerce"
            )

        # Remove rows with NaT (Not a Time) values after conversion
        df.dropna(subset=[date_column], inplace=True)

        filtered_df = df[
            (df[date_column] >= start_date) & (df[date_column] <= end_date)
        ]
        return filtered_df
    except (ValueError, KeyError) as e:
        print(f"Error filtering dates: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


# Load data into a dataframe
def load_data():
    """Load and prepare data, handling potential errors."""
    try:
        try:
            df = mysql_connect.df
        except Exception as e:
            print(f"Error connecting to MySQL: {e}. Falling back to CSV.")
            url = "./data.csv"
            df = pd.read_csv(url)

        # Find and convert date column (more robust)
        date_cols = [
            col
            for col in df.columns
            if pd.api.types.is_datetime64_dtype(
                pd.to_datetime(df[col], errors="coerce")
            )
            or pd.api.types.is_string_dtype(df[col])
        ]
        if date_cols:
            date_column = date_cols[0]
            if not pd.api.types.is_datetime64_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(
                    df[date_column], infer_datetime_format=True, errors="coerce"
                )
            df.dropna(subset=[date_column], inplace=True)  # Remove rows with NaT values

        return df
    except Exception as e:
        print(f"Critical error loading data: {e}")
        return pd.DataFrame(
            columns=[
                "Date_Time",
                "Theoretical_Power_Curve_kilowatt_hour",
                "LV_ActivePower_kW",
                "WindSpeed_m_per_s",
                "Wind_Direction_degree",
            ]
        )


# Load data
df = load_data()


# Create traces
def create_total_figure(df):
    """Create power output visualization for date range"""
    # Handle empty dataframe
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date/Time"],
            y=df["Theoretical_Power_Curve (KWh)"],
            mode="lines",
            name="Theoretical Power Curve (KWh)",
            marker=dict(symbol="circle", color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date/Time"],
            y=df["LV ActivePower (kW)"],
            mode="lines",
            name="LV Active Power (kW)",
            opacity=0.7,
            marker=dict(symbol="triangle-up-dot", color="red"),
        )
    )

    # Calculate and display average values
    avg_theoretical = df["Theoretical_Power_Curve (KWh)"].mean()
    avg_active = df["LV ActivePower (kW)"].mean()

    fig.add_shape(
        type="line",
        x0=df["Date/Time"].min(),
        y0=avg_theoretical,
        x1=df["Date/Time"].max(),
        y1=avg_theoretical,
        line=dict(color="green", width=2, dash="dash"),
    )

    fig.add_shape(
        type="line",
        x0=df["Date/Time"].min(),
        y0=avg_active,
        x1=df["Date/Time"].max(),
        y1=avg_active,
        line=dict(color="red", width=2, dash="dash"),
    )

    # Add annotations for average values
    fig.add_annotation(
        x=df["Date/Time"].iloc[len(df) // 4],
        y=avg_theoretical,
        text=f"Avg Theoretical: {avg_theoretical:.2f} KWh",
        showarrow=True,
        arrowhead=1,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="green",
        borderwidth=1,
    )

    fig.add_annotation(
        x=df["Date/Time"].iloc[len(df) // 2],
        y=avg_active,
        text=f"Avg Active: {avg_active:.2f} kW",
        showarrow=True,
        arrowhead=1,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
    )

    fig.update_layout(
        paper_bgcolor="#AFEEEE",
        font_color=colors["background"],
        title="Theoretical vs. Actual Power Output Over Time",
        xaxis_title="Date/Time",
        yaxis_title="Power (kW/KWh)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2,
        ),
    )
    return fig


def create_wind_speed_total(df):
    """Create wind speed and direction visualization for date range"""
    # Handle empty dataframe
    if df.empty:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "polar"}, {"type": "xy"}]])
        fig.add_annotation(
            text="No data available for selected period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        subplot_titles=("Wind Direction vs Speed", "Wind Speed Distribution"),
    )

    fig.add_trace(
        go.Scatterpolargl(
            r=df["Wind Speed (m/s)"],
            theta=df["Wind Direction (°)"],
            name="Wind Speed",
            marker=dict(
                size=4,
                color=df["Wind Speed (m/s)"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Wind Speed (m/s)", x=0.45),
            ),
            mode="markers",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df["Wind Speed (m/s)"],
            nbinsx=26,
            name="Wind Speed Distribution",
            marker_color="mediumseagreen",
        ),
        row=1,
        col=2,
    )

    # Calculate wind speed statistics
    avg_wind = df["Wind Speed (m/s)"].mean()
    max_wind = df["Wind Speed (m/s)"].max()

    # Add mean wind speed line
    fig.add_vline(
        x=avg_wind, line_width=2, line_dash="dash", line_color="red", row=1, col=2
    )

    # Add annotations
    fig.add_annotation(
        x=avg_wind,
        y=0.95,
        text=f"Mean: {avg_wind:.2f} m/s",
        showarrow=False,
        xref="x2",
        yref="paper",
        bgcolor="rgba(255,255,255,0.7)",
        row=1,
        col=2,
    )

    # Calculate predominant wind direction
    wind_dir_counts = (
        df["Wind Direction (°)"].apply(lambda x: (x // 45) * 45).value_counts()
    )
    predominant_dir = wind_dir_counts.idxmax()

    fig.update_layout(
        title=f"Wind Analysis (Max: {max_wind:.1f} m/s, Avg: {avg_wind:.1f} m/s)",
        paper_bgcolor="#AFEEEE",
        height=500,
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                direction="clockwise",
            )
        ),
    )

    # Add text annotation for predominant direction
    fig.add_annotation(
        text=f"Predominant direction: {predominant_dir}° ({['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int(predominant_dir//45 % 8)]})",
        xref="paper",
        yref="paper",
        x=0.25,
        y=0,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.7)",
    )

    return fig


def create_power_correlation_figure(df):
    """Create correlation analysis between wind speed and power output"""
    # Handle empty dataframe
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Create scatter plot with color representing power output
    fig.add_trace(
        go.Scatter(
            x=df["Wind Speed (m/s)"],
            y=df["LV ActivePower (kW)"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["LV ActivePower (kW)"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Power Output (kW)"),
            ),
            name="Power vs Wind Speed",
        )
    )

    # Add best fit line if there are enough data points
    if len(df) > 2:
        # Simple linear regression
        x = df["Wind Speed (m/s)"]
        y = df["LV ActivePower (kW)"]

        # Calculate regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        # Add regression line
        x_range = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode="lines",
                name=f"Best Fit (y = {z[0]:.2f}x + {z[1]:.2f})",
                line=dict(color="red", width=2),
            )
        )

        # Calculate correlation coefficient
        correlation = df["Wind Speed (m/s)"].corr(df["LV ActivePower (kW)"])

        # Add correlation annotation
        fig.add_annotation(
            x=0.95,
            y=0.05,
            xref="paper",
            yref="paper",
            text=f"Correlation: {correlation:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )

    fig.update_layout(
        title="Wind Speed vs Power Output Correlation",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Power Output (kW)",
        paper_bgcolor="#AFEEEE",
        height=500,
        hovermode="closest",
    )

    return fig


# Layout components
def create_switch_button():
    return html.Div(
        [
            dbc.Button("Date Based", color="warning", href="/show_factors_date"),
        ],
        style={"marginLeft": "50px"},
    )


def create_title():
    return html.Div(
        [
            html.H4(
                children="Range Based Visualization",
                style={"color": "black", "marginLeft": "60px"},
            ),
        ]
    )


def create_date_picker():
    initial_start_date = dt(2018, 1, 1)
    initial_end_date = dt(2018, 1, 31)

    if not df.empty and "date" in df.columns:
        # Set initial dates to first and last date in dataset if available
        try:
            min_date = df["date"].min()
            max_date = df["date"].max()
            if not pd.isna(min_date) and not pd.isna(max_date):
                initial_start_date = min_date
                initial_end_date = min_date + pd.Timedelta(days=30)
                if initial_end_date > max_date:
                    initial_end_date = max_date
        except:
            pass

    return html.Div(
        [
            html.Div(
                children=[
                    "Date Range: ",
                    dcc.DatePickerRange(
                        id="selection_based_on_dates",
                        min_date_allowed=dt(2018, 1, 1),
                        max_date_allowed=dt(2018, 12, 31),
                        start_date_placeholder_text="Start Date",
                        end_date_placeholder_text="End Date",
                        display_format="D/M/Y",
                        month_format="MMM Do, YY",
                        with_portal=True,
                        start_date=initial_start_date,
                        end_date=initial_end_date,
                    ),
                ]
            ),
        ]
    )


def create_filters():
    return html.Div(
        [
            html.Hr(),
            html.H5("Additional Filters", style={"marginLeft": "15px"}),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Wind Speed Range (m/s):", style={"marginLeft": "15px"}
                            ),
                            dcc.RangeSlider(
                                id="wind_speed_filter",
                                min=0,
                                max=25,
                                step=0.5,
                                marks={i: str(i) for i in range(0, 26, 5)},
                                value=[0, 25],
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Power Output Range (kW):", style={"marginLeft": "15px"}
                            ),
                            dcc.RangeSlider(
                                id="power_filter",
                                min=0,
                                max=2000,  # Adjust based on your data
                                step=100,
                                marks={i: str(i) for i in range(0, 2001, 500)},
                                value=[0, 2000],
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            html.Div(
                style={
                    "marginTop": "15px",
                    "marginBottom": "15px",
                    "textAlign": "center",
                },
                children=[
                    dbc.Button(
                        "Apply Filters",
                        id="apply_filters_button",
                        color="primary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Reset Filters", id="reset_filters_button", color="secondary"
                    ),
                ],
            ),
        ]
    )


# Layout components
switch = create_switch_button()
topic = create_title()
datePick = create_date_picker()
rangeBased = dbc.Row(
    [dbc.Col(datePick, width=4), dbc.Col(topic, width=6), dbc.Col(switch, width=2)]
)

# App layout for this page
layout = html.Div(
    children=[
        html.Div(
            className="container",
            children=[
                html.Br(),
                html.H1(
                    children="Wind Turbine Power Output Dashboard",
                    style={
                        "textAlign": "center",
                        "color": colors["background"],
                        "font-family": "Arial",
                    },
                ),
                html.Br(),
                ############### Range Based ##########
                rangeBased,
                create_filters(),
                # Store current filter state
                dcc.Store(
                    id="filter_state",
                    data={
                        "wind_min": 0,
                        "wind_max": 25,
                        "power_min": 0,
                        "power_max": 2000,
                    },
                ),
            ],
        ),
        html.Div(
            id="loading_indicator",
            children=[
                dbc.Spinner(
                    fullscreen=False,
                    color="primary",
                    type="grow",
                    spinnerClassName="spinner",
                )
            ],
        ),
        html.Div(
            children=[
                # Graphs
                html.Br(),
                html.Div(id="output_visualization_total"),
                html.Div(id="output_total_windspeed"),
                html.Div(id="output_correlation"),
            ]
        ),
        html.Div(
            id="range_stats_section", className="container", style={"marginTop": "20px"}
        ),
        html.Br(),
        html.Footer(
            html.P(
                "Wind Turbine Power Analytics Dashboard",
                style={"textAlign": "center", "color": colors["background"]},
            )
        ),
    ]
)


# Callbacks
@app.callback(
    [Output("filter_state", "data")],
    [
        Input("apply_filters_button", "n_clicks"),
        Input("reset_filters_button", "n_clicks"),
    ],
    [
        State("wind_speed_filter", "value"),
        State("power_filter", "value"),
        State("filter_state", "data"),
    ],
)
def update_filters(apply_clicks, reset_clicks, wind_range, power_range, current_state):
    ctx = dash.callback_context

    if not ctx.triggered:
        # No button clicked, return current state
        return [current_state]

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "reset_filters_button":
        # Reset to defaults
        return [{"wind_min": 0, "wind_max": 25, "power_min": 0, "power_max": 2000}]

    elif button_id == "apply_filters_button":
        # Update with new filter values
        return [
            {
                "wind_min": wind_range[0],
                "wind_max": wind_range[1],
                "power_min": power_range[0],
                "power_max": power_range[1],
            }
        ]

    # Default return current state
    return [current_state]


@app.callback(
    [Output("wind_speed_filter", "value"), Output("power_filter", "value")],
    [Input("filter_state", "data")],
)
def sync_sliders_to_state(filter_state):
    return [
        [filter_state["wind_min"], filter_state["wind_max"]],
        [filter_state["power_min"], filter_state["power_max"]],
    ]


@app.callback(
    [
        Output("output_visualization_total", "children"),
        Output("output_total_windspeed", "children"),
        Output("output_correlation", "children"),
        Output("range_stats_section", "children"),
    ],
    [
        Input("selection_based_on_dates", "start_date"),
        Input("selection_based_on_dates", "end_date"),
        Input("filter_state", "data"),
    ],
)
def update_all_visualizations(start_date, end_date, filter_state):
    try:
        if start_date is None or end_date is None:
            # Handle None dates
            return [
                html.Div("Please select a date range"),
                html.Div(),
                html.Div(),
                html.Div(),
            ]

        # First get data for the date range
        date_filtered_data = filter_data_based_on_dates(start_date, end_date, df)

        # Check if date_filtered_data is empty
        if date_filtered_data.empty:
            return [
                html.Div("No data available for the selected date range"),
                html.Div(),
                html.Div(),
                html.Div("No data available for the selected date range"),
            ]

        # Then apply additional filters
        wind_min = filter_state["wind_min"]
        wind_max = filter_state["wind_max"]
        power_min = filter_state["power_min"]
        power_max = filter_state["power_max"]

        # Handle missing wind speed data
        filtered_data = date_filtered_data.dropna(
            subset=["Wind Speed (m/s)"]
        )  # Remove rows with missing wind speed
        filtered_data["Wind Speed (m/s)"].fillna(
            filtered_data["Wind Speed (m/s)"].mean(), inplace=True
        )  # Fill with mean

        # Apply other filters (power)
        filtered_data = filtered_data[
            (filtered_data["Wind Speed (m/s)"] >= wind_min)
            & (filtered_data["Wind Speed (m/s)"] <= wind_max)
            & (filtered_data["LV ActivePower (kW)"] >= power_min)
            & (filtered_data["LV ActivePower (kW)"] <= power_max)
        ]

        # Generate visualizations
        try:
            power_fig = create_total_figure(filtered_data)
            wind_fig = create_wind_speed_total(filtered_data)
            corr_fig = create_power_correlation_figure(filtered_data)
        except KeyError as e:
            print(f"Error creating figures: Missing column {e}")
            return [
                html.Div(f"Error: Missing data column: {e}"),
                html.Div(),
                html.Div(),
                html.Div(),
            ]
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return [
                html.Div(f"An unexpected error occurred: {e}"),
                html.Div(),
                html.Div(),
                html.Div(),
            ]

        # Create stats
        if filtered_data.empty:
            stats_section = html.Div(
                "No data available for the selected period and filters"
            )
        else:
            # Calculate key statistics
            avg_power = filtered_data["LV ActivePower (kW)"].mean()
            max_power = filtered_data["LV ActivePower (kW)"].max()
            avg_wind = filtered_data["Wind Speed (m/s)"].mean()
            max_wind = filtered_data["Wind Speed (m/s)"].max()
            efficiency = (
                (
                    filtered_data["LV ActivePower (kW)"].sum()
                    / filtered_data["Theoretical_Power_Curve (KWh)"].sum()
                    * 100
                )
                if filtered_data["Theoretical_Power_Curve (KWh)"].sum() > 0
                else 0
            )

            # Create summary text
            summary_text = f"Analysis period: {pd.to_datetime(start_date).strftime('%d %b %Y')} to {pd.to_datetime(end_date).strftime('%d %b %Y')}"
            if wind_min > 0 or wind_max < 25 or power_min > 0 or power_max < 2000:
                summary_text += f" (Filtered: Wind {wind_min}-{wind_max} m/s, Power {power_min}-{power_max} kW)"

            # Create stats cards
            stats_row = html.Div(
                [
                    html.H5(
                        summary_text,
                        style={"textAlign": "center", "marginBottom": "15px"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Average Power"),
                                        dbc.CardBody(f"{avg_power:.2f} kW"),
                                    ],
                                    color="primary",
                                    outline=True,
                                ),
                                width=2,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Max Power"),
                                        dbc.CardBody(f"{max_power:.2f} kW"),
                                    ],
                                    color="danger",
                                    outline=True,
                                ),
                                width=2,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Avg Wind Speed"),
                                        dbc.CardBody(f"{avg_wind:.2f} m/s"),
                                    ],
                                    color="info",
                                    outline=True,
                                ),
                                width=2,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Max Wind Speed"),
                                        dbc.CardBody(f"{max_wind:.2f} m/s"),
                                    ],
                                    color="warning",
                                    outline=True,
                                ),
                                width=2,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Efficiency"),
                                        dbc.CardBody(f"{efficiency:.2f}%"),
                                    ],
                                    color="success",
                                    outline=True,
                                ),
                                width=2,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Data Points"),
                                        dbc.CardBody(f"{len(filtered_data)}"),
                                    ],
                                    color="secondary",
                                    outline=True,
                                ),
                                width=2,
                            ),
                        ]
                    ),
                ]
            )

        # Return all components
        return [
            dcc.Graph(id="total_graph", figure=power_fig),
            dcc.Graph(id="total_windspeed_graph", figure=wind_fig),
            dcc.Graph(id="correlation_graph", figure=corr_fig),
            stats_row if not filtered_data.empty else stats_section,
        ]

    except Exception as e:
        import traceback

        print(f"Error in update_all_visualizations: {e}")
        print(traceback.format_exc())

        # Return error message to the UI
        error_div = html.Div(
            f"An error occurred while updating the visualizations: {str(e)}",
            style={"color": "red", "margin": "20px"},
        )
        return [error_div, error_div, error_div, error_div]
