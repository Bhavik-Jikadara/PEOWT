# import modules
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime as dt
from init import app
from src import mysql_connect

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Filter data based on dates with proper date handling
def filter_data_based_on_dates(date1, date2, df):
    """Filter dataframe between two dates (inclusive)"""
    try:
        start_date = pd.to_datetime(date1)
        end_date = pd.to_datetime(date2)
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        return filtered_df
    except Exception as e:
        print(f"Error filtering dates: {e}")
        return pd.DataFrame()  # Return empty dataframe on error

# Filter data based on hours of a particular day
def filter_data_based_on_hours(date_str, from_hour, to_hour, df):
    """Filter dataframe for a specific date and time range"""
    try:
        # Convert to datetime properly
        date = pd.to_datetime(date_str.split()[0])
        date_str = date.strftime('%Y-%m-%d')
        
        # Filter by date
        data = df[df['date'] == date_str]
        
        # Filter by hour range
        hour_list = [str(hour).zfill(2) for hour in range(from_hour, to_hour)]
        return data[data.hour.isin(hour_list)]
    except Exception as e:
        print(f"Error filtering hours: {e}")
        return pd.DataFrame()  # Return empty dataframe on error

# Create marks dictionary for slider
def get_marks():
    """Generate time markers for the range slider"""
    return {val: f"{val:02d}:00" for val in range(25)}

# Load data into a dataframe
def load_data():
    """Load and prepare data"""
    try:
        # First try to get data from MySQL
        try:
            df = mysql_connect.df
        except:
            # Fall back to CSV if MySQL connection fails
            url = './data.csv'
            df = pd.read_csv(url)
        
        # Ensure date is in datetime format
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty dataframe with expected columns to prevent crashes
        return pd.DataFrame(columns=['date', 'time', 'hour', 
                                    'Theoretical_Power_Curve (KWh)', 
                                    'LV ActivePower (kW)',
                                    'Wind Speed (m/s)', 
                                    'Wind Direction (°)'])

# Load data
df = load_data()

# Graph functions
def create_daily_figure(df):
    """Create power output visualization"""
    # Handle empty dataframe
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected period",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['Theoretical_Power_Curve (KWh)'],
        mode='markers+lines',
        name='Theoretical Power Curve (KWh)',
        marker=dict(symbol="circle", color="green")))

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['LV ActivePower (kW)'],
        mode='markers+lines',
        name='LV Active Power (kW)',
        opacity=0.7,
        marker=dict(symbol="circle", color="red")))

    fig.update_layout(
        paper_bgcolor='#AFEEEE', 
        font_color=colors['background'],
        title='Variation in Theoretical Power and LV Active Power',
        xaxis_title="Time",
        yaxis_title="Power (kW/KWh)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        ),
        hovermode="x unified"
    )
    return fig

def create_wind_speed_daily(df):
    """Create wind speed and direction visualization"""
    # Handle empty dataframe
    if df.empty:
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}, {'type': 'xy'}]])
        fig.add_annotation(text="No data available for selected period",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
        
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type': 'polar'}, {'type': 'xy'}]],
                        subplot_titles=("Wind Direction vs Speed", "Wind Speed Distribution"))

    fig.add_trace(go.Scatterpolargl(
          r=df['Wind Speed (m/s)'],
          theta=df['Wind Direction (°)'],
          name="Wind Speed",
          marker=dict(size=4, color="darkorange"),
          mode="markers"
        ),
        row=1,
        col=1)

    fig.add_trace(go.Histogram(
        x=df['Wind Speed (m/s)'],
        nbinsx=26,
        name='Wind Speed',
        marker_color='darkorange'
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title='Wind Speed and Direction Analysis',
        paper_bgcolor='#AFEEEE',
        height=500,
        )
    
    # Add mean wind speed line
    if len(df) > 0:
        mean_wind_speed = df['Wind Speed (m/s)'].mean()
        fig.add_vline(x=mean_wind_speed, line_width=2, line_dash="dash", 
                    line_color="red", row=1, col=2)
        fig.add_annotation(
            x=mean_wind_speed, y=0.95, 
            text=f"Mean: {mean_wind_speed:.2f} m/s",
            showarrow=False, 
            xref="x2", yref="paper",
            bgcolor="rgba(255,255,255,0.7)",
            row=1, col=2
        )
        
    return fig

# Layout components
def create_switch_button():
    return html.Div(
        [
            dbc.Button("Range Based", color="warning", href="/factors"),
        ],
        style={"marginLeft": '40px'}
    )

def create_title():
    return html.Div(
        [
            html.H4(
                children='Date Based Visualization',
                style={
                    'color': 'black',
                    'marginLeft': '65px'
                }),
        ]
    )

def create_date_picker():
    initial_date = dt(2018, 1, 1)
    if not df.empty and 'date' in df.columns:
        # Set initial date to first date in dataset if available
        try:
            min_date = df['date'].min()
            if not pd.isna(min_date):
                initial_date = min_date
        except:
            pass
            
    return html.Div(
        [
            html.Div(style={'margin-left': '15px'}, children=[
                "Select Date: ",
                dcc.DatePickerSingle(
                    style={"margin-left": "15px"},
                    id='selection_based_on_hours',
                    min_date_allowed=dt(2018, 1, 1),
                    max_date_allowed=dt(2018, 12, 31),
                    display_format='D/M/Y',
                    month_format='MMM Do, YY',
                    with_portal=True,
                    date=str(initial_date)
                )]),
        ]
    )

# Build layout
switch1 = create_switch_button()
topic1 = create_title()
datePick1 = create_date_picker()
dateBased = dbc.Row([dbc.Col(datePick1, width=4), dbc.Col(topic1, width=6), dbc.Col(switch1, width=2)])

layout = html.Div(children=[
    html.Div(className="container", children=[
        html.Br(),
        html.H1(
            children='Wind Turbine Power Output Dashboard',
            style={
                'textAlign': 'center',
                'color': colors['background'],
                'font-family': 'Arial',
            }
        ),
        html.Br(),
        dateBased,
        html.Br(),
        html.Div(style={'margin-left': '15px'}, children=[
            "Select Time Range (in 24-hour format): ",
            dcc.RangeSlider(
                id='time_range',
                min=0,
                max=24,
                allowCross=False,
                step=1,
                marks=get_marks(),
                value=[0, 24],
                tooltip={"placement": "bottom", "always_visible": True}
            )]),
    ]),

    html.Div(children=[
        html.Div(id='output_visualization_daybase'),
        html.Div(id='output_daywise_windspeed'),
        html.Div(id='stats_section', className='container', style={'margin-top': '20px'}),
    ]),
    html.Br(),
    html.Footer(
        html.P("Wind Turbine Power Analytics Dashboard", 
              style={'textAlign': 'center', 'color': colors['background']})
    )
])

# Callbacks
@app.callback(
    Output('output_visualization_daybase', 'children'),
    [Input('selection_based_on_hours', 'date'),
     Input('time_range', 'value')])
def update_total_daywise(date, value):
    if date is None:
        # Handle None date
        return html.Div("Please select a date")
    
    data = filter_data_based_on_hours(date, value[0], value[1], df)
    fig = create_daily_figure(data)
    
    return dcc.Graph(
        id='daily_graph',
        figure=fig
    )

@app.callback(
    Output('output_daywise_windspeed', 'children'),
    [Input('selection_based_on_hours', 'date'),
     Input('time_range', 'value')])
def update_wind_visualization(date, value):
    if date is None:
        # Handle None date
        return html.Div("Please select a date")
        
    data = filter_data_based_on_hours(date, value[0], value[1], df)
    fig = create_wind_speed_daily(data)
    
    return dcc.Graph(
        id='daily_windspeed_graph',
        figure=fig
    )

@app.callback(
    Output('stats_section', 'children'),
    [Input('selection_based_on_hours', 'date'),
     Input('time_range', 'value')])
def update_stats_section(date, value):
    if date is None:
        # Handle None date
        return html.Div()
        
    data = filter_data_based_on_hours(date, value[0], value[1], df)
    
    if data.empty:
        return html.Div("No data available for the selected period")
    
    # Calculate key statistics
    avg_power = data['LV ActivePower (kW)'].mean()
    max_power = data['LV ActivePower (kW)'].max()
    avg_wind = data['Wind Speed (m/s)'].mean()
    max_wind = data['Wind Speed (m/s)'].max()
    efficiency = (data['LV ActivePower (kW)'].sum() / data['Theoretical_Power_Curve (KWh)'].sum() * 100) if data['Theoretical_Power_Curve (KWh)'].sum() > 0 else 0
    
    # Create stats cards
    stats_row = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Average Power"),
            dbc.CardBody(f"{avg_power:.2f} kW")
        ], color="primary", outline=True), width=2),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Max Power"),
            dbc.CardBody(f"{max_power:.2f} kW")
        ], color="danger", outline=True), width=2),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Avg Wind Speed"),
            dbc.CardBody(f"{avg_wind:.2f} m/s")
        ], color="info", outline=True), width=2),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Max Wind Speed"),
            dbc.CardBody(f"{max_wind:.2f} m/s")
        ], color="warning", outline=True), width=2),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Efficiency"),
            dbc.CardBody(f"{efficiency:.2f}%")
        ], color="success", outline=True), width=2),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Data Points"),
            dbc.CardBody(f"{len(data)}")
        ], color="secondary", outline=True), width=2),
    ])
    
    return stats_row