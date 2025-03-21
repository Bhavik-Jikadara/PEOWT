from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from src import mysql_connect
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.express as px

# Model configuration
MODEL_CONFIG = {
    'batch_size': 1,
    'epoch': 7,
    'neurons': 10,
    'predict_values': 72,
    'lag': 24
}

# Load data
df = mysql_connect.df
df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='mixed')

# Load the trained model
model = pickle.load(open('./models/wind_power_model.pkl', 'rb'))

# Color scheme for consistent branding
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#00897B',
    'background': '#F5F7FA',
    'card_bg': '#FFFFFF',
    'text': '#263238',
    'light_text': '#546E7A',
    'accent': '#FFC107',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#F44336',
    'chart_bg': '#E3F2FD',
    'wind_speed_color': '#00897B',
    'power_color': '#F57C00'
}

# Time series helper functions
def timeseries_to_supervised(data, lag=1):
    """Frame a time series as a supervised learning dataset"""
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
    """Create a differenced series"""
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

def inverse_difference(history, yhat, interval=1):
    """Invert differenced value"""
    return yhat + history[-interval]

def minmaxscaler(train, test):
    """Scale train and test data"""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
    """Inverse scaling for a forecasted value"""
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def forecast_lstm(model, batch_size, X):
    """Make a one-step forecast"""
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1, verbose=0)
    return yhat[0, 0]

def get_future_dates(start_date, hours=72):
    """Generate future dates for prediction timeline"""
    dates = []
    current_date = start_date
    for i in range(hours):
        dates.append(current_date + timedelta(hours=i))
    return dates

# Prepare data and make predictions
raw_values = df['LV_ActivePower_kW'].values
diff_values = difference(raw_values, 1)

supervised = timeseries_to_supervised(diff_values, MODEL_CONFIG['lag'])
supervised_values = supervised.values

train, test = supervised_values[0:-MODEL_CONFIG['predict_values']], supervised_values[-MODEL_CONFIG['predict_values']:]
scaler, train_scaled, test_scaled = minmaxscaler(train, test)

# Make predictions
predictions = []
expectations = []
test_pred = []

for i in range(len(test_scaled)):
    # Make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(model, MODEL_CONFIG['batch_size'], X)
    
    # Update test_scaled with predicted values
    test_pred = [yhat] + test_pred
    if len(test_pred) > MODEL_CONFIG['lag'] + 1:
        test_pred = test_pred[:-1]
    if i+1 < len(test_scaled):
        if i+1 > MODEL_CONFIG['lag']+1:
            test_scaled[i+1] = test_pred
        else:
            test_scaled[i+1] = np.concatenate((test_pred, test_scaled[i+1, i+1:]), axis=0)
    
    # Invert scaling and differencing
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    
    # Store forecast and expected values
    expected = raw_values[len(train) + i + 1]
    if expected != 0:
        predictions.append(yhat)
        expectations.append(expected)

# Calculate prediction statistics
last_actual_power = df['LV_ActivePower_kW'].iloc[-1]
avg_predicted_power = np.mean(predictions)
max_predicted_power = np.max(predictions)
min_predicted_power = np.min(predictions)
total_energy_production = sum(predictions)  # kWh over 72 hours

# Wind predictions - these would typically come from your model
# Using current values for demonstration
current_wind_speed = df['WindSpeed_m_per_s'].iloc[-72:].values
current_wind_direction = df['Wind_Direction_degree'].iloc[-72:].values

# Visualization functions
def plot_wind_speed(wind_speed=current_wind_speed, wind_direction=current_wind_direction):
    """Plot wind speed and direction"""
    # Create future dates for x-axis
    latest_date = df['Date_Time'].iloc[-1]
    future_dates = get_future_dates(latest_date, hours=72)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'xy'}]],
        subplot_titles=("Wind Direction vs Speed", "Wind Speed Over Time"),
        column_widths=[0.4, 0.6]
    )
    
    # Direction VS Speed - Polar plot
    fig.add_trace(
        go.Scatterpolargl(
            r=wind_speed,
            theta=wind_direction,
            name="Wind Speed",
            marker=dict(
                size=8, 
                color=COLORS['wind_speed_color'],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            mode="markers"
        ),
        row=1,
        col=1
    )
    
    # Hours vs Speed - Line chart
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=wind_speed,
            mode='lines+markers',
            name='Wind Speed (m/s)',
            marker=dict(
                symbol="circle", 
                color=COLORS['wind_speed_color'],
                size=6
            ),
            line=dict(width=3)
        ),
        row=1,
        col=2
    )
    
    fig.update_layout(
        height=500,
        title={
            'text': 'Wind Conditions Forecast (72 Hours)',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': COLORS['text']}
        },
        paper_bgcolor=COLORS['chart_bg'],
        plot_bgcolor=COLORS['chart_bg'],
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(wind_speed) * 1.2]
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="white",
            bordercolor=COLORS['light_text'],
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(
        title_text='Date & Time',
        row=1, col=2,
        tickformat='%d %b %H:%M',
        tickangle=-45
    )
    
    fig.update_yaxes(
        title_text='Wind Speed (m/s)',
        row=1, col=2,
        gridcolor='white'
    )
    
    return fig

def plot_predicted_power():
    """Plot predicted power output"""
    # Create future dates for x-axis
    latest_date = df['Date_Time'].iloc[-1]
    future_dates = get_future_dates(latest_date, hours=len(predictions))
    
    # Create figure
    fig = go.Figure()
    
    # Add predicted power trace
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Predicted Power Output',
            line=dict(
                color=COLORS['power_color'],
                width=3
            )
        )
    )
    
    # Add uncertainty range (example - you could replace with actual confidence intervals)
    upper_bound = [p * 1.15 for p in predictions]  # 15% above prediction
    lower_bound = [p * 0.85 for p in predictions]  # 15% below prediction
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper_bound,
            mode='lines',
            name='Upper Bound (15%)',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower_bound,
            mode='lines',
            name='Lower Bound (15%)',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba{tuple(int(COLORS["power_color"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}',
            showlegend=False
        )
    )
    
    # Add markers for key points
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='markers',
            name='Hourly Prediction',
            marker=dict(
                symbol='circle',
                size=8,
                color=COLORS['power_color'],
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        title={
            'text': 'Predicted Power Output (72 Hours)',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': COLORS['text']}
        },
        paper_bgcolor=COLORS['chart_bg'],
        plot_bgcolor=COLORS['chart_bg'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="white",
            bordercolor=COLORS['light_text'],
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            title='Date & Time',
            tickformat='%d %b %H:%M',
            tickangle=-45,
            gridcolor='white'
        ),
        yaxis=dict(
            title='Active Power (kW)',
            gridcolor='white'
        )
    )
    
    return fig

def generate_summary_cards():
    """Generate summary cards with prediction statistics"""
    current_time = datetime.now().strftime("%b %d, %Y %H:%M")
    
    card_style = {
        'backgroundColor': COLORS['card_bg'],
        'borderRadius': '8px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'padding': '20px',
        'height': '100%'
    }
    
    # Summary cards
    cards = [
        dbc.Col(
            html.Div([
                html.H4("Current Power", className="card-title", style={'color': COLORS['primary']}),
                html.H2(f"{last_actual_power:.2f} kW", style={'color': COLORS['text']}),
                html.P(f"Last measured: {current_time}", style={'color': COLORS['light_text']})
            ], style=card_style),
            md=3
        ),
        dbc.Col(
            html.Div([
                html.H4("Avg Predicted", className="card-title", style={'color': COLORS['primary']}),
                html.H2(f"{avg_predicted_power:.2f} kW", style={'color': COLORS['text']}),
                html.P("Next 72 hours average", style={'color': COLORS['light_text']})
            ], style=card_style),
            md=3
        ),
        dbc.Col(
            html.Div([
                html.H4("Peak Production", className="card-title", style={'color': COLORS['primary']}),
                html.H2(f"{max_predicted_power:.2f} kW", style={'color': COLORS['text']}),
                html.P("Maximum expected output", style={'color': COLORS['light_text']})
            ], style=card_style),
            md=3
        ),
        dbc.Col(
            html.Div([
                html.H4("Total Energy", className="card-title", style={'color': COLORS['primary']}),
                html.H2(f"{total_energy_production:.2f} kWh", style={'color': COLORS['text']}),
                html.P("Predicted 72-hour production", style={'color': COLORS['light_text']})
            ], style=card_style),
            md=3
        )
    ]
    
    return dbc.Row(cards, className="mb-4")

def create_hourly_table():
    """Create a table of hourly predictions"""
    latest_date = df['Date_Time'].iloc[-1]
    future_dates = get_future_dates(latest_date, hours=min(12, len(predictions)))
    
    # Create dataframe for first 12 hours of predictions
    hourly_data = pd.DataFrame({
        'Time': [d.strftime("%m/%d %H:%M") for d in future_dates],
        'Power (kW)': [f"{p:.2f}" for p in predictions[:12]],
        'Wind Speed (m/s)': [f"{w:.2f}" for w in current_wind_speed[:12]],
        'Direction (°)': [f"{d:.1f}" for d in current_wind_direction[:12]]
    })
    
    # Create table
    table = dbc.Table.from_dataframe(
        hourly_data,
        striped=True,
        bordered=True,
        hover=True,
        className="mb-4",
        style={'backgroundColor': COLORS['card_bg']}
    )
    
    return html.Div([
        html.H4("Hourly Forecast (Next 12 Hours)", className="my-3", style={'color': COLORS['primary']}),
        table
    ])

# App layout
header = html.Div([
    html.H2(
        "Wind Turbine Power Output Predictions",
        style={
            'color': COLORS['primary'],
            'marginBottom': '10px',
        }
    ),
    html.P(
        "Forecasting energy generation for the next 72 hours based on weather predictions",
        style={
            'color': COLORS['light_text'],
            'fontSize': '1.1rem',
            'marginBottom': '20px',
        }
    )
], style={'textAlign': 'center'})

# Action buttons
buttons = html.Div([
    dbc.Button(
        [html.I(className="fas fa-sync-alt mr-2"), "Update Predictions"],
        color="primary",
        href="/predictions",
        className="mr-2",
        style={
            'fontWeight': '500',
            'borderRadius': '4px',
            'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
            'marginRight': '10px'
        }
    ),
    dbc.Button(
        [html.I(className="fas fa-download mr-2"), "Export Data"],
        color="secondary",
        outline=True,
        href="#",
        style={
            'fontWeight': '500',
            'borderRadius': '4px',
            'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        }
    )
], style={'textAlign': 'right'})

# Main layout
layout = html.Div(
    style={'backgroundColor': COLORS['background'], 'padding': '20px'},
    children=[
        # Include FontAwesome for icons
        html.Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
        ),
        
        # Header row with buttons
        dbc.Row([
            dbc.Col(header, md=8),
            dbc.Col(buttons, md=4)
        ], className="mb-4"),
        
        # Summary statistics cards
        generate_summary_cards(),
        
        # Main charts
        dbc.Row([
            dbc.Col([
                html.Div(
                    dcc.Graph(
                        id='power-prediction-graph',
                        figure=plot_predicted_power(),
                        config={'responsive': True}
                    ),
                    style={
                        'backgroundColor': COLORS['card_bg'],
                        'borderRadius': '8px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '15px',
                    }
                )
            ], md=12, className="mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(
                    dcc.Graph(
                        id='wind-prediction-graph',
                        figure=plot_wind_speed(),
                        config={'responsive': True}
                    ),
                    style={
                        'backgroundColor': COLORS['card_bg'],
                        'borderRadius': '8px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '15px',
                    }
                )
            ], md=12, className="mb-4")
        ]),
        
        # Hourly data table
        dbc.Row([
            dbc.Col([
                html.Div(
                    create_hourly_table(),
                    style={
                        'backgroundColor': COLORS['card_bg'],
                        'borderRadius': '8px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '15px',
                    }
                )
            ], md=12)
        ])
    ]
)