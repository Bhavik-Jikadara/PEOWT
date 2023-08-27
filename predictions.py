from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import mysql_connect
import pickle
from sklearn.preprocessing import MinMaxScaler


batch_size = 1
epoch = 7
neurons = 10
predict_values = 72
lag = 24

df = mysql_connect.df
df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='mixed')


batch_size = 1
epoch = 7
neurons = 10
predict_values = 72
lag = 24


model = pickle.load(open('windpower_model.p', 'rb'))

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Create a differenced series
def difference(dataset, interval = 1):
    diff = list() # empty list
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differeced value
def inverse_difference(history, yhat, interval = 1):
    return yhat + history[-interval]



def minmaxscaler(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)

    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# Make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0,0]

raw_values = df['LV_ActivePower_kW'].values
diff_values = difference(raw_values, 1)

supervised = timeseries_to_supervised(diff_values, lag)
supervised_values = supervised.values

train, test = supervised_values[0:-predict_values], supervised_values[-predict_values:]

scaler, train_scaled, test_scaled = minmaxscaler(train, test)

predictions = list()
expectations = list()
test_pred = list()

for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(model, 1, X)

    # Replacing value in test scaled with the predicted value.
    test_pred = [yhat] + test_pred
    if len(test_pred) > lag + 1:
        test_pred = test_pred[:-1]
    if i+1<len(test_scaled):
        if i+1 > lag+1:
            test_scaled[i+1] = test_pred
        else:
            test_scaled[i+1] = np.concatenate((test_pred, test_scaled[i+1, i+1:]),axis=0)
    
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    expected = raw_values[len(train) + i + 1]
    

    if expected != 0:
        predictions.append(yhat)
        expectations.append(expected)

# =====================================================================================

def plot_wind_speed(df):
    fig = make_subplots(rows=1, cols=2, specs=[
                        [{'type': 'polar'}, {'type': 'xy'}]])
    # Direction VS Speed
    fig.add_trace(go.Scatterpolargl(
        r=df['WindSpeed_m_per_s'],
        theta=df['Wind_Direction_degree'],
        name="Wind Speed",
        marker=dict(size=5, color="mediumseagreen"),
        mode="markers"
    ),
        row=1,
        col=1)

    # Hours vs Speed
    fig.add_trace(go.Scatter(
        x=get_hours(),
        y=df['WindSpeed_m_per_s'],
        mode='lines+markers',
        name='Theoretical_Power_Curve (KWh)',
        marker=dict(symbol="circle", color="mediumseagreen"))
    )

    fig.update_layout(
        title='Predicted wind speed and wind direction for next 72 hrs.',
        paper_bgcolor='#AFEEEE',
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
        xaxis_title='Time (in hours)',
        yaxis_title='Wind Speed(m/s)'
    )
    return fig


def plot_predicted_power(df):
    fig = go.Figure()

    # LV activepower vs hours
    fig.add_trace(
        go.Scatter(
            x=get_hours(),
            y=predictions,
            mode='lines+markers',
            name='Predicted Active Power (kW)',
            marker=dict(
                symbol='circle',
                color='darkorange'
            )
        )
    )

    fig.update_layout(
        title='Predicted LV Active Power Output for next 72 hrs.',
        paper_bgcolor='#AFEEEE',
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
        xaxis_title='Time (in hours)',
        yaxis_title='Active Power(kW)'
    )

    return fig


def get_hours():
    hours = np.arange(72)
    return hours

# app layout


header = html.Div(
    "Predictions for next 72 hrs.",
    style={
        'textAlign': 'center',
        'marginLeft': '200px',
        'fontSize': '25px'
    })

button = dbc.Button(
    "Update Predictions",
    color="info",
    href="/predictions"
)

layout = html.Div(
    children=[
        html.Br(),
        dbc.Row([
                dbc.Col(header, width=10),
                dbc.Col(button, width=2)]),
        html.Br(),

        html.Div(id='active power graph', children=[
            dcc.Graph(figure=plot_predicted_power(df))
        ]),

        html.Div(id='wind speed and direction graph', children=[
            dcc.Graph(figure=plot_wind_speed(df))
        ])
    ]
)
