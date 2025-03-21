from dash import Dash


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/materia/bootstrap.min.css"
    ],
)
app.title = "Engergy Prediction"
server = app.server
