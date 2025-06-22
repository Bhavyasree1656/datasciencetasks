from flask import Flask, render_template
import pandas as pd
import joblib
import plotly
import plotly.graph_objs as go
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Load model and scaler
model = joblib.load("traffic_accident_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    # Load and clean dataset
    df = pd.read_csv("ADSI_Table.csv")
    X = df.drop(columns=["Sl. No.", "State/UT/City", "Total Traffic Accidents - Died"])
    y = df["Total Traffic Accidents - Died"]

    # Scale features using the same scaler
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)

    # Evaluation metrics
    metrics = {
        "MAE": round(mean_absolute_error(y, y_pred), 2),
        "MSE": round(mean_squared_error(y, y_pred), 2),
        "R2": round(r2_score(y, y_pred), 2)
    }

    # Debug: Optional
    print("Actual stats:", y.describe())
    print("Predicted stats:", pd.Series(y_pred).describe())

    # Plotly scatter plot
    trace = go.Scatter(
        x=y, y=y_pred, mode='markers',
        marker=dict(size=8, color="royalblue", line=dict(width=1, color="black")),
        text=df["State/UT/City"]
    )
    layout = go.Layout(
        title="Actual vs Predicted Traffic Accident Deaths",
        xaxis=dict(title="Actual Deaths"),
        yaxis=dict(title="Predicted Deaths"),
        template="plotly_white",
        height=500
    )
    fig = go.Figure(data=[trace], layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("index.html", metrics=metrics, graphJSON=graphJSON)

if __name__ == "__main__":
    app.run(debug=True)
