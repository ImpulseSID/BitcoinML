import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import plotly.graph_objects as go
import plotly.io as pio

# Base directory = project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "linear_regression")  # dedicated folder

# Force plotly to open in browser
pio.renderers.default = "browser"

def load_data(file_path: str):
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    X = df.drop(columns=["Close"])
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]

    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_and_evaluate(file_path: str, model_out: str, html_out: str):
    X_train, X_test, y_train, y_test = load_data(file_path)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.2f}")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to: {model_out}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, mode="lines", name="Predicted"))

    fig.update_layout(
        title="Linear Regression - Bitcoin Price Prediction (Daily)",
        xaxis_title="Date",
        yaxis_title="Close Price",
        hovermode="x unified"
    )

    # Save and open HTML file (only once)
    fig.write_html(html_out, auto_open=True)
    print(f"Graph saved to: {html_out}")

    return model, preds

if __name__ == "__main__":
    daily_file = os.path.join(DATA_DIR, "bitcoin_daily.csv")
    model_out = os.path.join(MODEL_DIR, "linear_reg_daily.pkl")
    html_out = os.path.join(MODEL_DIR, "linear_reg_daily.html")
    train_and_evaluate(daily_file, model_out=model_out, html_out=html_out)
