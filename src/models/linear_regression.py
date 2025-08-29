import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt


def load_data(file_path: str):
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Features: all except Close
    X = df.drop(columns=["Close"])
    # Target: next period Close
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]

    return train_test_split(X, y, test_size=0.2, shuffle=False)


def train_and_evaluate(file_path: str, model_out: str = "models/linear_reg.pkl"):
    X_train, X_test, y_train, y_test = load_data(file_path)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.2f}")

    # Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to: {model_out}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, preds, label="Predicted")
    plt.title("Linear Regression - Bitcoin Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

    return model, preds

if __name__ == "__main__":
    choice = input("Choose dataset (daily/weekly/monthly): ").strip().lower()

    file_map = {
        "daily": os.path.join("data", "processed", "bitcoin_daily.csv"),
        "weekly": os.path.join("data", "processed", "bitcoin_weekly.csv"),
        "monthly": os.path.join("data", "processed", "bitcoin_monthly.csv"),
    }

    if choice not in file_map:
        print("Invalid choice. Please enter daily, weekly, or monthly.")
    else:
        train_and_evaluate(file_map[choice])

