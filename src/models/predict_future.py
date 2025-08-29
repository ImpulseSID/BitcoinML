import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_model():
    model_path = os.path.join(MODEL_DIR, "linear_reg.pkl")
    return joblib.load(model_path)

def predict_future(dataset: str, steps: int):
    file_map = {
        "daily": os.path.join(DATA_DIR, "bitcoin_daily.csv"),
        "weekly": os.path.join(DATA_DIR, "bitcoin_weekly.csv"),
        "monthly": os.path.join(DATA_DIR, "bitcoin_monthly.csv")
    }

    if dataset not in file_map:
        print("Invalid dataset. Choose daily, weekly, or monthly.")
        return

    df = pd.read_csv(file_map[dataset], index_col="Date", parse_dates=True)
    model = load_model()

    future_preds = []
    temp_df = df.copy()

    for i in range(steps):
        latest_features = temp_df.drop(columns=["Close"]).iloc[-1].values.reshape(1, -1)
        pred_price = model.predict(latest_features)[0]

        # Save prediction
        next_date = temp_df.index[-1] + pd.tseries.frequencies.to_offset(
            {"daily": "1D", "weekly": "1W", "monthly": "1M"}[dataset]
        )
        future_preds.append((next_date, pred_price))

        # Append prediction back to dataframe
        new_row = temp_df.iloc[-1].copy()
        new_row["Close"] = pred_price
        temp_df.loc[next_date] = new_row

    return pd.DataFrame(future_preds, columns=["Date", "Predicted_Close"]).set_index("Date")

if __name__ == "__main__":
    choice = input("Choose dataset (daily/weekly/monthly): ").strip().lower()
    steps = int(input("How many steps to predict? "))

    preds = predict_future(choice, steps)
    print(preds)
