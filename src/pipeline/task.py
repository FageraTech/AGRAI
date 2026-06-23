import pandas as pd
import numpy as np
import joblib

TARGET_COUNTIES = [
    "Kiambu",
    "Kirinyaga",
    "Mombasa",
    "Nairobi",
    "Uasin-Gishu"
]

MODEL_PATH = (
    "C:/Users/Njoroge/Desktop/maize_price_model/data/Trained_models/"
    "best_random_forest_model.pkl"
)

def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    if "County" not in df.columns:
        df["County"] = (
            df["ID"]
            .str.split("_")
            .str[0]
        )
    return df

def get_county_data(df_in, county_name):
    c_df = (
        df_in[df_in["County"] == county_name]
        .copy()
    )
    if c_df.empty:
        return pd.DataFrame()

    c_df = (
        c_df
        .sort_values("Date")
        .set_index("Date")
    )

    ts = (
        c_df["WholeSale"]
        .resample("W-MON")
        .mean()
        .interpolate(method="linear", limit_direction="both")
    )

    data = pd.DataFrame({"price": ts})
    data["lag_1"] = data["price"].shift(1)
    data["lag_2"] = data["price"].shift(2)
    data["diff_lag1"] = data["lag_1"] - data["lag_2"]
    data["rolling_mean_3"] = data["price"].rolling(3).mean()
    
    data["week_num"] = data.index.isocalendar().week.astype(int)
    data["week_sin"] = np.sin(2 * np.pi * data["week_num"] / 52)
    data["week_cos"] = np.cos(2 * np.pi * data["week_num"] / 52)
    
    data["target"] = data["price"] - data["lag_1"]
    return data.dropna()

def prepare_training_data(df1, df2):
    """Combines both agriBORA data sources dynamically before feature building."""
    # Stack the datasets vertically and drop any overlapping rows
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["Date", "County", "WholeSale"])
    
    all_data = []
    for county in TARGET_COUNTIES:
        c_data = get_county_data(combined_df, county)
        if not c_data.empty:
            c_data["County"] = county
            all_data.append(c_data)

    final_df = (
        pd.concat(all_data)
        .reset_index()
    )
    return final_df

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def save_predictions(df, path):
    df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")

def split_data(df):
    """Standard time-series split logic for your training pipeline."""
    # Sort by date to maintain chronological sequence order
    df = df.sort_values("Date")
    
    # 80/20 train/validation split point calculation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Isolate features and the delta target column
    features = ["lag_1", "lag_2", "diff_lag1", "rolling_mean_3", "week_num", "week_sin", "week_cos"]
    
    X_train = train_df[features]
    y_train = train_df["target"]
    X_val = val_df[features]
    y_val = val_df["target"]
    
    return X_train, y_train, X_val, y_val

def predict_next_weeks(county_data, model, features):
    last = county_data.iloc[-1]
    X = last[features].values.reshape(1, -1)
    
    pred_diff = model.predict(X)[0]
    last_price = last["price"]
    
    week52_price = last_price + (0.6 * pred_diff)
    week1_price = week52_price + (2.4 * pred_diff)
    
    return round(week52_price, 2), round(week1_price, 2)