import joblib
import numpy as np

MODEL_PATH = "data/trained_models/best_random_forest_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_next_weeks(
        county_data,
        model,
        features=["diff_lag1"]
):

    last = county_data.iloc[-1]

    X = (
        last[features]
        .values
        .reshape(1, -1)
    )

    pred_diff = model.predict(X)[0]

    last_price = last["price"]

    week52 = last_price + (0.6 * pred_diff)

    week1 = week52 + (2.4 * pred_diff)

    return round(week52,2), round(week1,2)