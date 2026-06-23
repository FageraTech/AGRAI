import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_mae(
        actual,
        predicted
):

    return mean_absolute_error(
        actual,
        predicted
    )


def monitor_model(validation_df):

    validation_df["error"] = abs(
        validation_df["actual_w1"]
        -
        validation_df["pred_w1"]
    )

    mae = validation_df["error"].mean()

    print(
        f"Current MAE: {mae:.2f}"
    )

    return mae