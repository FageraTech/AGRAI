"""
Validation Module - AgriBORA Maize Forecasting
Handles:
- Overfitting check
- Real-world backtesting using finalW_df
"""

import numpy as np
import pandas as pd


# =====================================================
# 1. OVERFITTING CHECK
# =====================================================

def check_overfitting(results_df, best_model_name):

    best_row = results_df[
        results_df["Model"] == best_model_name
    ].iloc[0]

    train_mae = best_row["Train_MAE"]
    val_mae   = best_row["Val_MAE"]

    gap = abs(val_mae - train_mae)
    gap_ratio = gap / val_mae

    print("="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)

    print(f"Model     : {best_model_name}")
    print(f"Train MAE : {train_mae:.4f}")
    print(f"Val MAE   : {val_mae:.4f}")
    print(f"Gap       : {gap:.4f}")
    print(f"Ratio     : {gap_ratio:.4f}")

    if gap_ratio < 0.10:
        print(" Good generalization")

    elif gap_ratio < 0.20:
        print(" Mild overfitting")

    else:
        print(" Overfitting detected")

    return gap_ratio


# =====================================================
# 2. REAL-WORLD VALIDATION (FINALW_DF BACKTEST)
# =====================================================

def validate_with_final_weeks(
        finalW_df,
        df,
        model,
        features,
        get_county_data,
        predict_next_weeks
):

    """
    Uses last known weeks:
    - Week 52 (past)
    - Week 1 (future-like test)
    """

    # -----------------------------
    # Parse ID column
    # -----------------------------
    finalW_df = finalW_df.copy()

    finalW_df["County"] = finalW_df["ID"].str.split("_").str[0]
    finalW_df["Week"] = finalW_df["ID"].str.extract(r"Week_(\d+)").astype(int)

    # -----------------------------
    # Split actual values
    # -----------------------------
    week52 = finalW_df[finalW_df["Week"] == 52]
    week1  = finalW_df[finalW_df["Week"] == 1]

    validation_df = week52[["County", "WholeSale"]].rename(
        columns={"WholeSale": "actual_w52"}
    ).merge(
        week1[["County", "WholeSale"]].rename(
            columns={"WholeSale": "actual_w1"}
        ),
        on="County"
    )

    # -----------------------------
    # Predictions
    # -----------------------------
    pred_w52 = []
    pred_w1  = []

    for county in validation_df["County"]:

        c_data = get_county_data(df, county)

        if c_data.empty:
            pred_w52.append(np.nan)
            pred_w1.append(np.nan)
            continue

        w52_pred, w1_pred = predict_next_weeks(
            c_data,
            model,
            features
        )

        pred_w52.append(w52_pred)
        pred_w1.append(w1_pred)

    validation_df["pred_w52"] = pred_w52
    validation_df["pred_w1"]  = pred_w1

    # -----------------------------
    # Errors
    # -----------------------------
    validation_df["error_w52"] = abs(
        validation_df["actual_w52"] - validation_df["pred_w52"]
    )

    validation_df["error_w1"] = abs(
        validation_df["actual_w1"] - validation_df["pred_w1"]
    )

    validation_df["final_error"] = (
        validation_df["error_w52"] + validation_df["error_w1"]
    ) / 2

    print("="*60)
    print("FINAL VALIDATION RESULTS")
    print("="*60)

    print(validation_df)

    print("\nFinal MAE:",
          validation_df["final_error"].mean())

    return validation_df