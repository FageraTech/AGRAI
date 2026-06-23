"""
Model Training Module
AgriBORA Forecasting
"""

import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

def prepare_train_val(df, features, target):

    df = df.sort_values('Date')

    split_date = df['Date'].quantile(0.8)

    train = df[df['Date'] <= split_date]

    val = df[df['Date'] > split_date]

    X_train = train[features]
    y_train = train[target]

    X_val = val[features]
    y_val = val[target]

    return X_train, y_train, X_val, y_val


# =====================================================
# TRAIN MODELS WITH MLFLOW
# =====================================================

def train_models(X_train, y_train, X_val, y_val):

    models = {

        "LightGBM": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42
        )
    }

    results = []

    trained_models = {}

    mlflow.set_tracking_uri(
        "file:///content/mlruns"
    )

    mlflow.set_experiment(
        "Agribora_Maize_Forecasting"
    )

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)

            val_pred = model.predict(X_val)

            train_mae = mean_absolute_error(
                y_train,
                train_pred
            )

            val_mae = mean_absolute_error(
                y_val,
                val_pred
            )

            rmse = np.sqrt(

                mean_squared_error(
                    y_val,
                    val_pred
                )

            )

            gap = abs(
                train_mae - val_mae
            )

            mlflow.log_metric(
                "Train_MAE",
                train_mae
            )

            mlflow.log_metric(
                "Val_MAE",
                val_mae
            )

            mlflow.log_metric(
                "RMSE",
                rmse
            )

            mlflow.log_metric(
                "Gap",
                gap
            )

            mlflow.sklearn.log_model(
                model,
                name
            )

            results.append([
                name,
                train_mae,
                val_mae,
                rmse,
                gap
            ])

            trained_models[name] = model

            print(f"\n{name}")
            print(f"Train MAE: {train_mae:.4f}")
            print(f"Val MAE: {val_mae:.4f}")
            print(f"RMSE: {rmse:.4f}")

    results_df = pd.DataFrame(

        results,

        columns=[
            "Model",
            "Train_MAE",
            "Val_MAE",
            "RMSE",
            "Gap"
        ]

    )

    best_model_name = results_df.sort_values(
        "Val_MAE"
    ).iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    return (
        results_df,
        trained_models,
        best_model,
        best_model_name
    )