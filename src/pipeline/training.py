import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from tasks import *

df = load_data(
    "data/raw/agriBORA_maize_prices.csv"
)

final_df = prepare_training_data(df)

X_train,y_train,X_val,y_val = split_data(
    final_df
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_val)

mae = mean_absolute_error(
    y_val,
    pred
)

print("Validation MAE:", mae)

joblib.dump(
    model,
    "data/trained_models/best_random_forest_model.pkl"
)