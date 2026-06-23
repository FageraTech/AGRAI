import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Import the updated multi-input routines from tasks file
from src.pipeline.task import load_data, prepare_training_data, split_data

# Loading the two operational agriBORA files
df_base = load_data("data/raw/agriBORA_maize_prices.csv")
df_weeks = load_data("data/raw/agriBORA_maize_prices_weeks_46_to_51.csv")

# Clean and bundle them together into the final feature framework
final_df = prepare_training_data(df_base, df_weeks)

# Train-test splitting step
X_train, y_train, X_val, y_val = split_data(final_df)

# Initialize the model structure
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Fit the Random Forest model
model.fit(X_train, y_train)

# Evaluate predictions on validation set
pred = model.predict(X_val)
mae = mean_absolute_error(y_val, pred)

print("Validation MAE:", mae)

# Persist the output model binary
joblib.dump(
    model,
    "data/Trained_models/best_random_forest_model.pkl"
)
print("Model training complete and successfully saved.")
