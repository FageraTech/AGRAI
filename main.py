import os
import pandas as pd
from datetime import datetime

# Import your explicit task.py functions 
from src.pipeline.task import (
    load_data,
    prepare_training_data,
    load_model,
    save_predictions,
    predict_next_weeks,
    TARGET_COUNTIES
)


#  ENVIRONMENT PATH SELECTION


FILE_1_PATH = "data/raw/agriBORA_maize_prices.csv"
FILE_2_PATH = "data/raw/agriBORA_maize_prices_weeks_46_to_51.csv"
OUTPUT_PATH = "data/predictions/weekly_predictions.csv"

def run_main_pipeline():
    print(" Step 1: Loading raw historical and weekly agriBORA files...")
    if not os.path.exists(FILE_1_PATH) or not os.path.exists(FILE_2_PATH):
        print(f"❌ Error: Raw CSV files missing from your data/raw/ directory!")
        return

    df1 = load_data(FILE_1_PATH)
    df2 = load_data(FILE_2_PATH)

    print(" Step 2: Extracting county segments and engineering lag features...")
    # This combines df1 and df2 vertically and drops duplicates automatically
    feature_df = prepare_training_data(df1, df2)

    print(" Step 3: Fetching the newly trained Random Forest model binary...")
    # This calls load_model() which pulls the model from your static MODEL_PATH
    model = load_model()

    
    features = ["lag_1", "lag_2", "diff_lag1", "rolling_mean_3", "week_num", "week_sin", "week_cos"]
    formatted_visualization_rows = []

    print(" Step 4: Iterating through target markets and computing price deltas...")
    for county in TARGET_COUNTIES:
        
        # Isolate rows belonging to the specific Kenyan county market
        county_subset = feature_df[feature_df["County"] == county].sort_values("Date")
        
        if not county_subset.empty:
            
            recent_history = county_subset.tail(4)
            for _, row in recent_history.iterrows():
                formatted_visualization_rows.append({
                    "County": county,
                    "Target_Date": row["Date"].strftime('%Y-%m-%d'),
                    "Price_Type": "Actual Historical",
                    "Maize_Price_KES": round(row["price"], 2)
                })
            
            last_known_record = county_subset.iloc[-1]
            last_known_date = pd.to_datetime(last_known_record["Date"])
            
         
            week52_pred, week1_pred = predict_next_weeks(county_subset, model, features)
            
           
            date_next_week = last_known_date + pd.Timedelta(weeks=1)
            formatted_visualization_rows.append({
                "County": county,
                "Target_Date": date_next_week.strftime('%Y-%m-%d'),
                "Price_Type": "Forecast (Short-Term)",
                "Maize_Price_KES": week52_pred
            })
            
            # E. Format Long-Term prediction (2 weeks out)
            date_two_weeks_out = last_known_date + pd.Timedelta(weeks=2)
            formatted_visualization_rows.append({
                "County": county,
                "Target_Date": date_two_weeks_out.strftime('%Y-%m-%d'),
                "Price_Type": "Forecast (Long-Term)",
                "Maize_Price_KES": week1_pred
            })

    print(" Step 5: Restructuring data frames and compiling file formats...")
    results_df = pd.DataFrame(formatted_visualization_rows)
    
    # Create output directories automatically if they do not exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save predictions
    save_predictions(results_df, OUTPUT_PATH)
    print("System Run Completed: weekly_predictions.csv saved successfully!")

if __name__ == "__main__":
    run_main_pipeline()
