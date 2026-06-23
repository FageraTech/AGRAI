from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import functions from tasks.py
from src.pipeline.task import (
    load_data,
    prepare_training_data,
    load_model,
    save_predictions,
    predict_next_weeks,
    TARGET_COUNTIES
)

# DEFINE ENVIRONMENT FILE PATHS
BASE_DIR = "C:/Users/Njoroge/Desktop/maize_price_model/data/raw/"
FILE_1_PATH = f"{BASE_DIR}agriBORA_maize_prices.csv"
FILE_2_PATH = f"{BASE_DIR}agriBORA_maize_prices_weeks_46_to_51.csv"

MODEL_PATH = "C:/Users/Njoroge/Desktop/maize_price_model/data/Trained_models/best_random_forest_model.pkl"
OUTPUT_PATH = "C:/Users/Njoroge/Desktop/maize_price_model/data/predictions/weekly_predictions.csv"


def extract_task(**context):
    """Loads the two active csv source inputs and pushes them to XCom."""
    df1 = load_data(FILE_1_PATH)
    df2 = load_data(FILE_2_PATH)
    context["ti"].xcom_push(key="records_1", value=df1.to_json())
    context["ti"].xcom_push(key="records_2", value=df2.to_json())


def transform_task(**context):
    """Rebuilds dataframes and runs your combined county-level feature engine."""
    ti = context["ti"]
    raw_1 = ti.xcom_pull(task_ids="extract_data", key="records_1")
    raw_2 = ti.xcom_pull(task_ids="extract_data", key="records_2")

    df1 = pd.read_json(raw_1)
    df2 = pd.read_json(raw_2)

    final_df = prepare_training_data(df1, df2)
    ti.xcom_push(key="features", value=final_df.to_json())

#Saving the prediction and dashboard
def predict_task(**context):
    """
    Runs inference and formats the output into a clean, long-form 
    structure ideal for PowerBI, Tableau, or Streamlit dashboards.
    """
    ti = context["ti"]
    data_json = ti.xcom_pull(task_ids="feature_engineering", key="features")
    feature_df = pd.read_json(data_json)

    model = load_model(MODEL_PATH)
    features = ["lag_1", "lag_2", "diff_lag1", "rolling_mean_3", "week_num", "week_sin", "week_cos"]

    formatted_visualization_rows = []

    for county in TARGET_COUNTIES:
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
            
            date_two_weeks_out = last_known_date + pd.Timedelta(weeks=2)
            formatted_visualization_rows.append({
                "County": county,
                "Target_Date": date_two_weeks_out.strftime('%Y-%m-%d'),
                "Price_Type": "Forecast (Long-Term)",
                "Maize_Price_KES": week1_pred
            })

    results_df = pd.DataFrame(formatted_visualization_rows)
    save_predictions(results_df, OUTPUT_PATH)


# DAG STRUCTURAL BLUEPRINT
with DAG(
    dag_id="maize_price_forecasting",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["agribora", "forecasting", "random_forest"]
) as dag:

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_task
    )

    transform = PythonOperator(
        task_id="feature_engineering",
        python_callable=transform_task
    )

    predict = PythonOperator(
        task_id="generate_predictions",
        python_callable=predict_task
    )

    extract >> transform >> predict
