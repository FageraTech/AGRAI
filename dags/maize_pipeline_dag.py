from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.task import (
    load_data,
    prepare_training_data,
    load_model,
    save_predictions
)

DATA_PATH = "C:/Users/Njoroge/Desktop/maize_price_model/data/agriBORA_maize.csv"

MODEL_PATH = (
    "C:/Users/Njoroge/Desktop/maize_price_model/"
    "data/Trained_models/"
    "best_random_forest_model.pkl"
)

OUTPUT_PATH = (
    "C:/Users/Njoroge/Desktop/maize_price_model/"
    "data/predictions/"
    "weekly_predictions.csv"
)


def extract_task(**context):

    df = load_data(DATA_PATH)

    context["ti"].xcom_push(
        key="records",
        value=df.to_json()
    )


def transform_task(**context):

    import pandas as pd

    raw_data = context["ti"].xcom_pull(
        task_ids="extract_data",
        key="records"
    )

    df = pd.read_json(raw_data)

    final_df = prepare_training_data(df)

    context["ti"].xcom_push(
        key="features",
        value=final_df.to_json()
    )


def predict_task(**context):

    import pandas as pd

    data_json = context["ti"].xcom_pull(
        task_ids="feature_engineering",
        key="features"
    )

    feature_df = pd.read_json(data_json)

    model = load_model(MODEL_PATH)

    X = feature_df[["diff_lag1"]]

    predictions = model.predict(X)

    feature_df["prediction"] = predictions

    save_predictions(
        feature_df,
        OUTPUT_PATH
    )


with DAG(

    dag_id="maize_price_forecasting",

    start_date=datetime(2026, 1, 1),

    schedule="@weekly",

    catchup=False,

    tags=["agribora", "forecasting"]

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