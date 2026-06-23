# Maize Price Forecasting Pipeline

## Project Overview

This project develops a machine learning pipeline for forecasting wholesale maize prices across selected Kenyan counties.

The forecasting system uses historical market prices, feature engineering techniques, machine learning models, and validation procedures to predict future market prices.

## Counties Covered

* Kiambu
* Kirinyaga
* Mombasa
* Nairobi
* Uasin-Gishu

## Project Structure

```text
maize-price-forecasting/
├── dags/
├── src/pipeline/
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

## Features

* Feature Engineering
* Random Forest Forecasting
* MLflow Experiment Tracking
* Model Validation
* Apache Airflow Workflow
* Performance Monitoring

## Model

Selected Model:

* Random Forest Regressor

Evaluation Metric:

* Mean Absolute Error (MAE)

## Workflow

1. Load market data.
2. Engineer forecasting features.
3. Train models.
4. Evaluate performance.
5. Select best model.
6. Generate forecasts.
7. Validate predictions.
8. Monitor model accuracy.

## Future Improvements

* Automated retraining.
* Dashboard deployment.
* Scheduled forecasting.
* Cloud deployment.
* Model drift detection.

## Author

Jairus Odhiambo Rabala

Maize Price Forecasting Project
