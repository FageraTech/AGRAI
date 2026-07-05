# Agricultural Products Pipeline

## Author

Jairus Odhiambo Rabala

---

## Project Overview

This project develops a machine learning pipeline for forecasting maize prices across selected counties in Kenya.

The forecasting system uses historical market prices, feature engineering techniques, machine learning models, and validation procedures to predict future market prices.

### Counties Covered

* Kisumu
* Kirinyaga
* Kericho
* Nyerui
* Uasin-Gishu
* Bungoma
* Narok

---

## Project Structure

```text
main folder/
├── dags/                           # Apache Airflow workflow automation DAG files
│   └── crop_pipeline_dag.py
├── src/pipeline/                   # Core pipeline library files
│   ├── __init__.py
│   ├── tasks.py                    # Reusable feature engineering and data loading formulas
│   ├── training.py                 # Model training and artifact serialization script
│   ├── forecasting.py              # Inference routines and pricing logic
│   ├── validation.py               # Statistical performance evaluations
│   └── monitoring.py               # Data drift and system stability checks
├── data/                           # Data asset workspace
│   ├── raw/                        # Original agriBORA input CSV files
│   ├── processed/                  # Staged arrays for model input
│   ├── predictions/                # Output storage for weekly_predictions.csv
│   └── trained_models/             # Storage for best_random_forest_model.pkl
├── notebooks/                      # R&D Jupyter Development Notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_validation.ipynb
├── mlruns/                         # Local MLflow logging runs
├── app.py                          # Interactive Streamlit Frontend UI
├── main.py                         # End-to-End Local Execution Script
├── requirements.txt                # Python project dependencies
└── README.md                       # Project blueprint and operational guide
```

---

## Features

* **Feature Engineering**: Automated generation of time-series lags, moving rolling averages, and cyclical seasonal sine/cosine mathematical waves.
* **Random Forest Forecasting**: Robust ensemble-based estimation of price movements.
* **MLflow Experiment Tracking**: Systematic parameter, metric, and artifact tracking across model training loops.
* **Model Validation**: Strict chronological evaluation to handle time-series data constraints without information leakage.
* **Apache Airflow Workflow**: Fault-tolerant scheduling, isolation of tasks, and orchestration of weekly data streams.
* **Performance Monitoring**: Validation error diagnostics and prediction tracking over time.

---

## Model

* **Selected Model**: Random Forest Regressor
* **Evaluation Metric**: Mean Absolute Error (MAE)
* **Evaluation Metric**: Root Measn Squared Error(RMSE)

---

## 🧪 Notebooks (Research & Development Playground)

The files inside the `notebooks/` directory act as an interactive laboratory. They allow developers to experiment with new mathematical hypotheses, run trial modeling experiments, and visualize structures safely before translating code changes into our stable production files (`tasks.py` and `training.py`).

### Notebook Breakdown & Refining Workflow

* **`01_preprocessing.ipynb`**: Cleans raw data and handles structural anomalies or missing records found across different file streams.
* **`02_eda.ipynb`**: Conducts Exploratory Data Analysis to identify price trends, seasonal distribution flags, and outlier movements across the 5 target Kenyan counties.
* **`03_feature_engineering.ipynb`**: Used to prototype and stress-test new lag periods or moving metrics to see if they increase signal correlations before deploying them to the core pipeline library.
* **`04_modeling.ipynb`**: Serves as our algorithm proving ground. Use this notebook to experiment with hyperparameter combinations and compare alternative tree estimators.
* **`05_validation.ipynb`**: Verifies predictive stability by mapping rolling errors and testing cross-validation constraints against recent historical data profiles.

---

## Workflow

1. **Load market data**: Pull raw data assets from the `data/raw/` directory structure.
2. **Engineer forecasting features**: Build time-series lags, rolling means, and cyclical features.
3. **Train models**: Execute fitting routines on historical training data segments.
4. **Evaluate performance**: Calculate validation step metrics using Mean Absolute Error (MAE).
5. **Select best model**: Package and store the best-performing iteration file as a serialized pickle binary.
6. **Generate forecasts**: Run model inference on the latest input datasets to project market deltas.
7. **Validate predictions**: Apply safety checks to confirm that outputs match acceptable economic thresholds.
8. **Monitor model accuracy**: Track system outputs against upcoming real-world retail pricing indexes.

---

## ⚙️ Infrastructure & Operational Setup Guide

### 1. Setting Up Infrastructure & Tools

* **Python (3.14+)**: Ensure Python is added to local system's environment variables.
* **Docker Desktop**: Download and install Docker Desktop. This engine is required to host the Apache Airflow scheduler. **Crucial Windows Setting**: Ensure that the **WSL2 Engine backend** check box remains active during the software's initial configuration setup.

### 2. Virtual Environment Configuration (`n`)

Open your PowerShell terminal inside the `folder` root directory and execute these instructions sequentially to launch your local environment:

```powershell
# Create the virtual sandbox environment
python -m venv n

# Activate the virtual environment path mapping link
\n\Scripts\activate

# Upgrade pip to the latest stable edition

python -m pip install --upgrade pip

# Install the required pipeline packages and dependencies
pip install -r requirements.txt
```

### 3. Model Training (`src/pipeline/training.py`)

The model training logic has been cleanly integrated into the production library folder (`src/pipeline/`). To execute retraining safely without encountering any environment cross-referencing path errors, use python execution from the root directory:

```powershell
python src/pipeline/training.py
```

*Note: This script uses system-level path injection (`sys.path.append`) at startup to register imports from `src.pipeline.tasks` seamlessly.*

### 4. Running the Pipeline End-to-End Locally (`main.py`)

To trigger a manual run of the extraction, transformations, feature engineering, and model inference operations locally from start to finish, execute:

```powershell
python main.py
```

This script reads files from `data/raw/`, processes features, loads your saved model, and automatically updates the data file at `data/predictions/weekly_predictions.csv` with formatted long-form rows.

### 5. Running the Apache Airflow Workflow

To turn over the automated weekly pipeline tasks to the production scheduling engine via Docker, run this command in your active project root terminal:

```powershell
# Boot up the container ecosystem components in detached background daemon mode
docker compose up -d
```

1. Open your web browser window and navigate to: `http://localhost:8080`
2. Enter the workspace credentials (Default workspace profiles use: `admin` / `admin`).
3. Locate **`main folder`** inside the dashboard registry list, toggle the pipeline switch to **On**, and click the **Play (Trigger)** icon to test-run the orchestration sequence.

### 6. Visualizing Forecast Results with Streamlit

To view interactive trend data plots alongside your model's short-term and long-term county projections, execute the following visualization script command:

```powershell
streamlit run app.py
```

Your computer will open an active application screen tab at **`http://localhost:8501`**. Use the interactive drop-down picker tool to navigate across market zones.

---

## Future Improvements

* **Automated retraining**: Scheduling programmatic `training.py` runs inside Airflow to update model weights as new data arrives.
* **Dashboard deployment**: Migrating our local Streamlit framework to an open-source virtual private server (VPS).
* **Scheduled forecasting**: Connecting automated notification alerts when anomalies or steep price jumps are detected.
* **Cloud deployment**: Moving raw storage locations to cloud object storage.
* **Model drift detection**: Building statistical population monitoring routines into `monitoring.py` to trigger warning logs when real-world price distributions shift away from baseline training matrices.
