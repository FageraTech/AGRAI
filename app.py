import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Maize Price Forecast", layout="wide")
st.title("🌽 agriBORA Weekly Maize Price Forecasting Dashboard")

# Points directly to your automated forecasting storage path
PREDICTIONS_PATH = "data/predictions/weekly_predictions.csv"

try:
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Target_Date"] = pd.to_datetime(df["Target_Date"])

    # Sidebar layout filter for Kenyan County Markets
    counties = df["County"].unique()
    selected_county = st.sidebar.selectbox("Select Market Location:", counties)

    # Filter data based on selected location
    county_df = df[df["County"] == selected_county].sort_values("Target_Date")

    st.subheader(f"Price Trend and Projections for {selected_county} County")

    # Create an interactive line chart using Plotly
    fig = px.line(
        county_df, 
        x="Target_Date", 
        y="Maize_Price_KES", 
        color="Price_Type",
        title=f"Market Trend: {selected_county}",
        labels={"Target_Date": "Date", "Maize_Price_KES": "Price (KES)"},
        markers=True,
        color_discrete_map={
            "Actual Historical": "#1f77b4",
            "Forecast (Short-Term)": "#ff7f0e",
            "Forecast (Long-Term)": "#2ca02c"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Show raw tabular breakdown summary below the chart
    st.markdown("### Data Breakdown Summary")
    st.dataframe(county_df, use_container_width=True)

except FileNotFoundError:
    st.warning(f"Waiting for your Airflow DAG or main.py to run and generate the data file at: {PREDICTIONS_PATH}")
