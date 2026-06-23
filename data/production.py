"""
Production Module for AgriBORA Maize Price Forecasting
Generates forecasts and submission files
"""

import pandas as pd
import numpy as np
import joblib
from feature import get_county_data, get_feature_list

def predict_next_weeks(county_data, model, features):
    """
    Predict maize prices for next 2 weeks using winner's logic
    
    Parameters:
    -----------
    county_data : DataFrame
        Historical data for one county with features
    model : trained model
        ML model for prediction
    features : list
        List of feature columns
    
    Returns:
    --------
    tuple : (week52_price, week1_price)
    """
    
    # Get last week's data
    last = county_data.iloc[-1]
    
    # Prepare features
    X = last[features].values.reshape(1, -1)
    
    # Predict price change
    pred_diff = model.predict(X)[0]
    
    # Get last known price
    last_price = last['price']
    
    # Heuristic scaling (winner's logic)
    week52_price = last_price + (0.6 * pred_diff)
    week1_price = week52_price + (2.4 * pred_diff)
    
    return week52_price, week1_price

def validate_on_final_weeks(finalW_df, df, model, features, target_counties):
    """
    Validate model on final weeks data
    
    Parameters:
    -----------
    finalW_df : DataFrame
        Final weeks data (ground truth)
    df : DataFrame
        Training data
    model : trained model
    features : list
        Feature columns
    target_counties : list
        Target counties
    
    Returns:
    --------
    DataFrame : Validation results
    """
    
    # Parse final weeks
    finalW_clean = finalW_df.copy()
    finalW_clean['Date'] = pd.to_datetime(finalW_clean['Date'])
    finalW_clean['County'] = finalW_clean['ID'].str.split('_').str[0]
    finalW_clean['Week'] = finalW_clean['ID'].str.extract(r'Week_(\d+)').astype(int)
    
    # Split by week
    week52 = finalW_clean[finalW_clean['Week'] == 52]
    week1 = finalW_clean[finalW_clean['Week'] == 1]
    
    # Create validation dataframe
    validation_df = week52[['County', 'WholeSale']].rename(columns={'WholeSale': 'actual_w52'}).merge(
        week1[['County', 'WholeSale']].rename(columns={'WholeSale': 'actual_w1'}),
        on='County'
    )
    
    # Get predictions
    pred_w52_list = []
    pred_w1_list = []
    
    for county in validation_df['County']:
        county_data = get_county_data(df, county)
        
        if county_data.empty:
            pred_w52_list.append(0)
            pred_w1_list.append(0)
            continue
        
        w52, w1 = predict_next_weeks(county_data, model, features)
        pred_w52_list.append(w52)
        pred_w1_list.append(w1)
    
    validation_df['pred_w52'] = pred_w52_list
    validation_df['pred_w1'] = pred_w1_list
    
    # Calculate errors
    validation_df['error_w52'] = abs(validation_df['actual_w52'] - validation_df['pred_w52'])
    validation_df['error_w1'] = abs(validation_df['actual_w1'] - validation_df['pred_w1'])
    validation_df['final_error'] = (validation_df['error_w52'] + validation_df['error_w1']) / 2
    
    final_mae = validation_df['final_error'].mean()
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(validation_df[['County', 'actual_w52', 'pred_w52', 'actual_w1', 'pred_w1', 'final_error']].to_string(index=False))
    print(f"\n Final Validation MAE: {final_mae:.6f} KES")
    
    return validation_df

def generate_production_forecast(df, model, features, target_counties):
    """
    Generate production forecast for all target counties
    
    Parameters:
    -----------
    df : DataFrame
        Training data
    model : trained model
    features : list
        Feature columns
    target_counties : list
        Target counties
    
    Returns:
    --------
    DataFrame : Production forecast
    """
    
    production = []
    
    for county in target_counties:
        county_data = get_county_data(df, county)
        
        if county_data.empty:
            print(f"Warning: No data for {county}")
            continue
        
        w52, w1 = predict_next_weeks(county_data, model, features)
        production.append([county, round(w52, 2), round(w1, 2)])
    
    production_df = pd.DataFrame(production, columns=['County', 'Week_52_Price', 'Week_1_Price'])
    
    print("\n" + "="*60)
    print("PRODUCTION FORECAST")
    print("="*60)
    print(production_df.to_string(index=False))
    
    return production_df

def create_submission(df, model, features, target_counties):
    """
    Create submission file in required format
    
    Parameters:
    -----------
    df : DataFrame
        Training data
    model : trained model
    features : list
        Feature columns
    target_counties : list
        Target counties
    
    Returns:
    --------
    DataFrame : Submission file
    """
    
    submission = []
    
    for county in target_counties:
        county_data = get_county_data(df, county)
        
        if county_data.empty:
            continue
        
        w52, w1 = predict_next_weeks(county_data, model, features)
        
        submission.append([f"{county}_Week_52", round(w52, 2), round(w52, 2)])
        submission.append([f"{county}_Week_1", round(w1, 2), round(w1, 2)])
    
    submission_df = pd.DataFrame(submission, columns=['ID', 'Target_RMSE', 'Target_MAE'])
    
    return submission_df

def save_submission(submission_df, filename="submissions/submission.csv"):
    """Save submission file"""
    
    import os
    os.makedirs("submissions", exist_ok=True)
    
    submission_df.to_csv(filename, index=False)
    print(f" Submission saved to {filename}")
    
    return submission_df

def save_production_forecast(production_df, filename="production_forecast.csv"):
    """Save production forecast"""
    
    production_df.to_csv(filename, index=False)
    print(f" Production forecast saved to {filename}")
    
    return production_df

if __name__ == "__main__":
    print("Production module loaded.")
    print("Available functions:")
    print("  - predict_next_weeks()")
    print("  - validate_on_final_weeks()")
    print("  - generate_production_forecast()")
    print("  - create_submission()")
    print("  - save_submission()")