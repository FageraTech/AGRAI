"""
Feature Engineering Module - AgriBORA Forecasting
"""

import pandas as pd
import numpy as np


# =====================================================
# CREATE FEATURES FOR COUNTIES
# =====================================================
def get_county_data(df_in, county_name):

    c_df = df_in[df_in['County'] == county_name].copy()

  
    c_df['Date'] = pd.to_datetime(c_df['Date'], errors='coerce')

    # remove bad dates if any
    c_df = c_df.dropna(subset=['Date'])

    # sort + set datetime index
    c_df = c_df.sort_values('Date')
    c_df = c_df.set_index('Date')

    # resample works
    ts = (
        c_df['WholeSale']
        .resample('W-MON')
        .mean()
        .interpolate(method='linear', limit_direction='both')
    )

    data = pd.DataFrame({'price': ts})

    data['lag_1'] = data['price'].shift(1)
    data['lag_2'] = data['price'].shift(2)
    data['lag_3'] = data['price'].shift(3)

    data['diff_lag1'] = data['lag_1'] - data['lag_2']
    data['diff_lag2'] = data['lag_2'] - data['lag_3']

    data['rolling_mean_3'] = data['price'].rolling(3).mean()
    data['rolling_std_3'] = data['price'].rolling(3).std()

    data['week_num'] = data.index.isocalendar().week.astype(int)
    data['week_sin'] = np.sin(2 * np.pi * data['week_num'] / 52)
    data['week_cos'] = np.cos(2 * np.pi * data['week_num'] / 52)

    data['target'] = data['price'] - data['lag_1']

    return data.dropna()


# =====================================================
# BUILD FINAL DATASET (ALL COUNTIES)
# =====================================================

def build_final_df(df, target_counties):

    all_data = []

    for county in target_counties:

        county_df = get_county_data(df, county)

        county_df["County"] = county

        all_data.append(county_df)

    final_df = pd.concat(all_data).reset_index()

    # IMPORTANT: keep time order stable
    final_df = final_df.sort_values(["County", "Date"]).reset_index(drop=True)

    return final_df


# =====================================================
# FEATURES LIST (SINGLE SOURCE OF TRUTH)
# =====================================================

def get_feature_columns():

    return [
        'lag_1',
        'lag_2',
        'lag_3',
        'diff_lag1',
        'diff_lag2',
        'rolling_mean_3',
        'rolling_std_3',
        'week_sin',
        'week_cos'
    ]


# =====================================================
# TARGET COLUMN
# =====================================================

def get_target_column():

    return "target"