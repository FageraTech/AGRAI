"""
Preprocessing Module
AgriBORA Maize Forecasting
"""

import pandas as pd
import numpy as np


# =====================================================
# CLEAN AGRIBORA DATA
# =====================================================

def clean_agri_data(
        agri_df,
        agriW51_df,
        target_counties
):

    print("="*60)
    print("CLEANING AGRIBORA DATA")
    print("="*60)

    agri_df = agri_df.copy()
    agriW51_df = agriW51_df.copy()

    # -----------------------------------
    # Convert dates
    # -----------------------------------

    agri_df['Date'] = pd.to_datetime(agri_df['Date'], errors='coerce')
    agriW51_df['Date'] = pd.to_datetime(agriW51_df['Date'], errors='coerce')

    # -----------------------------------
    # Filter counties
    # -----------------------------------

    agri_df = agri_df[agri_df['County'].isin(target_counties)]
    agriW51_df = agriW51_df[agriW51_df['County'].isin(target_counties)]

    # -----------------------------------
    # MERGE ONLY SELECTED DATASETS
    # -----------------------------------

    df = pd.concat([
        agri_df,
        agriW51_df
    ], ignore_index=True)

    # -----------------------------------
    # Remove bad rows
    # -----------------------------------

    df = df.dropna(subset=['Date', 'WholeSale'])

    # -----------------------------------
    # Remove duplicates
    # -----------------------------------

    df = df.drop_duplicates(
        subset=['County', 'Date'],
        keep='last'
    )

    # -----------------------------------
    # Filter date range
    # -----------------------------------

    df = df[df['Date'] >= '2022-01-01']

    # -----------------------------------
    # Sort time series
    # -----------------------------------

    df = df.sort_values(['County', 'Date']).reset_index(drop=True)

    print(f"Final shape: {df.shape}")

    return df

# -----------------------------
# Cleaning Kamis data
# -----------------------------
def clean_kamis_data(kamis_df, target_counties):

    print("="*60)
    print("CLEANING KAMIS DATA")
    print("="*60)

    kamis_clean = kamis_df.copy()

    # --------------------------------------
    # Select required columns
    # --------------------------------------
    kamis_clean = kamis_clean[
        ['County', 'Date', 'Wholesale',
         'Commodity_Classification', 'Year_Week', 'WeekofYear']
    ].rename(columns={'Wholesale': 'WholeSale'})

    # --------------------------------------
    # Convert date
    # --------------------------------------
    kamis_clean['Date'] = pd.to_datetime(kamis_clean['Date'], errors='coerce')

    # --------------------------------------
    # Filter date
    # --------------------------------------
    kamis_clean = kamis_clean[kamis_clean['Date'] >= '2022-01-01']

    # --------------------------------------
    # Filter commodity
    # --------------------------------------
    kamis_clean = kamis_clean[
        kamis_clean['Commodity_Classification'] == 'Dry_White_Maize'
    ]

    # --------------------------------------
    # Filter counties
    # --------------------------------------
    kamis_clean = kamis_clean[
        kamis_clean['County'].isin(target_counties)
    ]

    # =====================================================
    # IMPORTANT FIX 
    # Instead of drop_duplicates(),
    # we aggregate at County-Date level
    # because KAMIS has multiple market rows per day
    # =====================================================

    kamis_clean = (
        kamis_clean
        .groupby(['County', 'Date'], as_index=False)
        .agg({
            'WholeSale': 'mean',
            'Commodity_Classification': 'first',
            'Year_Week': 'first',
            'WeekofYear': 'first'
        })
    )

    # --------------------------------------
    # Sort data
    # --------------------------------------
    kamis_clean = kamis_clean.sort_values(['County', 'Date']).reset_index(drop=True)

    # --------------------------------------
    # Fill missing values
    # --------------------------------------
    kamis_clean['WholeSale'] = (
        kamis_clean.groupby('County')['WholeSale']
        .transform(lambda x: x.fillna(x.median()))
    )

    print("\nFinal Shape:", kamis_clean.shape)

    return kamis_clean

#  ============================================
# CLEAN KAMIS RAW DATA
# =============================================

def clean_kamis_raw_data(kamis_raw, target_counties):

    kamis_raw_clean = kamis_raw.copy()

    # --------------------------------------------
    # Clean text columns
    # --------------------------------------------

    kamis_raw_clean['Commodity'] = (
        kamis_raw_clean['Commodity']
        .str.lower()
        .str.strip()
    )

    kamis_raw_clean['Classification'] = (
        kamis_raw_clean['Classification']
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------
    # Create Commodity_Classification
    # --------------------------------------------

    kamis_raw_clean['Commodity_Classification'] = np.where(

        (
            (kamis_raw_clean['Commodity'] == 'dry maize')
            &
            (kamis_raw_clean['Classification'] == 'white maize')
        ),

        'Dry_White_Maize',

        'Other'
    )

    # --------------------------------------------
    # Keep Dry White Maize only
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean[
        kamis_raw_clean['Commodity_Classification']
        == 'Dry_White_Maize'
    ]

    # --------------------------------------------
    # Select columns
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean[
        [
            'County',
            'Date',
            'Wholesale',
            'Commodity_Classification',
            'Year',
            'Week'
        ]
    ]

    # --------------------------------------------
    # Rename columns
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean.rename(
        columns={
            'Wholesale':'WholeSale',
            'Week':'WeekofYear'
        }
    )

    # --------------------------------------------
    # Convert prices to numeric
    # --------------------------------------------

    kamis_raw_clean['WholeSale'] = pd.to_numeric(
        kamis_raw_clean['WholeSale'],
        errors='coerce'
    )

    # --------------------------------------------
    # Fill missing values
    # --------------------------------------------

    kamis_raw_clean['WholeSale'] = (

        kamis_raw_clean.groupby('County')['WholeSale']

        .transform(

            lambda x: x.fillna(x.median())

        )

    )

    kamis_raw_clean = kamis_raw_clean.dropna(
        subset=['WholeSale']
    )

    # --------------------------------------------
    # Convert dates
    # --------------------------------------------

    kamis_raw_clean['Date'] = pd.to_datetime(
        kamis_raw_clean['Date']
    )

    # --------------------------------------------
    # Create Year_Week
    # --------------------------------------------

    kamis_raw_clean['Year_Week'] = (

        kamis_raw_clean['Year'].astype(str)

        + '_'

        + kamis_raw_clean['WeekofYear'].astype(str)

    )

    # --------------------------------------------
    # Keep target counties
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean[
        kamis_raw_clean['County'].isin(target_counties)
    ]

    # --------------------------------------------
    # Filter years
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean[
        kamis_raw_clean['Date'] >= '2022-01-01'
    ]

    # --------------------------------------------
    # Remove duplicates
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean.drop_duplicates(
        subset=['County', 'Date'],
        keep='first'
    )

    # --------------------------------------------
    # Final columns
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean[
        [
            'County',
            'Date',
            'WholeSale',
            'Commodity_Classification',
            'Year_Week',
            'WeekofYear'
        ]
    ]

    # --------------------------------------------
    # Sort values
    # --------------------------------------------

    kamis_raw_clean = kamis_raw_clean.sort_values(
        ['County', 'Date']
    ).reset_index(drop=True)

    print(f"Final Shape: {kamis_raw_clean.shape}")

    return kamis_raw_clean