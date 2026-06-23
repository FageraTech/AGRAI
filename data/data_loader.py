"""
Data Loader Module
AgriBORA Maize Forecasting
"""

import pandas as pd
import os


# =====================================================
# BASE DATA PATH
# =====================================================

DATA_PATH = "/content/drive/MyDrive/Agribora_Final/data"


# =====================================================
# TARGET COUNTIES
# =====================================================

def get_target_counties():

    return [
        'Kiambu',
        'Kirinyaga',
        'Mombasa',
        'Nairobi',
        'Uasin-Gishu'
    ]


# =====================================================
# GENERIC CSV LOADER
# =====================================================

def load_csv(file_path):

    try:

        df = pd.read_csv(file_path)

        print("="*60)
        print(f"Loaded: {os.path.basename(file_path)}")
        print(f"Shape : {df.shape}")
        print("="*60)

        return df

    except Exception as e:

        print("="*60)
        print("ERROR LOADING FILE")
        print(file_path)
        print(e)
        print("="*60)

        return None


# =====================================================
# LOAD MAIN AGRIBORA
# =====================================================

def load_agri_main():

    return load_csv(
        f"{DATA_PATH}/agriBORA_maize_prices.csv"
    )


# =====================================================
# LOAD AGRIBORA WEEK 46-51
# =====================================================

def load_agri_weeks():

    return load_csv(
        f"{DATA_PATH}/agriBORA_maize_prices_weeks_46_to_51.csv"
    )


# =====================================================
# LOAD KAMIS
# =====================================================

def load_kamis():

    return load_csv(
        f"{DATA_PATH}/kamis_maize_prices.csv"
    )


# =====================================================
# LOAD KAMIS RAW
# =====================================================

def load_kamis_raw():

    return load_csv(
        f"{DATA_PATH}/kamis_maize_prices_raw.csv"
    )


# =====================================================
# LOAD FINAL WEEKS
# =====================================================

def load_final_weeks():

    return load_csv(
        f"{DATA_PATH}/agriBORA_Final_Weeks_maize_price.csv"
    )


# =====================================================
# LOAD EVERYTHING
# =====================================================

def load_all_data():

    data = {

        'agri_df': load_agri_main(),

        'agriW51_df': load_agri_weeks(),

        'kamis': load_kamis(),

        'kamis_raw': load_kamis_raw(),

        'finalW_df': load_final_weeks()
    }

    print("\nAll datasets loaded successfully")

    return data


# =====================================================
# RUN FILE DIRECTLY
# =====================================================

if __name__ == "__main__":

    data = load_all_data()

    print("\nAvailable keys:")
    print(data.keys())