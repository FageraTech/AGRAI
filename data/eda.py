"""
EDA Module for AgriBORA Maize Price Forecasting
Visualization and exploratory analysis functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_trends(df, title="Maize Price Trends by County"):
    """Plot price trends over time"""
    
    plt.figure(figsize=(12, 6))
    
    for county in df['County'].unique():
        data = df[df['County'] == county]
        plt.plot(data['Date'], data['WholeSale'], label=county, marker='.', markersize=3)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Wholesale Price (KES)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_price_distribution(df, title="Price Distribution by County"):
    """Plot boxplot of prices by county"""
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='County', y='WholeSale')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('County')
    plt.ylabel('Price (KES)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weekly_seasonality(df, title="Weekly Price Distribution"):
    """Plot seasonality patterns by week"""
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='WeekofYear', y='WholeSale')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Week of Year')
    plt.ylabel('Price (KES)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compare_price_sources(agri_df, kamis_df, kamis_raw_df, target_counties):
    """Compare prices across three sources"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def plot_monthly_boxplot_by_county(df):

    temp = df.copy()

    temp['YearMonth'] = temp['Date'].dt.strftime('%Y-%m')

    g = sns.catplot(
        data=temp,
        x='YearMonth',
        y='WholeSale',
        col='County',
        kind='box',
        col_wrap=2,
        height=4,
        sharey=True
    )

    g.set_xticklabels(rotation=90)

    plt.show()
        
    # Mean prices
    agri_means = []
    kamis_means = []
    raw_means = []
    
    for county in target_counties:
        agri_means.append(agri_df[agri_df['County'] == county]['WholeSale'].mean())
        
        kamis_subset = kamis_df[kamis_df['County'] == county]
        kamis_means.append(kamis_subset['WholeSale'].mean() if len(kamis_subset) > 0 else 0)
        
        raw_subset = kamis_raw_df[kamis_raw_df['County'] == county]
        raw_means.append(raw_subset['WholeSale'].mean() if len(raw_subset) > 0 else 0)
    
    x = np.arange(len(target_counties))
    width = 0.25
    
    axes[0,0].bar(x - width, agri_means, width, label='AgriBORA', color='blue')
    axes[0,0].bar(x, kamis_means, width, label='Kamis', color='orange')
    axes[0,0].bar(x + width, raw_means, width, label='Kamis Raw', color='green')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(target_counties, rotation=45)
    axes[0,0].set_title('Mean Price Comparison')
    axes[0,0].set_ylabel('Price (KES)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Price differences
    diffs = [agri_means[i] - kamis_means[i] if kamis_means[i] > 0 else 0 for i in range(len(target_counties))]
    colors = ['red' if d < -10 else 'orange' if d < -5 else 'green' for d in diffs]
    
    axes[0,1].bar(target_counties, diffs, color=colors)
    axes[0,1].axhline(y=0, color='black', linewidth=1)
    axes[0,1].set_title('Price Difference (AgriBORA - Kamis)')
    axes[0,1].set_ylabel('Difference (KES)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Data volume
    agri_rows = [len(agri_df[agri_df['County'] == county]) for county in target_counties]
    kamis_rows = [len(kamis_df[kamis_df['County'] == county]) for county in target_counties]
    
    axes[1,0].bar(x - width/2, agri_rows, width, label='AgriBORA', color='blue')
    axes[1,0].bar(x + width/2, kamis_rows, width, label='Kamis', color='orange')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(target_counties, rotation=45)
    axes[1,0].set_title('Data Volume Comparison')
    axes[1,0].set_ylabel('Number of Rows')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes[1,1].axis('off')
    
    plt.suptitle('Kamis vs AgriBORA: Complete Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_data_coverage(df, title="Data Coverage After Merge"):
    """Plot data coverage scatter plot"""
    
    plt.figure(figsize=(12, 6))
    
    for county in df['County'].unique():
        data = df[df['County'] == county]
        plt.scatter(data['Date'], data['WholeSale'], s=10, label=county, alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (KES)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_commodity_distribution(kamis_raw_df):
    """Plot commodity distribution from Kamis Raw"""
    
    commodity_counts = kamis_raw_df[['Commodity', 'Classification']].value_counts().reset_index(name='count')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    top10 = commodity_counts.head(10)
    axes[0].barh(range(len(top10)), top10['count'], color='skyblue')
    axes[0].set_yticks(range(len(top10)))
    axes[0].set_yticklabels(top10['Commodity'] + ' - ' + top10['Classification'])
    axes[0].set_xlabel('Number of Rows')
    axes[0].set_title('Top 10 Commodity-Classification Combinations')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Pie chart
    dry_maize = commodity_counts[commodity_counts['Commodity'] == 'dry maize']['count'].sum()
    total = commodity_counts['count'].sum()
    other = total - dry_maize
    
    axes[1].pie([dry_maize, other], 
                labels=[f'Dry Maize\n({dry_maize} rows)', f'Other\n({other} rows)'],
                autopct='%1.1f%%', colors=['green', 'lightgray'], startangle=90)
    axes[1].set_title('Dry Maize vs Other Commodities')
    
    plt.suptitle('Kamis Raw Data - Commodity Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("EDA module loaded. Available functions:")
    print("  - plot_price_trends()")
    print("  - plot_price_distribution()")
    print("  - plot_weekly_seasonality()")
    print("  - compare_price_sources()")
    print("  - plot_data_coverage()")
    print("  - plot_commodity_distribution()")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# CORRELATION ANALYSIS
# ==================================================

def correlation_analysis(
        agri_df,
        kamis_df,
        kamis_raw_df,
        target_counties):

    print("="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    results=[]

    for county in target_counties:

        # Agri
        agri=agri_df[
            agri_df['County']==county
        ][['Date','WholeSale']]

        agri=agri.rename(
            columns={'WholeSale':'agri'}
        )


        # Kamis
        kamis=kamis_df[
            kamis_df['County']==county
        ][['Date','WholeSale']]

        kamis=kamis.rename(
            columns={'WholeSale':'kamis'}
        )


        # Kamis Raw
        raw=kamis_raw_df[
            kamis_raw_df['County']==county
        ][['Date','WholeSale']]

        raw=raw.rename(
            columns={'WholeSale':'raw'}
        )


        # merge only overlapping dates

        merged=pd.merge(
            agri,
            kamis,
            on='Date',
            how='inner'
        )

        merged=pd.merge(
            merged,
            raw,
            on='Date',
            how='inner'
        )


        if len(merged)<2:

            print(
                f"{county}: Not enough overlap"
            )

            continue


        corr_kamis=merged[
            'agri'
        ].corr(
            merged['kamis']
        )

        corr_raw=merged[
            'agri'
        ].corr(
            merged['raw']
        )


        mae=mean_absolute_error(
            merged['agri'],
            merged['kamis']
        )

        rmse=np.sqrt(
            mean_squared_error(
                merged['agri'],
                merged['kamis']
            )
        )

        r2=r2_score(
            merged['agri'],
            merged['kamis']
        )


        results.append({

            'County':county,
            'Correlation_Kamis':
                round(corr_kamis,3),

            'Correlation_Raw':
                round(corr_raw,3),

            'MAE':
                round(mae,2),

            'RMSE':
                round(rmse,2),

            'R2':
                round(r2,3),

            'Rows':
                len(merged)

        })

    return pd.DataFrame(results)



# ==================================================
# VISUALIZE RESULTS
# ==================================================

def plot_correlation_results(df):

    fig,axes=plt.subplots(
        2,
        2,
        figsize=(14,10)
    )

    sns.barplot(
        data=df,
        x='County',
        y='Correlation_Kamis',
        ax=axes[0,0]
    )

    axes[0,0].set_title(
        'Correlation'
    )



    sns.barplot(
        data=df,
        x='County',
        y='MAE',
        ax=axes[0,1]
    )

    axes[0,1].set_title(
        'MAE'
    )



    sns.barplot(
        data=df,
        x='County',
        y='RMSE',
        ax=axes[1,0]
    )

    axes[1,0].set_title(
        'RMSE'
    )



    sns.barplot(
        data=df,
        x='County',
        y='R2',
        ax=axes[1,1]
    )

    axes[1,1].set_title(
        'R²'
    )

    plt.tight_layout()

    plt.show()



# ==================================================
# SCATTER PLOT
# ==================================================

def plot_scatter_comparison(
        agri_df,
        kamis_df,
        county):


    agri=agri_df[
        agri_df['County']==county
    ][['Date','WholeSale']]

    kamis=kamis_df[
        kamis_df['County']==county
    ][['Date','WholeSale']]


    agri=agri.rename(
        columns={'WholeSale':'agri'}
    )

    kamis=kamis.rename(
        columns={'WholeSale':'kamis'}
    )


    merged=pd.merge(
        agri,
        kamis,
        on='Date',
        how='inner'
    )


    plt.figure(figsize=(7,6))

    sns.scatterplot(
        data=merged,
        x='agri',
        y='kamis'
    )

    plt.title(
        f'{county}: Agri vs Kamis'
    )

    plt.grid()

    plt.show()