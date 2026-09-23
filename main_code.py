import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import EstimationWarning, ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.simplefilter('ignore', EstimationWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

def run_pipeline():
    
    print("MARUTI SUZUKI DEMAND FORECASTING ENGINE (2025)")
    

    # 1. Generate Historical Monthly Sales & Macro Data (2017-2024)
    np.random.seed(42)
    dates = pd.date_range(start='2017-01-01', end='2024-12-01', freq='MS')
    n_months = len(dates)

    trend = np.linspace(130000, 160000, n_months)
    month_effect = np.array([1.0, 0.98, 1.05, 1.02, 0.96, 0.94, 0.97, 1.01, 1.18, 1.25, 1.20, 1.08])
    seasonality = np.tile(month_effect, 8)

    repo_rates = np.sin(np.linspace(0, 3*np.pi, n_months)) * 0.75 + 6.25
    petrol_prices = np.linspace(70, 102, n_months) + np.random.normal(0, 1, n_months)
    festive_flag = [1 if d.month in [9, 10, 11] else 0 for d in dates]

    sales = (trend * seasonality) - (repo_rates * 1500) - (petrol_prices * 120) + np.random.normal(0, 2500, n_months)

    df_base = pd.DataFrame({
        'Date': dates,
        'Units_Sold': sales,
        'Repo_Rate': repo_rates,
        'Petrol_Price': petrol_prices,
        'Festive_Season': festive_flag,
        'Type': 'Actual'
    })

    # 2. Variable Correlation Heatmap (exploratory data analysis)
    print("Generating correlation heatmap.....")
    corr_cols = ['Units_Sold', 'Repo_Rate', 'Petrol_Price', 'Festive_Season']
    corr_matrix = df_base[corr_cols].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation coefficient'}
    )
    plt.title("Correlation Between Sales and Macro/Seasonal Drivers", fontsize=13, pad=12)
    plt.tight_layout()
    print(corr_matrix.round(2).to_string())
    print()
    plt.show()

    # 3. Time-Series Cross-Validation
    print("Running Time-Series Cross-Validation.....")
    tscv = TimeSeriesSplit(n_splits=3)
    X = df_base[['Repo_Rate', 'Petrol_Price', 'Festive_Season']]
    y = df_base['Units_Sold']

    mapes = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = SARIMAX(y.iloc[train_idx], exog=X.iloc[train_idx], order=(1,1,1), seasonal_order=(1,1,1,12),
                         enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        preds = res.predict(start=test_idx[0], end=test_idx[-1], exog=X.iloc[test_idx])
        mape = mean_absolute_percentage_error(y.iloc[test_idx], preds) * 100
        mapes.append(mape)
        print(f"Fold {fold} MAPE: {mape:.2f}%")

    print(f"\n>> Average CV MAPE: {np.mean(mapes):.2f}%\n")

    # 4. Fit Final Model & Predict 2025
    final_model = SARIMAX(y, exog=X, order=(1,1,1), seasonal_order=(1,1,1,12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    future_dates = pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
    future_exog = pd.DataFrame({
        'Repo_Rate': [6.25] * 12,
        'Petrol_Price': np.linspace(96, 100, 12),
        'Festive_Season': [1 if d.month in [9, 10, 11] else 0 for d in future_dates]
    }, index=range(len(df_base), len(df_base) + 12))

    forecast_vals = final_model.predict(start=len(df_base), end=len(df_base)+11, exog=future_exog)

    df_forecast = pd.DataFrame({
        'Date': future_dates,
        'Units_Sold': forecast_vals.values,
        'Repo_Rate': future_exog['Repo_Rate'].values,
        'Petrol_Price': future_exog['Petrol_Price'].values,
        'Festive_Season': future_exog['Festive_Season'].values,
        'Type': 'Forecast'
    })

    df_time_series = pd.concat([df_base, df_forecast], ignore_index=True)

    # 5. Multi-Dimensional Granular Expansion (10 States x 6 Models x 108 Months = 6,480 Rows)
    # Reduced West Bengal to 0.02 and allocated 0.05 to Jharkhand (Sum = 1.0)
    states = {
        'Maharashtra': 0.18, 'Delhi NCR': 0.15, 'Gujarat': 0.12,
        'Tamil Nadu': 0.11, 'Karnataka': 0.10, 'Uttar Pradesh': 0.10,
        'Punjab & Haryana': 0.09, 'Kerala': 0.08, 'West Bengal': 0.02,
        'Jharkhand': 0.05
    }

    models = {
        'Swift': {'weight': 0.22, 'color': "#D80000", 'img': 'https://www.varunmaruti.com/uploads/products/colors/new-swift-luster-blue.png'},
        'Baleno': {'weight': 0.20, 'color': "#060E67", 'img': 'https://imgd.aeplcdn.com/370x208/n/cw/ec/102663/baleno-exterior-right-front-three-quarter-69.png?isig=0&q=80'},
        'Brezza': {'weight': 0.18, 'color': "#080302", 'img': 'https://shivamautozone.com/wp-content/uploads/2026/07/brezza-2026-bl.png'},
        'Ertiga': {'weight': 0.15, 'color': '#6C757D', 'img': 'https://www.varunmaruti.com/uploads/products/colors/ertiga-pearl-metallic-arctic-white.png'},
        'Dzire': {'weight': 0.13, 'color': "#2F5987", 'img': 'https://static-cdn.team-bhp.com/prod/new-car-cms/Maruti-Suzuki/New-Dzire/2024/11/11/1925973a-30dd-49e4-b4a3-4411b24c34e5-6-_11_.png?w=688&dpr=3&optimize=low&format=auto&quality=50'},
        'Alto K10': {'weight': 0.12, 'color': "#527E5E", 'img': 'https://shivamautozone.com/wp-content/uploads/2023/08/altok10-silver.png'}
    }

    full_rows = []
    for _, row in df_time_series.iterrows():
        for state, s_weight in states.items():
            for model_name, m_info in models.items():
                units = row['Units_Sold'] * s_weight * m_info['weight']
                full_rows.append({
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'State': state,
                    'Model': model_name,
                    'Units_Sold': round(units, 2),
                    'Repo_Rate': row['Repo_Rate'],
                    'Petrol_Price': row['Petrol_Price'],
                    'Festive_Season': row['Festive_Season'],
                    'Type': row['Type'],
                    'Image_URL': m_info['img'],
                    'Model_Color': m_info['color']
                })

    final_df = pd.DataFrame(full_rows)
    os.makedirs('data', exist_ok=True)
    out_path = 'data/maruti_sales_forecast_powerbi.csv'
    final_df.to_csv(out_path, index=False)
    print(f"--- Process Complete! Exported {len(final_df)} rows to '{out_path}' ---")

    # 6. Total Units Sold Over Time - Actual vs Forecast (Area Chart)
    print("\nGenerating Actual vs Forecast area chart.....")
    csv_df = pd.read_csv(out_path, parse_dates=['Date'])

    # Aggregate back to the national monthly level (sum across all states & models)
    monthly = (
        csv_df.groupby('Date')
        .agg(Units_Sold=('Units_Sold', 'sum'), Type=('Type', 'first'))
        .sort_index()
        .reset_index()
    )

    actual_part = monthly[monthly['Type'] == 'Actual']
    forecast_part = monthly[monthly['Type'] == 'Forecast']

    # Bridge the gap so the forecast line/area connects seamlessly to the last actual point
    if not forecast_part.empty:
        bridge_row = actual_part.iloc[[-1]]
        forecast_part = pd.concat([bridge_row, forecast_part], ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(actual_part['Date'], actual_part['Units_Sold'],
                     color='#8ecae6', alpha=0.6, zorder=1)
    ax.plot(actual_part['Date'], actual_part['Units_Sold'],
            color='#1f77b4', linewidth=2, label='Actual Sales', zorder=2)

    ax.fill_between(forecast_part['Date'], forecast_part['Units_Sold'],
                     color='#023e7d', alpha=0.55, zorder=1)
    ax.plot(forecast_part['Date'], forecast_part['Units_Sold'],
            color='#03045e', linewidth=2.5, linestyle='--', label='Forecasted Sales', zorder=2)

    # Mark the Actual -> Forecast transition point clearly
    if not forecast_part.empty:
        transition_date = bridge_row['Date'].values[0]
        transition_val = bridge_row['Units_Sold'].values[0]
        ax.axvline(transition_date, color='gray', linestyle=':', linewidth=1.2, zorder=0)
        ax.scatter([transition_date], [transition_val], color='black', zorder=3, s=25)
        ax.annotate('Forecast starts', xy=(transition_date, transition_val),
                    xytext=(10, 15), textcoords='offset points', fontsize=9,
                    color='dimgray')

    ax.set_title('Maruti Suzuki: Monthly Units Sold - Actual vs Forecast', fontsize=14, pad=12)
    ax.set_xlabel('Date (Year-Month)', fontsize=12)
    ax.set_ylabel('Total Units Sold', fontsize=12)
    ax.yaxis.set_major_formatter(lambda val, pos: f'{val/1e6:.2f}M' if val >= 1e6 else f'{val/1e3:.0f}K')
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    ax.legend(loc='upper left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_pipeline()
