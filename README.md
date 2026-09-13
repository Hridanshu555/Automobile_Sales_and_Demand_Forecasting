# Task_1_Future_Interns_Internship_Sales_and_Demand_Forecasting-
A SARIMAX equipped ML Model made using Python programming backed with Power BI Visualization for Automobile Sales and Demand forecasting and analysis 


# 🚘 Maruti Suzuki Demand Forecasting Engine & Macroeconomic Analytics (2025)

An end-to-end econometric forecasting engine and interactive Power BI executive dashboard designed to project monthly vehicle demand for Maruti Suzuki. 

By integrating historical sales data (2017–2024) with real-world macroeconomic indicators (**Repo Rate**, **Petrol Prices**, and **Festive Spikes**), this project bridges the gap between machine learning time-series modeling and executive decision-making.

---

## 📌 Project Overview

Traditional automotive forecasting often relies purely on past sales volumes, failing to account for macroeconomic headwinds. This pipeline models vehicle demand by evaluating:
1. **Macroeconomic Sensitivity:** How changes in central bank interest rates (Repo Rate) and fuel prices influence consumer purchase intent.
2. **Seasonality & Festive Demand:** Accounting for high-volume quarter spikes during major festive seasons (Sept–Nov).
3. **Econometric Time-Series Forecasting:** Utilizing a **SARIMAX** model with exogenous regressors to project monthly sales through 2025.
4. **Granular Multi-State Expansion:** Distributing national forecasts across 10 major Indian states and 6 key Maruti Suzuki model lines.

---

## 🛠️ Tech Stack & Key Libraries

* **Language:** Python 3.10+
* **Time-Series & Statistics:** `statsmodels` (SARIMAX), `scikit-learn` (TimeSeriesSplit, MAPE)
* **Data Processing & Manipulation:** `pandas`, `numpy`
* **Business Intelligence & Visualization:** Power BI Desktop (DAX, Dynamic Custom Tooltips, Image URL Rendering)

---

## ⚙️ Architecture & Data Pipeline

The Python engine (`run_pipeline.py`) performs the following sequence:
-[ Historical Data Generation (2017-2024) ]-> -[ Macro Features: Repo Rate | Petrol Price | Festive Flag ]-> -[ 3-Fold Time-Series Cross-Validation (Evaluated via MAPE) ]-> -[ Final SARIMAX Fitting & 2025 Monthly Demand Prediction ]-> -[ Multi-Dimensional Expansion (10 States × 6 Vehicle Models) ]-> -[ Export to data/maruti_sales_forecast_powerbi.csv ]
