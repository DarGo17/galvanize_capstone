import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib

# Load pre-trained Prophet model and historical data
model = joblib.load('prophet_model.pkl')
combined_df = joblib.load('combined_df.pkl')  # Your combined_df with Date, price, MORTGAGE30US
historical = combined_df[['Date', 'price', 'MORTGAGE30US']].dropna()

conflict_periods = {
    "Great Recession (2008-2009)": ("2008-01-01", "2009-12-31"),
    "War on Terror (2008-2011)": ("2008-01-01", "2011-12-31"),
    "Arab Spring (2011-2014)": ("2011-01-01", "2014-12-31"),
    "Crimea Annexation (2014)": ("2014-01-01", "2014-12-31"),
    "US-China Trade War (2018-2019)": ("2018-07-01", "2019-12-31"),
    "COVID-19 (2020-2022)": ("2020-03-01", "2022-06-30"),
    "Russia-Ukraine War (2022-2025)": ("2022-02-01", "2025-01-01"),
    "Israel–Hamas Escalation (2023-2025)": ("2023-10-01", "2025-01-01")
}

st.title("Fayetteville Home Prices Forecast with Conflict Period Insights")

# --- User Inputs ---
months_to_predict = st.slider("Select months into the future to forecast", 1, 120, 12)
conflict_selected = st.selectbox("Select a Conflict Period to Highlight", list(conflict_periods.keys()))
show_conflict = st.checkbox("Highlight Conflict Period on Graph", value=True)
show_trend = st.checkbox("Show Trend Line", value=True)
show_uncertainty = st.checkbox("Show Uncertainty Interval", value=True)
show_historical = st.checkbox("Show Historical Prices", value=True)
show_interest = st.checkbox("Show Average Mortgage Interest Rate Line", value=False)

# Date range selector for zooming
min_date = historical['Date'].min()
max_date = historical['Date'].max() + pd.DateOffset(months=months_to_predict)
date_range = st.date_input("Select Date Range to Display", [min_date, max_date])

# --- Forecast ---
future = model.make_future_dataframe(periods=months_to_predict, freq='M')
forecast = model.predict(future)

# Filter forecast and historical data by date range
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
forecast_filtered = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
historical_filtered = historical[(historical['Date'] >= start_date) & (historical['Date'] <= end_date)]

# --- Plot ---
fig, ax = plt.subplots(figsize=(12,6))

# Plot historical data
if show_historical:
    ax.plot(historical_filtered['Date'], historical_filtered['price'], 'k.', label='Historical Prices')

# Plot predicted yhat
ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], 'b-', label='Predicted')

# Plot trend if toggled
if show_trend:
    ax.plot(forecast_filtered['ds'], forecast_filtered['trend'], 'g--', label='Trend')

# Plot uncertainty intervals if toggled
if show_uncertainty:
    ax.fill_between(forecast_filtered['ds'], forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'], color='blue', alpha=0.2, label='Uncertainty Interval')

# Highlight conflict period if toggled
if show_conflict:
    conflict_start, conflict_end = pd.to_datetime(conflict_periods[conflict_selected])
    ax.axvspan(conflict_start, conflict_end, color='orange', alpha=0.3, label='Selected Conflict Period')

    # Conflict stats
    mask = (historical['Date'] >= conflict_start) & (historical['Date'] <= conflict_end)
    conflict_prices = historical.loc[mask, 'price']
    conflict_rates = historical.loc[mask, 'MORTGAGE30US']

    if not conflict_prices.empty:
        price_change_pct = 100 * (conflict_prices.iloc[-1] - conflict_prices.iloc[0]) / conflict_prices.iloc[0]
        avg_interest_rate = conflict_rates.mean()
        st.write(f"### Conflict Period: {conflict_selected}")
        st.write(f"- Home Price Change: {price_change_pct:.2f}%")
        st.write(f"- Average Interest Rate: {avg_interest_rate:.2f}%")

        # Annotate price change on graph
        ax.annotate(f'{price_change_pct:.2f}% Price Change', xy=(conflict_end, conflict_prices.iloc[-1]),
                    xytext=(conflict_end, conflict_prices.max()),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)

# Optionally plot mortgage interest rate line
if show_interest:
    ax2 = ax.twinx()
    ax2.plot(historical_filtered['Date'], historical_filtered['MORTGAGE30US'], 'r-', label='Mortgage Interest Rate')
    ax2.set_ylabel('Mortgage Interest Rate (%)', color='r')
    ax2.tick_params(axis='y', colors='r')
    ax2.legend(loc='upper right')

ax.set_title("Fayetteville Home Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Home Price ($)")
ax.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# --- Download Forecast Data ---
csv = forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button(label="Download Forecast as CSV", data=csv, file_name='forecast.csv', mime='text/csv')
