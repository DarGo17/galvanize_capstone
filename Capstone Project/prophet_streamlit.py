import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib

# ================= Load pre-trained Prophet model and historical data ====================
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

# ========================================== Inputs =======================================
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

# ====================================== Plot =========================================
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

#  conflict period highlight
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

        # price change on graph
        ax.annotate(f'{price_change_pct:.2f}% Price Change', xy=(conflict_end, conflict_prices.iloc[-1]),
                    xytext=(conflict_end, conflict_prices.max()),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)

# mortgage interest rate line
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

# ================================ Download Forecast Data ==========================================
csv = forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button(label="Download CSV", data=csv, file_name='prophet.csv', mime='text/csv')

###############################################################################################################


# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import json
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta

# # ========== Setup for Property Valuation Tab ==========
# API_KEY = "bc87829e1d2d4ee68dcbb775c90b598a"
# VALUE_URL = "https://api.rentcast.io/v1/avm/value"
# RENT_URL = "https://api.rentcast.io/v1/avm/rent/long-term"
# headers = {"X-Api-Key": API_KEY}

# def fetch_property_value(address):
#     params = {"address": address}
#     response = requests.get(VALUE_URL, headers=headers, params=params)
#     return response.json() if response.status_code == 200 else None

# def fetch_rent_estimate(address):
#     params = {"address": address}
#     response = requests.get(RENT_URL, headers=headers, params=params)
#     return response.json() if response.status_code == 200 else None

# def generate_trend(current_value, growth_rate=0.03, years=10):
#     dates = [datetime.now() - timedelta(days=365 * i) for i in reversed(range(years))]
#     values = [current_value / ((1 + growth_rate) ** (years - i - 1)) for i in range(years)]
#     return pd.DataFrame({'Year': [d.year for d in dates], 'Value': values})

# # ========== Main App ==========
# st.set_page_config(page_title="Real Estate Forecast & Valuation", layout="wide")
# tab1, tab2 = st.tabs(["📈 Forecasting Model", "🏠 Valuation & Rent Estimates"])

# # ========== Tab 1: Forecasting ==========
# with tab1:
#     st.title('Fayetteville Home Price Forecast with Economic and Conflict Regressors')

#     df = pd.read_csv('regressor_forecast.csv')
#     df['ds'] = pd.to_datetime(df['ds'])
#     df['y'] = df['yhat']
#     df.drop('yhat', axis=1, inplace=True)

#     if st.checkbox("Show Raw Data"):
#         st.write(df.head())

#     model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
#     for regressor in ['MORTGAGE30US', 'Sales_Volume', 'conflict_type']:
#         model.add_regressor(regressor)

#     model.fit(df[['ds', 'y', 'MORTGAGE30US', 'Sales_Volume', 'conflict_type']])

#     periods = st.slider('Months to Forecast', min_value=12, max_value=240, value=120, step=12)
#     future = model.make_future_dataframe(periods=periods, freq='M')

#     st.sidebar.header("Future Assumptions")
#     future_mortgage = st.sidebar.slider('Future Mortgage Rate (%)', 3.0, 10.0, 6.0, step=0.1)
#     future_sales_vol = st.sidebar.slider('Future Sales Volume', 500, 5000, 1500, step=100)
#     future_conflict = st.sidebar.selectbox('Future Conflict Type', options=[0, 1, 2, 3],
#                                            format_func=lambda x: ['None', 'Economic', 'US War', 'International Conflict'][x])

#     future['MORTGAGE30US'] = future['ds'].apply(lambda _: future_mortgage)
#     future['Sales_Volume'] = future['ds'].apply(lambda _: future_sales_vol)
#     future['conflict_type'] = future['ds'].apply(lambda _: future_conflict)

#     future = future.merge(df[['ds', 'MORTGAGE30US', 'Sales_Volume', 'conflict_type']],
#                           on='ds', how='left', suffixes=('', '_hist'))

#     for reg in ['MORTGAGE30US', 'Sales_Volume', 'conflict_type']:
#         future[reg] = future[reg].combine_first(future[f'{reg}_hist'])
#         future.drop(columns=[f'{reg}_hist'], inplace=True)

#     forecast = model.predict(future)

#     fig1 = model.plot(forecast)
#     st.pyplot(fig1)

#     if st.checkbox('Show Forecast Components'):
#         fig2 = model.plot_components(forecast)
#         st.pyplot(fig2)

#     if st.button('Download Forecast Data'):
#         forecast.to_csv('forecast_with_regressors.csv', index=False)
#         st.success('Forecast data saved as forecast_with_regressors.csv')

# # ========== Tab 2: Valuation ==========
# with tab2:
#     st.title("🏡 Property Valuation & Rent Insights Dashboard")

#     address = st.text_input("Enter Full Property Address", "3821 Hargis St, Austin, TX 78723")

#     if address.strip():
#         st.info(f"📡 Fetching data for **{address}**...")

#         value_data = fetch_property_value(address)
#         rent_data = fetch_rent_estimate(address)

#         if value_data is None or rent_data is None:
#             st.error("Failed to retrieve property or rent data.")
#         else:
#             tabA, tabB, tabC = st.tabs(["🏷️ Valuation", "📊 Trends", "🧾 API Raw Output"])

#             with tabA:
#                 price = value_data.get("price", 0)
#                 rent = rent_data.get("rent", 0)

#                 st.metric("Estimated Home Value", f"${price:,.0f}")
#                 st.caption(f"Range: ${value_data.get('priceRangeLow', 0):,.0f} - ${value_data.get('priceRangeHigh', 0):,.0f}")

#                 st.metric("Estimated Monthly Rent", f"${rent:,.0f}")
#                 st.caption(f"Range: ${rent_data.get('rentRangeLow', 0):,.0f} - ${rent_data.get('rentRangeHigh', 0):,.0f}")

#             with tabB:
#                 st.subheader("Simulated Historical Trends")
#                 value_trend = generate_trend(price, growth_rate=0.035)
#                 rent_trend = generate_trend(rent, growth_rate=0.025)

#                 st.markdown("#### Home Value Trend")
#                 st.line_chart(value_trend.set_index('Year'))

#                 st.markdown("#### Rent Trend")
#                 st.line_chart(rent_trend.set_index('Year'))

#                 st.caption("Trends are approximated using average appreciation assumptions.")

#             with tabC:
#                 st.subheader("Raw API Responses")
#                 st.json(value_data)
#                 st.json(rent_data)
#     else:
#         st.warning("⚠️ Please enter a valid address to begin.")

