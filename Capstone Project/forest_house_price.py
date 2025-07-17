import streamlit as st
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from collections import Counter
import scipy.stats as stats
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd





# The Model
home_prices_df = pd.read_csv("Zillow Home Data.csv")
hpi_df = pd.read_csv("housing_price_index.csv")
home_sales_count = pd.read_csv("Sold_Homes_US.csv")
mortgage_data = pd.read_csv("mortgage_rates.csv")


Fayetteville_home_price_DF = home_prices_df[(home_prices_df["StateName"] == "NC") & (home_prices_df["RegionName"] == "Fayetteville, NC")]
hpi_df = pd.read_csv("housing_price_index.csv")

east_north = hpi_df[hpi_df['place_name'] == "East North Central Division"] 
monthly = east_north[east_north["frequency"] == 'monthly']
clean_hpi_data = monthly.drop(['hpi_type', 'hpi_flavor', 'frequency', 'level', 'place_name', 'place_id'], axis= 1)

fayetteville_prices = Fayetteville_home_price_DF.drop(columns=["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]).T
fayetteville_prices.columns = ["Price"]
fayetteville_prices.index = pd.to_datetime(fayetteville_prices.index)

Fayetteville_home_sales_count = home_sales_count[(home_prices_df["StateName"] == "NC")]
Fay_mean_sales_by_month = Fayetteville_home_sales_count.mean(axis=0, numeric_only=True).to_frame().T
Fay_mean_sales_by_month = Fay_mean_sales_by_month.drop(["RegionID", "SizeRank"], axis=1)
Fay_mean_sales_by_month.columns = pd.to_datetime(Fay_mean_sales_by_month.columns, )
test_df = Fay_mean_sales_by_month.T
Fay_mean_sales_by_month_T = Fay_mean_sales_by_month.T
Fay_mean_sales_by_month_T = Fay_mean_sales_by_month_T.reset_index()
Fay_mean_sales_by_month_T.columns = ['Date', 'Sales Per Month']  # Rename columns


clean_df = Fayetteville_home_price_DF.drop(["RegionID", "SizeRank", "RegionName", "StateName", "RegionType"], axis=1).T
cleaner_df = clean_df.reset_index()
df = cleaner_df.rename(columns={106: 'price', 'index': 'Date'})
df['Date'] = pd.to_datetime(df['Date'])                   
df["year"] = df['Date'].dt.year

# annual_avg = df.groupby('year')['price'].mean().reset_index()
rates_and_cost = pd.concat([mortgage_data, fayetteville_prices], ignore_index=True)

# * Datasets to pull from:
# 1. df = average prices in Fayetteville by month
# 2. annual_avg - average prices in Fayetteville by year 
# 3. mortgage_data = mortgage interest rate 30 year fixed by month since Jan 2000
# 4. Fay_mean_sales_by_month = transaction volume broked down by month since Feb 2008

# * conflict_periods_2 = [
    
#     ("2008-02-29", "2011-12-31", "War on Terror"),
    
#     ("2008-02-29", "2009-06-30", "Great Recession"),
    
#     ("2011-01-01", "2014-12-31", "Arab Spring"),
    
#     ("2014-02-01", "2014-12-31", "Crimea Annexation"),
    
#     ("2018-07-01", "2019-12-31", "US-China Trade War"),
    
#     ("2020-03-01", "2022-06-30", "COVID-19"),
    
#     ("2022-02-01", "2025-01-01", "Russia-Ukraine War"),
    
#     ("2023-10-01", "2025-01-01", "Israel–Hamas Escalation")]

Fay_mean_sales_by_month = Fay_mean_sales_by_month.T
Fay_mean_sales_by_month = Fay_mean_sales_by_month.reset_index()
Fay_mean_sales_by_month.columns = ['Date', 'Sales_Volume']

mortgage_data['observation_date'] = pd.to_datetime(mortgage_data['observation_date'])

mortgage_data.rename(columns={'observation_date': 'Date'}, inplace=True)

combined_df = df.merge(mortgage_data, on='Date', how='outer')\
                .merge(Fay_mean_sales_by_month, on='Date', how='outer')


# Make sure you have a 'year' column
combined_df['year'] = combined_df['Date'].dt.year

# Fill NaN in SalesVolume with mean SalesVolume of that year
combined_df['Sales_Volume'] = combined_df.groupby('year')['Sales_Volume']\
    .transform(lambda x: x.fillna(x.mean()))


combined_df['price'] = combined_df.groupby('year')['price'].transform(lambda x: x.fillna(x.mean()))

combined_df['MORTGAGE30US'] = combined_df.groupby('year')['MORTGAGE30US'].transform(lambda x: x.fillna(x.mean()))

#  1 = Economic 
#  2 = US War
#  3 = International Conlfict

conflict_periods_years = [
    (2008, 2009, 1),  # Great Recession → Economic
    (2008, 2011, 2),  # War on Terror → US War
    (2011, 2014, 3),  # Arab Spring → International Conflict
    (2014, 2014, 3),  # Crimea Annexation → International Conflict
    (2018, 2019, 1),  # US-China Trade War → Economic
    (2020, 2022, 1),  # COVID → Economic
    (2022, 2025, 3),  # Russia-Ukraine → International Conflict
    (2023, 2025, 3)   # Israel–Hamas → International Conflict
]


# Ensure you have a 'year' column
combined_df['year'] = combined_df['Date'].dt.year

# Initialize the conflict column with default value, e.g. 0 (no conflict)
combined_df['conflict_type'] = 0

# Iterate over conflict periods and assign codes
for start_year, end_year, code in conflict_periods_years:
    mask = (combined_df['year'] >= start_year) & (combined_df['year'] <= end_year)
    combined_df.loc[mask, 'conflict_type'] = code


combined_df_post_2008_with_sales_volume = combined_df[combined_df['Date'] >= '2008-06-30']

combined_df_no_sales_volume = combined_df[combined_df['Date'] <= '2008-06-30']
combined_df_no_sales_volume = combined_df_no_sales_volume.drop('Sales_Volume', axis = 1)

combined_df_post_2008_with_sales_volume['Month'] = combined_df_post_2008_with_sales_volume['Date'].dt.month
combined_df_post_2008_with_sales_volume['Day'] = combined_df_post_2008_with_sales_volume['Date'].dt.day

combined_df_post_2008_with_sales_volume = combined_df_post_2008_with_sales_volume.drop('Date', axis = 1)


#  PR3EDICTING 
X_sales_volume = combined_df_post_2008_with_sales_volume.drop('price', axis=1)
y_sales_volume = combined_df_post_2008_with_sales_volume['price']

X_volume_train, X_volume_test, y_volume_train, y_volume_test = train_test_split(X_sales_volume, y_sales_volume, test_size=.2)


# Preprocessing
numeric_features = ['year', 'MORTGAGE30US', 'Sales_Volume', 'Month', 'Day']
categorical_features = ['conflict_type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit model
rf_pipeline.fit(X_volume_train, y_volume_train)

# Evaluate
rf_predictions = rf_pipeline.predict(X_volume_test)
rf_r2 = r2_score(y_volume_test, rf_predictions)
rf_mae = mean_absolute_error(y_volume_test, rf_predictions)

# Streamlit UI
st.title("🏡 Random Forest Housing Price Predictor (Fayetteville, NC)")

st.write(f"### Model Performance:\n- R2 Score: {rf_r2:.2f}\n- MAE: {rf_mae:.2f}")

# Streamlit Input Fields
Interest_Rate = st.slider("Mortgage Interest Rate (%)", 0.0, 10.0, 5.0)
Starting_Price = st.slider("Starting Price (not used in prediction)", 0.0, 1_000_000.0, 250_000.0)  # Optional / not used in model
Year = st.slider("Year", 2025, 2100, 2025)
Month = st.slider("Month", 1, 12, 6)
Day = st.slider("Day", 1, 31, 15)
monthly_home_sales = st.slider("Monthly Home Sales Volume", 500.0, 5000.0, 2000.0)
conflict_type = st.selectbox("Conflict Type", options=[0, 1, 2, 3],
    format_func=lambda x: ["No Issue", "Economic Instability", "US War", "International Conflict"][x]
)

input_data = pd.DataFrame([{
    'MORTGAGE30US': Interest_Rate,
    'Sales_Volume': monthly_home_sales,
    'conflict_type': conflict_type,
    'year': Year,
    'Month': Month,
    'Day': Day
}])

if st.button("Show Feature Importances"):
    # Get feature names from preprocessing pipeline
    onehot_columns = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['conflict_type'])
    
    all_numeric = numeric_features
    all_features = list(all_numeric) + list(onehot_columns)

    importances = rf_pipeline.named_steps['model'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    st.write("### Feature Importances")
    st.bar_chart(feature_importance_df.set_index('Feature'))