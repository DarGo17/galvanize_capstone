# === Imports === 

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, root_mean_squared_error
from collections import Counter
import scipy.stats as stats
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import matplotlib.pyplot as plt


#  === Import Prophet Model ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('linear_price_model.pkl')

class PredictionInput(BaseModel):
    MORTGAGE30US: float
    Sales_Volume: float
    conflict_type: int
    year: int
    Month: int
    Day: int

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        data = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))