from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('linear_price_model.pkl')

class HouseData(BaseModel):
    MORTGAGE30US: float
    Sales_Volume: float
    conflict_type: int
    year: int
    Day: int
    Month: int

app = FastAPI()

@app.post("/predict")
def predict(data: HouseData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)
    return {"predicted_price": prediction[0]}
