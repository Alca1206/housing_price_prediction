import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd 
import joblib


class HousePricePredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class HouseData(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


app = FastAPI()

try:
    package = joblib.load("../model/full_model_package.pkl")
    scaler = package['scaler']
    model_columns = package['columns']
    state_dict = package['model_state']
    model = HousePricePredictor(input_dim=len(model_columns))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error : {e}")
    
@app.get("/")
def home():
    return {
        "message": "Welcome to the King County House Price API!",
        "status": "Running",
        "instructions": "Go to /docs to test the model."
    }
@app.post("/predict")
def predict_price(house_data: HouseData):
    df = pd.DataFrame([house_data.dict()])

    df_encoded = pd.get_dummies(df, columns=["zipcode"])
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    scaled_data = scaler.transform(df_encoded)
    input_tensor = torch.tensor(scaled_data).float()   
    with torch.no_grad():
        prediction = model(input_tensor)

    return {"predicted_price": prediction.item()}