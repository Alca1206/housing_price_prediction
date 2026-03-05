import streamlit as st
import requests
import pandas as pd


st.set_page_config(page_title="Housing Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 King County Housing Price Prediction")
st.write("Enter the details of the house below to get an AI-powered price estimate!")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Info")
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=33, value=None, placeholder="Enter number of bedrooms (e.g. 3)", step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=8.0, value=None, placeholder="Enter Number of bathrooms (e.g. 2.5)", step=0.25)   
    floors = st.number_input("Floors", min_value=1.0, max_value=3.5, value=None, placeholder="Enter Floors (e.g. 1.5)", step=0.5)
    zipcode = st.text_input("Zipcode", value=None, placeholder="Enter zipcode (e.g. 98001)")
    lat = st.number_input("Latitude", min_value=45.0, max_value=48.0, value=47.6, step=0.1)
    long = st.number_input("Longitude", min_value=-123.0, max_value=-120.0, value=-122.3, step=0.1)

with col2:
    st.subheader("Squeare Footage")
    sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=2000, step =10)
    sqft_lot = st.number_input("total land Size (sqft)", min_value=0, value=5000, step=10)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, value=500, step=10)
    sqft_living15 = st.number_input("15 Neighbors' Living Area (sqft)", min_value=0, value=2000, step=10)
    sqft_lot15 = st.number_input("15 Neighbors' total land Size (sqft)", min_value=0, value=5000, step=10)

with col3:
    st.subheader("condition & Age")
    condition = st.selectbox("Condition (1-5)", options=[1, 2, 3, 4, 5], index=2)
    grade = st.selectbox("Grade (1-13)", options=list(range(1, 14)), index=6)
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000, step=1)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2024, value=0, step=1)
    waterfront = st.selectbox("Waterfront (No or yes)", options=["No", "Yes"], index=0)
    view = st.selectbox("View (0-4)", options=[0, 1, 2, 3, 4], index=0)

st.markdown("---")
if st.button("🔮 Predict House Price", use_container_width=True):

    if None in [bedrooms, bathrooms, sqft_living, zipcode]: # Add all your variables here
        st.warning("⚠️ Please fill out all the boxes before predicting!")

    else:
        input_data = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": 1 if waterfront == "Yes" else 0,
            "view": view,
            "condition": condition,
            "grade": grade,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "zipcode": zipcode,
            "lat": lat,
            "long": long,
            "sqft_living15": sqft_living15,
            "sqft_lot15": sqft_lot15
     }
    
        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            if response.status_code == 200:
                result = response.json()
                price = result["predicted_price"]
                st.success(f"### Estimated Price: ${price:,.2f}")
            else:
                st.error(f"Error from API: {response.text}")
        except requests.RequestException as e:
            st.error("🚨 Could not connect to the FastAPI backend. Is your Uvicorn server running?")
        
    

