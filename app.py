
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Page configuration
st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="sonal-bhutani-ml11/california-housing-price-prediction",
        filename="california_housing_model.pkl"
    )
    return joblib.load(model_path)

model = load_model()

# App title
st.title("🏠 California House Price Predictor")
st.write("Enter the housing details to predict the median house value.")

# User inputs
MedInc = st.number_input("Median Income", min_value=0.0, value=3.0)
HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=36.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0)

# Prediction
if st.button("Predict House Price"):
    input_data = pd.DataFrame([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]], columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ])

    prediction = model.predict(input_data)[0]

    # Dataset target is measured in units of $100,000
    price_dollars = prediction * 100000

    st.success(f"Predicted Median House Value: ${price_dollars:,.2f}")
