import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

current_dir = Path(__file__).resolve(strict=True).parent
model_path = current_dir / '..' / 'model' / 'xgb_actual_cost_model.pkl'
# Check if the model file exists
if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load trained model
model = joblib.load(model_path)

st.set_page_config(page_title="Project Cost Predictor", layout="centered")

st.title("Project Cost Predictor")
st.write("Estimate the actual cost of a new project based on sustainability and project details.")

# --- User Inputs ---
industry = st.selectbox("Industry", ["Biotech", "Medtech", "Water", "Heavy Industry"])
budget = st.number_input("Initial Budget (£)", min_value=10000, step=10000, value=100000)
duration = st.slider("Project Duration (months)", 1, 36, 12)
sustainability = st.slider("Sustainability Score (1–100)", 1, 100, 70)
client_satisfaction = st.slider("Client Satisfaction (1–5)", 1, 5, 4)
resource_index = st.slider("Resource Usage Index (1–10)", 1.0, 10.0, 5.0)

# --- One-hot Encoding for Industry ---
industries = ['Industry_Biotech', 'Industry_Heavy Industry', 'Industry_Medtech', 'Industry_Water']
industry_encoded = [1 if f"Industry_{industry}" == col else 0 for col in industries]

# --- Define Feature Order as in Training ---
trained_feature_order = [
    'Initial_Budget',
    'Project_Duration',
    'Sustainability_Score',
    'Client_Satisfaction',
    'Resource_Usage_Index',
    'Industry_Biotech',
    'Industry_Heavy Industry',
    'Industry_Medtech',
    'Industry_Water'
]

# --- Create Input DataFrame in Correct Order ---
input_values = [
    budget,
    duration,
    sustainability,
    client_satisfaction,
    resource_index,
    *industry_encoded
]

input_df = pd.DataFrame([input_values], columns=trained_feature_order)

# --- Prediction ---
if st.button("Predict Cost"):
    predicted_cost = model.predict(input_df)[0]
    st.success(f"Predicted Project Cost: £{predicted_cost:,.2f}")
    st.caption("Prediction generated using a trained XGBoost model on synthetic data.")
