# app.py
# Obesity Level Prediction App using Streamlit
# Author: Your Name
# Date: 2025

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Load trained models and artifacts
# -------------------------------
try:
    model = joblib.load("best_model.pkl")           # Trained classifier
    scaler = joblib.load("scaler.pkl")              # Fitted StandardScaler
    feature_names = joblib.load("feature_names.pkl") # Feature names after preprocessing
except FileNotFoundError as e:
    st.error(f"Error: Required file not found. Make sure '{e.filename}' is in the current directory.")
    st.stop()

# -------------------------------
# Class labels (must match training)
# -------------------------------
class_names = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🫀 Obesity Level Predictor")
st.markdown("Enter your details below to predict your obesity risk level.")

# Split input fields into two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=10, max_value=90, value=25)
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    gender = st.radio("Gender", ["Male", "Female"])
    family_history = st.radio("Family history of overweight?", ["no", "yes"])
    favc = st.radio("Frequently eats high-calorie food?", ["no", "yes"])
    fcvc = st.slider("Vegetable consumption frequency (1-3)", 1, 3, 2)
    ncp = st.number_input("Number of main meals per day", min_value=1, max_value=5, value=3)

with col2:
    caec = st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.radio("Do you smoke?", ["no", "yes"])
    ch2o = st.number_input("Daily water intake (liters)", min_value=0.5, max_value=4.0, value=2.0)
    scc = st.radio("Do you monitor calorie intake?", ["no", "yes"])
    faf = st.number_input("Physical activity frequency (days/week)", min_value=0, max_value=7, value=2)
    tue = st.slider("Time using electronic devices (scale 0-2)", 0, 2, 1)
    calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "Main mode of transportation",
        ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
    )

# -------------------------------
# Calculate BMI from height and weight
# -------------------------------
bmi = round(weight / (height ** 2), 2)
st.write(f"**Calculated BMI:** {bmi}")

# -------------------------------
# Encode categorical inputs to match training format
# -------------------------------
input_data = pd.DataFrame({
    "Gender": [1 if gender == "Male" else 0],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "family_history_with_overweight": [1 if family_history == "yes" else 0],
    "FAVC": [1 if favc == "yes" else 0],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC_Always": [1 if caec == "Always" else 0],
    "CAEC_Frequently": [1 if caec == "Frequently" else 0],
    "CAEC_Sometimes": [1 if caec == "Sometimes" else 0],
    "SMOKE": [1 if smoke == "yes" else 0],
    "CH2O": [ch2o],
    "SCC": [1 if scc == "yes" else 0],
    "FAF": [faf],
    "TUE": [tue],
    "CALC_Always": [1 if calc == "Always" else 0],
    "CALC_Frequently": [1 if calc == "Frequently" else 0],
    "CALC_Sometimes": [1 if calc == "Sometimes" else 0],
    "MTRANS_Bike": [1 if mtrans == "Bike" else 0],
    "MTRANS_Motorbike": [1 if mtrans == "Motorbike" else 0],
    "MTRANS_Public_Transportation": [1 if mtrans == "Public_Transportation" else 0],
    "MTRANS_Walking": [1 if mtrans == "Walking" else 0],
    "BMI": [bmi]
})

# -------------------------------
# Reindex to match training features (add missing columns as 0)
# -------------------------------
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# -------------------------------
# Apply scaling using the fitted StandardScaler
# -------------------------------
try:
    input_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()

# -------------------------------
# Make prediction
# -------------------------------
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0].max()  # Confidence score

# -------------------------------
# Display result
# -------------------------------
st.markdown("---")
st.subheader("📊 Prediction Result")
st.success(f"**Predicted Obesity Level:** {class_names[prediction]}")
st.info(f"**Confidence:** {probability:.2%}")
st.info(f"**BMI:** {bmi}")

# -------------------------------
# Health tip based on result
# -------------------------------
if "Normal" in class_names[prediction]:
    st.balloons()
    st.write("✅ Great job! You're in a healthy weight range.")
elif "Obesity" in class_names[prediction]:
    st.warning("💡 Consider increasing physical activity and consulting a nutritionist.")
else:
    st.info("💡 Maintain healthy habits to stay on track.")
