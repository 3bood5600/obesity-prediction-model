import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------------------------------------------------------
# App Title and Description
# -----------------------------------------------------------------------------
st.title("🫀 Obesity Level Predictor")
st.markdown("Enter your lifestyle and physical details to predict your obesity risk level.")

# -----------------------------------------------------------------------------
# Load Models and Artifacts (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e.filename}. Make sure it's in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

model, scaler, feature_names = load_models()

# Class names must match those used during training
class_names = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

# -----------------------------------------------------------------------------
# User Input Form (Two-column layout)
# -----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("👤 Your Information")

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

    submitted = st.form_submit_button("🔮 Predict Obesity Level")

# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
if submitted:
    # Calculate BMI
    bmi = round(weight / (height ** 2), 2)
    st.write(f"📏 **Your BMI:** {bmi} (kg/m²)")

    # Prepare input data with correct one-hot encoding (lowercase matching)
    input_data = pd.DataFrame({
        "Gender_Male": [1 if gender == "Male" else 0],
        "family_history_with_overweight_yes": [1 if family_history == "yes" else 0],
        "FAVC_yes": [1 if favc == "yes" else 0],
        "SMOKE_yes": [1 if smoke == "yes" else 0],
        "SCC_yes": [1 if scc == "yes" else 0],
        "CAEC_Always": [1 if caec == "Always" else 0],
        "CAEC_Frequently": [1 if caec == "Frequently" else 0],
        "CAEC_Sometimes": [1 if caec == "Sometimes" else 0],
        "CALC_Always": [1 if calc == "Always" else 0],
        "CALC_Frequently": [1 if calc == "Frequently" else 0],
        "CALC_Sometimes": [1 if calc == "Sometimes" else 0],
        "MTRANS_Bike": [1 if mtrans == "Bike" else 0],
        "MTRANS_Motorbike": [1 if mtrans == "Motorbike" else 0],
        "MTRANS_Public_Transportation": [1 if mtrans == "Public_Transportation" else 0],
        "MTRANS_Walking": [1 if mtrans == "Walking" else 0],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CH2O": [ch2o],
        "FAF": [faf],
        "TUE": [tue],
        "BMI": [bmi]
    })

    # Ensure column order and add missing columns as 0
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Scale the input
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error(f"❌ Error during scaling: {e}")
        st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = probabilities.max()
        predicted_class = class_names[prediction]

        # -----------------------------------------------------------------------------
        # Display Results
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.success(f"**Obesity Level:** {predicted_class}")
        st.info(f"**Confidence Score:** {confidence:.2%}")

        # Show probability distribution
        prob_df = pd.DataFrame({
            "Obesity Level": class_names,
            "Probability": probabilities
        })
        fig = px.bar(
            prob_df,
            x="Obesity Level",
            y="Probability",
            color="Probability",
            color_continuous_scale="Blues",
            title="Prediction Confidence per Class"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig)

        # -----------------------------------------------------------------------------
        # Health Tips
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("💡 Health Recommendation")

        if "Normal_Weight" in predicted_class:
            st.balloons()
            st.write("✅ **Great job!** You're in a healthy weight range. Keep up the good habits!")
        elif "Obesity" in predicted_class:
            st.warning("⚠️ High risk detected. Consider increasing physical activity, improving diet, and consulting a nutritionist or doctor.")
        elif "Overweight" in predicted_class:
            st.info("🏋️ You're in the overweight range. Small lifestyle changes can make a big difference. Try walking more and reducing high-calorie snacks.")
        else:
            st.info("🥗 You're underweight. Consider consulting a dietitian to ensure you're getting enough nutrients.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
