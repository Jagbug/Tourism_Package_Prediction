import os
import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# 1. STREAMLIT APP CONFIGURATION
# ==========================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_package_model.joblib")
CLASSIFICATION_THRESHOLD = 0.45

# Page Title and Subtitle
st.set_page_config(page_title="Tourism Package Predictor", layout="centered")
st.title("🌲 Visit with Us - Wellness Tourism Package Predictor")
st.write("Enter customer parameters below to calculate purchase probability and prediction targets.")

# ==========================================================
# 2. LOAD PRE-TRAINED MODEL PIPELINE
# ==========================================================
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Predictive Model Pipeline loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==========================================================
# 3. USER INPUT CONFIGURATION INTERFACE
# ==========================================================
st.subheader("📊 Customer Profiling & Interaction Inputs")
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.slider("Duration of Pitch (minutes)", 0, 100, 15)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
    MonthlyIncome = st.number_input("Monthly Income (INR)", min_value=1000.0, value=30000.0)

# ==========================================================
# 4. PREPARE DATAFRAME & PREDICT
# ==========================================================
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

if st.button("🔮 Predict Purchase Likelihood"):
    prediction_probability = model.predict_proba(input_data)[0, 1]
    class_decision = int(prediction_probability >= CLASSIFICATION_THRESHOLD)
    
    st.markdown("--- ")
    st.subheader("Prediction Results")
    st.metric(label="Calculated Purchase Probability", value=f"{prediction_probability * 100:.2f}%")
    
    if class_decision == 1:
        st.success("🎯 Prediction: Customer **WILL PURCHASE** the Wellness Tourism Package!")
    else:
        st.warning("⚠️ Prediction: Customer is **UNLIKELY** to purchase the travel package.")
