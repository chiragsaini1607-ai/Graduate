import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load('admission_model.pkl')

st.title("Graduate Admission Predictor")

# Input fields
gre = st.slider("GRE Score", 290, 340, 320)
toefl = st.slider("TOEFL Score", 92, 120, 110)
rating = st.slider("University Rating", 1, 5, 4)
sop = st.slider("SOP Strength", 1.0, 5.0, 4.0, 0.5)
lor = st.slider("LOR Strength", 1.0, 5.0, 4.0, 0.5)
cgpa = st.slider("CGPA", 6.8, 9.92, 8.5, 0.01)
research = st.selectbox("Research Experience", [0, 1])

if st.button("Predict"):
    model = load_model()
    features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    prediction = model.predict(features)[0]
    
    st.success(f"Admission Chance: {prediction:.2f} ({prediction*100:.1f}%)")
