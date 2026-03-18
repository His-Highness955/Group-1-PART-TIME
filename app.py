import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Group 1 Clinical Portal", layout="wide", page_icon="🧠")

# --- Model Loading Logic ---
@st.cache_resource
def get_model_assets():
    model_path = 'group5_stroke_svm_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

assets = get_model_assets()

# --- Helper Functions ---
def save_patient_data(patient_name, input_df, score, risk_lvl):
    file_path = 'patient_records.csv'
    new_record = input_df.copy()
    new_record.insert(0, 'patient_name', patient_name)
    new_record['score'] = score
    new_record['risk_level'] = risk_lvl
    new_record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(file_path)
    new_record.to_csv(file_path, mode='a', header=not file_exists, index=False)

# --- Branding Header ---
st.markdown("<div style='text-align: center;'><h1> Stroke & Heart Risk Clinical Portal</h1><h3>BOUESTI CIS STUDENT GROUP 1 PROJECT</h3></div>", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Prediction Portal", "📊 Data Dashboard", "🗃️ Patient Records"])

# --- TAB 1: PREDICTION PORTAL ---
with tab1:
    if assets is None:
        st.error("Model file 'group5_stroke_svm_model.pkl' not found.")
        st.stop()

    with st.sidebar:
        st.header("👤 Patient Demographics")
        patient_name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 0, 120, 45)
        # Marriage selectbox removed from UI
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.header("🏥 Clinical Metrics")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        heart_disease = st.radio("Heart Disease?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 95.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        
        st.header("Lifestyle & History")
        # Added 'Student' as requested
        work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked", "Student"])
        smoke = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        st.subheader("Additional Risk Factors")
        ckd = st.checkbox("Kidney Disease (CKD)")
        stress = st.checkbox("High Stress Levels")

    st.subheader("Run Diagnostic Analysis")
    if st.button("Calculate Risk Profile"):
        if not patient_name:
            st.warning("Please provide a patient name.")
        else:
            # Prepare Input (Marriage is removed from input_dict but handled in processed_df)
            input_dict = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': "Yes", # Default value sent to model to avoid KeyError
                'work_type': work,
                'Residence_type': residence,
                'avg_glucose_level': glucose,
                'bmi': bmi,
                'smoking_status': smoke
            }
            input_df = pd.DataFrame([input_dict])
            
            # Encoding categorical data
            processed_df = input_df.copy()
            for col, le in assets['encoders'].items():
                val = str(processed_df[col].iloc[0])
                # Handle 'Student' or unknown values
                if val not in le.classes_:
                    processed_df[col] = le.transform([le.classes_[0]])[0] 
                else:
                    processed_df[col] = le.transform([val])[0]
            
            # Scaling & Prediction
            X_scaled = assets['scaler'].transform(processed_df)
            probability = assets['model'].predict_proba(X_scaled)[0][1] * 100
            
            # Clinical Adjustments
            final_score = min(100, probability + (int(ckd) * 10) + (int(stress) * 5))
            
            # Result Formatting
            if final_score > 75:
                risk_lvl, color = "CRITICAL", "#FF0000"
            elif final_score > 40:
                risk_lvl, color = "ELEVATED", "#FFA500"
            else:
                risk_lvl, color = "STABLE", "#008000"

            # Display Results
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Base SVM Probability", f"{probability:.1f}%")
            c2.metric("Final Risk Score", f"{final_score:.1f}%")
            c3.markdown(f"**Risk Level:** <h2 style='color:{color};'>{risk_lvl}</h2>", unsafe_allow_html=True)
            
            save_patient_data(patient_name, input_df.drop(columns=['ever_married']), round(final_score, 2), risk_lvl)
            st.success(f"Analysis for {patient_name} saved.")

# --- Remaining tabs logic continues here ---
