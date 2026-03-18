import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="Group 1 part time Clinical Portal", layout="wide", page_icon="🧠")

# --- Model Training/Loading Logic ---
@st.cache_resource
def get_model_assets():
    model_path = 'group5_stroke_svm_model.pkl'
    data_file = 'healthcare-dataset-stroke-data.csv'
    
    # If model doesn't exist, we train it on the fly using the dataset
    if not os.path.exists(model_path):
        if not os.path.exists(data_file):
            st.error(f"Error: {data_file} not found. Please ensure it is in the directory.")
            return None
            
        df_train = pd.read_csv(data_file)
        df_train['bmi'] = df_train['bmi'].fillna(df_train['bmi'].median())
        df_train = df_train.drop(columns=['id'])
        
        encoders = {}
        cat_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status']
        for col in cat_cols:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            encoders[col] = le
            
        X = df_train.drop(columns=['stroke'])
        y = df_train['stroke']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        
        model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y)
        
        assets = {'model': model, 'scaler': scaler, 'encoders': encoders, 'columns': X.columns.tolist()}
        joblib.dump(assets, model_path)
        return assets
    
    return joblib.load(model_path)

assets = get_model_assets()

# --- Helper Functions ---
def save_patient_data(patient_name, input_df, pred_type, score, risk_lvl):
    file_path = 'patient_records.csv'
    new_record = input_df.copy()
    new_record['patient_name'] = patient_name
    new_record['prediction_type'] = pred_type
    new_record['score'] = score
    new_record['risk_level'] = risk_lvl
    new_record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(file_path)
    new_record.to_csv(file_path, mode='a', header=not file_exists, index=False)

def engineer_features(age, glucose, bmi_val):
    age_grp = 'young' if age <= 35 else 'middle' if age <= 55 else 'senior'
    bmi_grp = 'underweight' if bmi_val < 18.5 else 'normal' if bmi_val < 25 else 'overweight' if bmi_val < 30 else 'obese'
    glu_grp = 'normal' if glucose < 100 else 'prediabetes' if glucose < 126 else 'diabetes'
    return age_grp, glu_grp, bmi_grp

# --- Branding Header ---
st.markdown("<div style='text-align: center;'><h1>🧠 Stroke&heart Risk Clinical Portal</h1><h3>BOUESTI CIS STUDENT GROUP 1 part time PROJECT</h3></div>", unsafe_allow_html=True)
st.divider()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Prediction Portal", "📊 Data Dashboard", "🗃️ Patient Records"])

# --- TAB 1: PREDICTION PORTAL ---
with tab1:
    with st.sidebar:
        st.header("👤 Patient Demographics")
        patient_name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 0, 120, 45)
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.header("🏥 Clinical Metrics")
        hypertension = st.radio("Hypertension History?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        heart_disease = st.radio("Heart Disease?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        sys_bp = st.number_input("Systolic BP", 70, 250, 120)
        dia_bp = st.number_input("Diastolic BP", 40, 150, 80)
        glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 95.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        
        st.header("🚬 Lifestyle & History")
        work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoke = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        diabetes = st.checkbox("Known Diabetes")
        ckd = st.checkbox("Kidney Disease (CKD)")
        stress = st.checkbox("High Stress Levels")

    st.subheader("Run Diagnostic Analysis")
    if st.button("🚀 Calculate Risk Profile"):
        if not patient_name:
            st.error("Please provide a patient name.")
        elif assets is None:
            st.error("Model assets not loaded properly.")
        else:
            # 1. Prepare Input for SVM
            input_dict = {
                'gender': gender, 'age': age, 'hypertension': hypertension, 
                'heart_disease': heart_disease, 
                'work_type': work, 'Residence_type': residence,
                'avg_glucose_level': glucose, 'bmi': bmi,
                'smoking_status': smoke
            }
            input_df = pd.DataFrame([input_dict])
            
            # 2. Encode categorical data
            processed_df = input_df.copy()
            for col, le in assets['encoders'].items():
                val = str(processed_df[col].iloc[0])
                processed_df[col] = le.transform([val])[0] if val in le.classes_ else 0
            
            # 3. Scaling & Prediction (The Fix)
            X_scaled = assets['scaler'].transform(processed_df)
            probability = assets['model'].predict_proba(X_scaled)[0][1] * 100
            
            # 4. Clinical Adjustments (Heuristic factors)
            bp_adj = 15 if (sys_bp >= 140 or dia_bp >= 90) else 0
            final_score = min(100, probability + bp_adj + (int(ckd) * 10) + (int(stress) * 5))
            
            # 5. Result Formatting
            if final_score > 75:
                risk_lvl, color = "CRITICAL", "red"
            elif final_score > 40:
                risk_lvl, color = "ELEVATED", "orange"
            else:
                risk_lvl, color = "STABLE", "green"

            # 6. Display Results
            st.divider()
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Base Probability", f"{probability:.1f}%")
            res_c2.metric("Adjusted Clinical Score", f"{final_score:.1f}%")
            res_c3.markdown(f"**Risk Level:** <h2 style='color:{color};'>{risk_lvl}</h2>", unsafe_allow_html=True)
            
            save_patient_data(patient_name, input_df, "Stroke", round(final_score, 2), risk_lvl)
            st.success(f"Risk analysis for {patient_name} has been saved to records.")

# --- TAB 2: DATA DASHBOARD ---
with tab2:
    st.header("Dataset Overview")
    if os.path.exists('healthcare-dataset-stroke-data.csv'):
        df_view = pd.read_csv('healthcare-dataset-stroke-data.csv')
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("### Stroke Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='stroke', data=df_view, palette='viridis', ax=ax1)
            st.pyplot(fig1)
            
        with col_b:
            st.write("### Age vs BMI Analysis")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x='age', y='bmi', hue='stroke', data=df_view, alpha=0.5, ax=ax2)
            st.pyplot(fig2)
    else:
        st.warning("CSV data file not found for dashboard view.")


# --- TAB 3: PATIENT RECORDS ---
with tab3:
    st.header("Clinical History Database")
    if os.path.exists('patient_records.csv'):
        history_df = pd.read_csv('patient_records.csv')
        st.dataframe(history_df, use_container_width=True)
        if st.button("🗑️ Clear All Records"):
            os.remove('patient_records.csv')
            st.rerun()
    else:
        st.info("No clinical records found in the database.")
