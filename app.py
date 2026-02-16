import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="CNC Tool Wear", layout="wide")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('tool_wear_advanced.pkl')
    features = joblib.load('best_features.pkl')
    return model, features

model, feature_names = load_models()

st.title(" CNC Tool Wear Predictor")
st.markdown("**Production-ready model: 56.7% CV Accuracy** (18 experiments)")

# FIXED: Direct input widgets for TOP 5 features
col1, col2, col3 = st.columns(3)
with col1:
    x1_acc_std = st.number_input("**X1_ActualAcceleration_std**", value=0.1, step=0.01)
with col2:
    x1_curr_std = st.number_input("**X1_OutputCurrent_std**", value=0.05, step=0.01)
with col3:
    y1_acc_std = st.number_input("**Y1_ActualAcceleration_std**", value=0.2, step=0.01)

col1, col2 = st.columns(2)
with col1:
    x1_acc_var = st.number_input("**X1_ActualAcceleration_var**", value=0.02, step=0.01)
with col2:
    x1_curr_var = st.number_input("**X1_OutputCurrent_var**", value=0.01, step=0.01)

# Create input vector (FIXED)
input_data = pd.DataFrame(0.0, index=[0], columns=feature_names)
input_data.at[0, 'X1_ActualAcceleration_std'] = x1_acc_std
input_data.at[0, 'X1_OutputCurrent_std'] = x1_curr_std
input_data.at[0, 'Y1_ActualAcceleration_std'] = y1_acc_std
input_data.at[0, 'X1_ActualAcceleration_var'] = x1_acc_var
input_data.at[0, 'X1_OutputCurrent_var'] = x1_curr_var

# Predict
if st.button(" PREDICT TOOL WEAR", type="primary"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**Prediction**", "UNWORN" if prediction == 0 else "WORN")
    with col2:
        st.metric("**Confidence**", f"{max(prob):.1%}")
    with col3:
        st.metric("**Model CV**", "56.7%")

st.markdown("---")
st.caption("**Portfolio Project: Raw data → 64 features → 10 best → 56.7% CV**")
