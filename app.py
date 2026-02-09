import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Loan Approval System")

st.title("💰 Loan Approval Prediction")

model = joblib.load("loan_model.pkl")

st.sidebar.header("Enter Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male","Female"])
married = st.sidebar.selectbox("Married", ["Yes","No"])
education = st.sidebar.selectbox("Education", ["Graduate","Not Graduate"])
income = st.sidebar.number_input("Applicant Income", 1000, 50000, 5000)
loan = st.sidebar.number_input("Loan Amount", 10, 1000, 100)
credit = st.sidebar.selectbox("Credit History", [0,1])
area = st.sidebar.selectbox("Property Area", ["Urban","Rural","Semiurban"])

# Encode manually same order as training
input_data = pd.DataFrame([[
    1 if gender=="Male" else 0,
    1 if married=="Yes" else 0,
    1 if education=="Graduate" else 0,
    income,
    loan,
    credit,
    {"Urban":2,"Semiurban":1,"Rural":0}[area]
]])

if st.sidebar.button("Predict"):
    result = model.predict(input_data)[0]

    if result == 0:
        st.error("❌ Loan Rejected")
    else:
        st.success("✅ Loan Approved")
