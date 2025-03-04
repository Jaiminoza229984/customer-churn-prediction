import streamlit as st
import requests

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", [0, 1])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
tenure = st.slider("Tenure", 1, 72, 30)
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)

if st.button("Predict Churn"):
    data = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "tenure": tenure, "MonthlyCharges": monthly_charges}
    response = requests.post("http://127.0.0.1:8000/predict/", json=data)
    st.json(response.json())
