import streamlit as st

st.title("Customer Churn Prediction")
st.success("Streamlit is Working!")
import streamlit as st
import joblib

st.title("Customer Churn Prediction")

try:
    model = joblib.load("model.pkl")
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error("Model Loading Failed")
    st.exception(e)
