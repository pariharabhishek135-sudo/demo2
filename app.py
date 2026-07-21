import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction")

model = joblib.load("model.pkl")

CreditScore = st.number_input("Credit Score", 300, 900, 650)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", 18, 100, 30)
Tenure = st.number_input("Tenure", 0, 10, 5)
Balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)

if st.button("Predict"):
    data = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary]
    })

    try:
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("Customer Will Churn")
        else:
            st.success("Customer Will Not Churn")

    except Exception as e:
        st.error("Prediction Error")
        st.exception(e)
