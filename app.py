import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main {
    background-color:#f5f7fa;
}
h1{
    color:#0f172a;
    text-align:center;
}
.stButton>button{
    width:100%;
    height:50px;
    background:#2563eb;
    color:white;
    font-size:18px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = joblib.load("model.pkl")

# ---------------- Sidebar ----------------
st.sidebar.title("🏦 Customer Churn")
st.sidebar.info(
"""
### Instructions
- Enter customer details
- Click Predict
- View prediction result
"""
)

st.title("🏦 Customer Churn Prediction")
st.markdown("---")

# ---------------- Input ----------------
col1, col2 = st.columns(2)

with col1:
    CreditScore = st.number_input("Credit Score",300,900,650)
    Geography = st.selectbox("Geography",["France","Germany","Spain"])
    Gender = st.selectbox("Gender",["Male","Female"])
    Age = st.number_input("Age",18,100,30)
    Tenure = st.number_input("Tenure",0,10,5)

with col2:
    Balance = st.number_input("Balance",0.0,300000.0,50000.0)
    NumOfProducts = st.selectbox("Number of Products",[1,2,3,4])
    HasCrCard = st.selectbox("Has Credit Card",[0,1])
    IsActiveMember = st.selectbox("Active Member",[0,1])
    EstimatedSalary = st.number_input("Estimated Salary",0.0,300000.0,50000.0)

st.markdown("")

# ---------------- Predict ----------------
if st.button("🔍 Predict Customer Churn"):

    data = pd.DataFrame({
        "CreditScore":[CreditScore],
        "Geography":[Geography],
        "Gender":[Gender],
        "Age":[Age],
        "Tenure":[Tenure],
        "Balance":[Balance],
        "NumOfProducts":[NumOfProducts],
        "HasCrCard":[HasCrCard],
        "IsActiveMember":[IsActiveMember],
        "EstimatedSalary":[EstimatedSalary]
    })

    prediction = model.predict(data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("❌ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)
        st.subheader("Prediction Probability")
        st.progress(int(prob[0][1]*100))
        st.write(f"**Churn Probability:** {prob[0][1]*100:.2f}%")

st.markdown("---")
st.caption("Developed by Abhishek Parihar | AIML Student")
