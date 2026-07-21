import streamlit as st
import pandas as pd
import joblib
from PIL import Image
st.set_page_config(page_title='Customer Churn Prediction',page_icon='🏦',layout='wide')
model=joblib.load('model.pkl')
st.title('🏦 Customer Churn Prediction')
c1,c2=st.columns(2)
with c1:
    CreditScore=st.number_input('Credit Score',300,900,650)
    Geography=st.selectbox('Geography',['France','Germany','Spain'])
    Gender=st.selectbox('Gender',['Male','Female'])
    Age=st.number_input('Age',18,100,35)
    Tenure=st.slider('Tenure',0,10,5)
with c2:
    Balance=st.number_input('Balance',0.0,300000.0,50000.0)
    NumOfProducts=st.selectbox('Products',[1,2,3,4])
    HasCrCard=st.selectbox('Credit Card',[0,1])
    IsActiveMember=st.selectbox('Active Member',[0,1])
    EstimatedSalary=st.number_input('Estimated Salary',0.0,300000.0,50000.0)
if st.button('Predict'):
    df=pd.DataFrame({'CreditScore':[CreditScore],'Geography':[Geography],'Gender':[Gender],'Age':[Age],'Tenure':[Tenure],'Balance':[Balance],'NumOfProducts':[NumOfProducts],'HasCrCard':[HasCrCard],'IsActiveMember':[IsActiveMember],'EstimatedSalary':[EstimatedSalary]})
    pred=model.predict(df)[0]
    if hasattr(model,'predict_proba'):
        p=model.predict_proba(df)[0][1]
        st.progress(int(p*100))
        st.write(f'Churn Probability: {p*100:.2f}%')
    st.success('Customer Will Not Churn' if pred==0 else 'Customer Will Churn')
  
st.markdown("""
<div style="
background-color:#0F172A;
padding:15px;
border-radius:10px;
text-align:center;
color:white;
">
<h4>👨‍💻 Developed by Abhishek Parihar</h4>
<p>B.Tech | Artificial Intelligence & Machine Learning</p>
<p>Customer Churn Prediction System using Machine Learning</p>
</div>
""", unsafe_allow_html=True)
st.markdown("## 📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 Customers",
        value="10,000",
        delta="+250"
    )

with col2:
    st.metric(
        label="📉 Churn Rate",
        value="20.4%",
        delta="-1.2%"
    )

with col3:
    st.metric(
        label="💳 Active Members",
        value="5,151",
        delta="+98"
    )

with col4:
    st.metric(
        label="💰 Avg Balance",
        value="$76,485",
        delta="+3.4%"
    )

st.markdown("---")
if prediction[0] == 1:
    st.metric(
        "Risk Level",
        "🔴 HIGH"
    )
else:
    st.metric(
        "Risk Level",
        "🟢 LOW"
    )
