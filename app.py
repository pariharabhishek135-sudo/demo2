import streamlit as st
import pandas as pd
import joblib
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
