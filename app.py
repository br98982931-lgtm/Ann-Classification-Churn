import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow  as tf

# load the trained model

model=tf.keras.models.load_model('model.h5')

#Load the encoder and scalers

with open('encoded_geography.pkl','rb') as file:
    encoded_geography=pickle.load(file)

with open('encoded_gender.pkl','rb') as file:
    encoded_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


#streamlit app

st.title("Customer Churn Prediction")

#User input

CreditScore=st.number_input('Credit score')
Geography=st.selectbox('Geography',encoded_geography.categories_[0])
Gender=st.selectbox('Gender',encoded_gender.classes_)
Age=st.slider('Age',18,92)
Tenure=st.slider('Tenure',1,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('Number of Product',1,4)
HasCrCard=st.selectbox('Has Credit Card',[0,1])
IsActiveMember=st.selectbox('Is Active Member',[0,1])
EstimatedSalary=st.number_input('estimated Salary')

input_data = {
    'CreditScore': [CreditScore],
    'Geography':[Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
}
input_df = pd.DataFrame(input_data)

geo_encoded = encoded_geography.transform([[Geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=encoded_geography.get_feature_names_out(['Geography'])
)

input_df['Gender'] = encoded_gender.transform(input_df['Gender'])

input_df = pd.concat(
    [input_df.drop('Geography', axis=1), geo_encoded_df],
    axis=1
)

input_scaled = scaler.transform(input_df)
prediction=model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability {prediction_proba}")
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')