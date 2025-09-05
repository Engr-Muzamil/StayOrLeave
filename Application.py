import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load Model & Encoders
model = load_model('model.h5')  # Trained Model

with open('One_hot.pkl', 'rb') as file:  
    One_hot = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Churn Prediction')

# User Input
geography = st.selectbox('Geography', One_hot.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
estimated_salary = st.number_input('Estimated Salary')
num_of_products = st.slider('Num of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = One_hot.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=One_hot.get_feature_names_out(['Geography']))

# Merge all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display result
st.write(f"Churn Probability: {prediction_prob:.2f}")
if prediction_prob > 0.5:
    st.write('⚠️ The customer is likely to leave the bank.')
else:
    st.write('✅ The customer is likely to stay with the bank.')
