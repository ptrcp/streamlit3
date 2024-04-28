import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
model = joblib.load('modelUTS.pkl')

def main():
    st.title('Machine Learning Model Deployment')

    # Add user input components for 5 features
    #credit_score = st.slider('CreditScore', min_value=0.0, max_value=10.0, value=0.1)
    credit_score = st.number_input(label, min_value=0, max_value=None, value="min",
                                   step=None, format=None, key=None, help=None, on_change=None, args=None,
                                   kwargs=None, *, placeholder=None,disabled=False,
                                   label_visibility="visible")
    st.write('The credit score number is ', credit_score)
    
    #geography = st.slider('Geography', min_value=0.0, max_value=10.0, value=0.1)
    geography = st.selectbox("Location", ("0:France", "1:Spain", "2:Germany"),
                             index=None, placeholder="Select geography...",)
    st.write('You selected:', geography)
    
    gender = st.slider('Gender', min_value=0.0, max_value=10.0, value=0.1)
    age = st.slider('Age', min_value=0.0, max_value=10.0, value=0.1)
    
    tenure = st.slider('Tenure', min_value=0.0, max_value=10.0, value=0.1)
    balance = st.slider('Balance', min_value=0.0, max_value=10.0, value=0.1)
    product = st.slider('NumOfProducts', min_value=0.0, max_value=10.0, value=0.1)
    crcard = st.slider('HasCrCard', min_value=0.0, max_value=10.0, value=0.1)
    active = st.slider('IsActiveMember', min_value=0.0, max_value=10.0, value=0.1)
    salary = st.slider('EstimatedSalary', min_value=0.0, max_value=10.0, value=0.1)

    
    if st.button('Make Prediction'):
        features = [credit_score,geography,gender,age,tenure,
                   balance,product,crcard,active,salary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()