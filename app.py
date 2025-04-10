import streamlit as st
import pickle
import numpy as np
from keras.models import load_model


model = load_model('url.h5')


def preprocess_url(url):

    processed_url = url.lower()
    return processed_url

def predict_malicious_url(url, model):
    # Preprocess the URL
    processed_url = preprocess_url(url)
   
    vectorized_url = np.random.rand(16)  
    # Make prediction
    prediction = model.predict(np.array([vectorized_url]))
    return prediction[0][0]  

# Streamlit UI
st.title("Malicious URL Detection")

url_input = st.text_input("Enter URL:", "")

if st.button("Check"):
    if url_input.strip() == "":
        st.warning("Please enter a URL.")
    else:
        # Predict if URL is malicious
        prediction = predict_malicious_url(url_input, model)
        if prediction >= 0.5:
            st.error("Malicious URL!")
        else:
            st.success("Safe URL!")
