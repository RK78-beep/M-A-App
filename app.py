import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

st.title("M&A Deal Success Predictor")

uploaded_file = st.file_uploader("Upload your CSV file with deal data")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head())

    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        predictions = model.predict(data)
        st.write("Predictions:", predictions)
    except Exception as e:
        st.error(f"Model loading or prediction failed: {e}")