import os
import streamlit as st
import pandas as pd
import pickle
from ml_utility import (
    read_data, preprocess_data, train_model, evaluate_model, train_all_models, user_value_prediction
)
from xgboost import XGBClassifier

# Directory management
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

# Ensure directories exist
os.makedirs(f"{parent_dir}/trained_model", exist_ok=True)
os.makedirs(f"{parent_dir}/data", exist_ok=True)

# Streamlit app configuration
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")
st.title("Wine Quality Prediction Application")

# Load dataset
dataset_list = os.listdir(f"{parent_dir}/data")
dataset = st.selectbox("Select Dataset", dataset_list)

df = read_data(dataset)

if df is not None:
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    x_train, x_test, y_train, y_test, scaler = preprocess_data(df)

    # Save the scaler for future use
    scaler_path = f"{parent_dir}/trained_model/scaler.pkl"
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    # Train multiple models
    if st.button("Train All Models"):
        results = train_all_models(x_train, x_test, y_train, y_test)
        st.write("Model Performance:")
        st.write(results)

    # Check and train XGBoost model if not found
    model_path = f"{parent_dir}/trained_model/XGBoost.pkl"
    if not os.path.exists(model_path):
        st.warning("XGBoost model not found. Training a new model...")
        xgboost_model = XGBClassifier()
        xgboost_model.fit(x_train, y_train)
        with open(model_path, "wb") as file:
            pickle.dump(xgboost_model, file)
        st.success("XGBoost model trained and saved successfully!")

    # Select user inputs for prediction
    st.header("Wine Quality Prediction")
    input_features = {
        "Fixed Acidity": st.number_input("Fixed Acidity", min_value=3.0, max_value=16.0, step=0.1),
        "Volatile Acidity": st.number_input("Volatile Acidity", min_value=0.1, max_value=1.5, step=0.01),
        "Citric Acid": st.number_input("Citric Acid", min_value=0.0, max_value=1.0, step=0.01),
        "Residual Sugar": st.number_input("Residual Sugar", min_value=0.5, max_value=15.5, step=0.1),
        "Chlorides": st.number_input("Chlorides", min_value=0.01, max_value=0.6, step=0.01),
        "Free Sulfur Dioxide": st.number_input("Free Sulfur Dioxide", min_value=1, max_value=72, step=1),
        "Total Sulfur Dioxide": st.number_input("Total Sulfur Dioxide", min_value=6, max_value=289, step=1),
        "Density": st.number_input("Density", min_value=0.9900, max_value=1.0040, step=0.0001, format="%.4f"),
        "pH": st.number_input("pH", min_value=2.7, max_value=4.0, step=0.01),
        "Sulphates": st.number_input("Sulphates", min_value=0.3, max_value=2.0, step=0.01),
        "Alcohol": st.number_input("Alcohol", min_value=8.0, max_value=15.0, step=0.1)
    }

    # Predict using the saved model
    if st.button("Predict Quality"):
        try:
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            with open(scaler_path, "rb") as file:
                scaler = pickle.load(file)
            # Scale input features
            scaled_features = scaler.transform([list(input_features.values())])
            prediction = user_value_prediction(scaled_features, model)
            if prediction[0] == 1:
                st.success("Good Quality Wine 🍷")
            else:
                st.error("Bad Quality Wine 😞")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
