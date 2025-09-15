import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("Click-Through Rate Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(data.head())

    features = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Gender"]
    if all(f in data.columns for f in features):
        X = data[features]
        
        cat_cols = ["Gender"]
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ], remainder="passthrough")

        log_model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", pickle.load(open("best_ctr_model.pkl", "rb")))
        ])
        
        predictions = log_model.predict(X)
        data["Predicted Click"] = predictions
        st.write("Predictions:")
        st.dataframe(data[["Predicted Click"]])
    else:
        st.error("Uploaded file must include all required features.")
