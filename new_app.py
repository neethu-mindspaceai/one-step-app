# import streamlit as st
import requests
import json

# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

from fastapi import FastAPI
from pydantic import BaseModel
# import mlflow.sklearn
import pickle

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# model = mlflow.sklearn.load_model("runs:/1f3a84d8cf09450997762c9d70c076d4/model")


@app.get("/")
async def root():
    return f"The model is : {model}"

@app.post("/predict/")
async def predict(input_data: InputData):
    model = pickle.load(open("model.pkl","rb"))
    # Extract input features from the request
    features = [input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4]
    # Make predictions using the loaded model
    prediction = model.predict([features])[0]
    # Create a response dictionary
    response_data = {"prediction": int(prediction)}
    # The prediction result needs to be converted from numpy.int32 to integer or string.

    return response_data


# # Define the Streamlit app title and description
# st.title("Machine Learning Model Deployment")
# st.write("Use this app to make predictions with the deployed model.")

# # Create input fields for user to enter data
# st.header("Input Data")
# feature1 = st.number_input("Feature 1", value=5.1, step=0.1)
# feature2 = st.number_input("Feature 2", value=3.5, step=0.1)
# feature3 = st.number_input("Feature 3", value=1.4, step=0.1)
# feature4 = st.number_input("Feature 4", value=0.2, step=0.1)

# # Create a button to trigger predictions
# if st.button("Predict"):
#     # Define the input data as a dictionary
#     input_data = {
#         "feature1": feature1,
#         "feature2": feature2,
#         "feature3": feature3,
#         "feature4": feature4
#     }

#     # Make a POST request to the FastAPI model
#     model_url = "http://127.0.0.1:8000/predict/"  # Replace with your FastAPI model URL
#     response = requests.post(model_url, json=input_data)

#     if response.status_code == 200:
#         prediction = json.loads(response.text)["prediction"]
#         st.success(f"Model Prediction: {prediction}")
#     else:
#         st.error("Failed to get a prediction. Please check your input data and try again.")