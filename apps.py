import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from deploy_helper import load_and_prep_arr, classes_and_models, update_logger, predict_json
from utils import cusum, ewma
import numpy as np
import pandas as pd
# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rnn-anomaly-6b217fba9aa1.json" # change for your GCP key
PROJECT = "rnn-anomaly" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to water burst detection")
st.header("Identify if a burst occurs!")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(dataframe, NEURAL_NETWORK, DETECTOR) -> bool:
    """
    Takes DataFrame and uses TF model to make a prediction.
    Returns:
     is_burst: bool
    """

    inputs, labels = load_and_prep_arr(dataframe)
    inputs = np.expand_dims(inputs, axis = 0)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=NEURAL_NETWORK,
                         instances=inputs)

    # preds -> list, 1 * 216 * 2
    # labels -> list, 216 * 2
    if DETECTOR == 'CUSUM':
        is_burst = cusum(preds[0], labels.to_numpy().tolist())
    elif DETECTOR == 'EWMA':
        is_burst = ewma(preds[0], labels.to_numpy().tolist())
    return is_burst

# Pick the model version
choose_head = st.sidebar.selectbox(
    "Pick neural network to predict the normal values",
    ("Model V1","Model V2", "Model V3"),
)

choose_detector = st.sidebar.selectbox(
    "Pick the detector to analyze the residual",
    ("CUSUM", "EWMA"),
)
# Model choice logic
CLASSES = ['Burst', 'Normal']
if choose_head == "Model V1":
    NN = classes_and_models["model_1"]["model_name"]
elif choose_head == "Model V2":
    NN = classes_and_models["model_2"]["model_name"]
elif choose_head == "Model V3":
    NN = classes_and_models["model_3"]["model_name"]

if choose_detector == "CUSUM":
    DETECTOR = 'CUSUM'
elif choose_detector == "EWMA":
    DETECTOR = 'EWMA'

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an array of sensor readings",
                                 type=["csv"])

session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an csv file.")
    st.stop()
else:
    # reset the buffer
    uploaded_file.seek(0)
    # read csv file using pandas
    session_state.uploaded_file = pd.read_csv(uploaded_file)
    st.write('Below is the raw data:', session_state.uploaded_file)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True

# If click predict
if session_state.pred_button:
    # make prediction based on the uploaded csv file
    session_state.is_burst = make_prediction(session_state.uploaded_file, NEURAL_NETWORK=NN, DETECTOR = DETECTOR)
    if session_state.is_burst:
        st.write("Be carefule, there is a burst!!!")
    else:
        st.write('Congratulations, no burst occurs')

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(arr=session_state.uploaded_file,
                            nn_used=NN,
                            detector_used = DETECTOR,
                            pred_class=session_state.is_burst,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(arr=session_state.arr,
                                nn_used=NN,
                                detector_used=DETECTOR,
                                pred_class=session_state.is_burst,
                                correct=False,
                                user_label=session_state.correct_class))