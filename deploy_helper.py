import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from dataset import Dataset
import streamlit as st

classes_and_models = {
    "model_1": {
        "model_name": "rnn_for_anomaly_v1"
    },
    "model_2": {
        "model_name": "rnn_for_anomaly_v2"
    },
    "model_3": {
        "model_name": "rnn_for_anomaly_v3"
    }
}


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    # convert the input to json file
    instances_list = instances.tolist()  # turn input into list (ML Engine wants JSON)
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list}

    # excute based on the json file
    response = service.projects().predict(
        name=name,
        body=input_data_json
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_arr(dataframe):
    """
    Normalize the input dataframe and return
    """
    train_path = 'train/normalpressure.csv'
    test_path = 'test/burstpressure.csv'
    DataLoader = Dataset(train_path, test_path, input_width=72)

    norm_df = (dataframe.iloc[:, 0:2] - DataLoader.mean) / DataLoader.std
    inputs, labels = norm_df.iloc[:72, 0:2], norm_df.iloc[72:, 0:2]

    # make sure the shape is correct
    assert len(inputs) == 72 and len(labels) == 216
    return inputs, labels

def update_logger(arr, nn_used, detector_used, pred_class, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns
    logger dictionary.
    """
    logger = {
        "arr": arr,
        "nn_used": nn_used,
        'detector_used': detector_used,
        "pred_class": pred_class,
        "correct": correct,
        "user_label": user_label
    }
    return logger