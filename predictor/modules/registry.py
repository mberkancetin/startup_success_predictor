import glob
import os
from tensorflow import keras
from predictor.params import *


def load_model():

    latest_model = keras.models.load_model("predictor/models/palantir_v4.keras")
    second_model = keras.models.load_model("predictor/models/second_model.keras")
    return latest_model, second_model


def save_model(model: keras.Model = None) -> None:

    model_path = "predictor/models/palantir_v5.keras"
    model.save(model_path)
