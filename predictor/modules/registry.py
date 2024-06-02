import glob
import os
from tensorflow import keras
from predictor.params import *


def load_model():

    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    return latest_model


def save_model(model: keras.Model = None) -> None:

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "asd.h5")
    model.save(model_path)
