import pandas as pd
import numpy as np

from predictor.utils import simple_time_and_memory_tracker



@simple_time_and_memory_tracker
def initialize_model(params: dict) -> Model:
    """
        Receives a dictionary of parameters as input,
        returns to a ML model.
    """
    pass



@simple_time_and_memory_tracker
def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
