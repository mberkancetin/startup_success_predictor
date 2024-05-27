import pandas as pd
import numpy as np

from predictor.utils import simple_time_and_memory_tracker


@simple_time_and_memory_tracker
def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
        Receives a pandas DataFrame,
        preprocesses features, creates preprocessing pipelines,
        fit_transform the raw data into preprocessed data,
        returns to a numpy array.
    """
