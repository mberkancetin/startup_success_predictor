import pandas as pd
import numpy as np

from predictor.modules.data import clean_data
from predictor.modules.registry import save_model
from predictor.modules.preprocess import preprocessor
from predictor.modules.model import initialize_model, compile_model, train_model, evaluate_model

import os

def preprocess_train(
        split_ratio: float = 0.02,
        learning_rate = 0.01,
        batch_size = 32,
        patience = 20
    ) -> float:
    data_path = os.path.dirname(__file__)
    file_path = os.path.join(data_path, "..", "..", "raw_data", "X_y_data3.csv")
    model_path = os.path.join(data_path, "..", "..", "models", "palantir_v4.keras")

    data = pd.read_csv(file_path)
    data_clean = clean_data(data)

    X = data_clean.drop("y", axis=1)
    y = data_clean.y

    X_processed = preprocessor(X, fit_tranform=True)
    X_shape = X_processed.shape[1]
    print(f"preprocessed, shape:{X_shape}")

    model = initialize_model((X_shape,))
    print("initialized")
    model.summary

    model = compile_model(model)
    print("compiled")

    model, history = train_model(model,
                                 X_processed,
                                 y,
                                 )
    print(history)
    model.save(filepath=model_path)

# preprocess_train()
def evaluate():
    evaluate_model()

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
        Receives a pandas DataFrame,
        preprocesses it, calls the existing model and predicts,
        returns to a numpy array.
    """
    X_df = pd.DataFrame(X_pred, columns=["months_since_founded",
                                "lat",
                                "lon",
                                "company_size",
                                "no_founders",
                                "industry_groups",
                                "funding_status",
                                "revenue_range",
                                "total_funding",
                                "has_debt_financing",
                                "has_grant",
                                "has_corporate_round"])
    return X_df

