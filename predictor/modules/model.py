import pandas as pd
import numpy as np

from predictor.utils import simple_time_and_memory_tracker

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

@simple_time_and_memory_tracker
def initialize_model(input_shape: tuple) -> Model:
    """
        Receives a dictionary of parameters as input,
        returns to a ML model.
    """
    reg = regularizers.l1_l2(l2=0.005)

    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(50, activation="relu", kernel_regularizer=reg))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(20, activation="tanh"))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
        Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    return model


@simple_time_and_memory_tracker
def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[Model, dict]:
    """
        Fit the model and return a tuple (fitted_model, history)
    """
    es = EarlyStopping(
    patience=20,
    restore_best_weights=True,
    verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=700,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32
    ) -> tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """
    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    return metrics
