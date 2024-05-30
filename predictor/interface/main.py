import pandas as pd
import numpy as np




def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
        Receives a pandas DataFrame,
        preprocesses it, calls the existing model and predicts,
        returns to a numpy array.
    """
    pass

if __name__ == '__main__':
    try:
        # preprocess_and_train()
        # preprocess()
        # train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
