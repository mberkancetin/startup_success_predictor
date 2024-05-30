import pandas as pd
import numpy as np

from predictor.utils import simple_time_and_memory_tracker


@simple_time_and_memory_tracker
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    df = df.drop_duplicates()
    df = df[df.total_funding_usd > 0]
    
