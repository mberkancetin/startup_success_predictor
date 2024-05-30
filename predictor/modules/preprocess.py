import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
from predictor.utils import simple_time_and_memory_tracker


@simple_time_and_memory_tracker
def preprocessor(
    X: pd.DataFrame,
    y: pd.DataFrame,
):
    """
        Receives raw X and y DataFrames and preprocess
        fit and transform into X_processed DataFrame.
    """
    one_hot_category = [
        "state", "funding_status", "industry_groups", 'private_ipo',
    ]

    ordinal_category = [
        "no_employees", "revenue_range"
    ]

    numerical_features = [
        'founded_year', 'private_ipo', 'website', 'phone', "no_founders",
        'email', 'linkedin', 'twitter', 'facebook', 'no_investors', 'no_fund_rounds', 'operting_status',
        'no_sub_orgs', 'has_preseed', 'has_seed', 'has_series_a', 'has_series_b', 'has_series_c',
        'has_series_d', 'has_series_e', 'has_angel', 'has_debt_financing',
        'has_grant', 'has_corporate_round', 'has_series_x'
    ]

    no_employees_ordinal = [
        '11-50', '51-100', '101-250', '251-500', '501-1000', '1001-5000', '5001-10000', '10001+'
    ]

    revenue_range_ordinal = [
        'Less than $1M', '$1M to $10M', '$10M to $50M', '$50M to $100M', '$100M to $500M', '$500M to $1B', '$1B to $10B', '$10B+'
    ]

    feat_ordinal_dict = {
        "no_employees": no_employees_ordinal,
        "revenue_range": revenue_range_ordinal
    }

    encoder_ordinal = OrdinalEncoder(
        categories = [feat_ordinal_dict[i] for i in ordinal_category],
        dtype = np.int64
    )

    preproc_ordinal = make_pipeline(
        SimpleImputer(strategy = "most_frequent"),
        encoder_ordinal,
        MinMaxScaler()
    )

    preproc_min_numerical = make_pipeline(
        KNNImputer(),
        MinMaxScaler())

    preproc_nominal = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # preproc_robust_numerical = make_pipeline(
    #    KNNImputer(),
    #    RobustScaler())

    preproc = make_column_transformer(
            (preproc_ordinal, ordinal_category),
            (preproc_min_numerical, numerical_features),
            (preproc_nominal, one_hot_category),
            # (preproc_robust_numerical, robust_category),
            remainder="drop"
    )

    return pd.DataFrame(preproc.fit_transform(X, y))
