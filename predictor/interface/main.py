import pandas as pd
import numpy as np




def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
        Receives a pandas DataFrame,
        preprocesses it, calls the existing model and predicts,
        returns to a numpy array.
    """
    pass

def user_input(X: dict) -> pd.DataFrame:
    """
        Takes user input as dictionary and tranforms it into a pd.Dataframe.
        X = pd.DataFrame(dict(
            founded_year=[int(founded_year)],
            location=[str(location)],
            company_size=[str(company_size)],
            industry=[str(industry)],
            total_funding=[float(total_funding)],
            social_activity_wb=[int(bool(social_activity[0]))],
            social_activity_ph=[int(bool(social_activity[1]))],
            social_activity_em=[int(bool(social_activity[2]))],
            social_activity_ln=[int(bool(social_activity[3]))],
            social_activity_tw=[int(bool(social_activity[4]))],
            social_activity_fb=[int(bool(social_activity[5]))]
        ))
    """
    X_df = pd.DataFrame(X, columns=["Founding Year",
                                    "Location",
                                    "Company Size",
                                    "Industry Group",
                                    "Total Funding (USD)",
                                    "Website",
                                    "Phone",
                                    "Email",
                                    "Linkedin",
                                    "Twitter",
                                    "Facebook"])
    return X_df
