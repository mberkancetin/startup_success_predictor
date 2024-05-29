import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

from startup_success_predictor.params import *
from startup_success_predictor.predictor.preprocessor import preprocess_features
from startup_success_predictor.predictor.registry import load_model
from startup_success_predictor.interface.main import pred

app = FastAPI()

app.state.model = load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/predict")
def predict(
        founded_year: int,          # Year between 2009-2024
        location: str,              # Region in Germany
        company_size: str,          # Company size in range
        industry: str,              # Industry of the company
        total_funding: float,       # Total funding amount in USD
        social_activity: list,      # List of boolean values
    ) -> dict:
    """
        Makes a single prediction that returns a probability ratio between 0 and 1.
    """

    X_pred = pd.DataFrame(dict(
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
        social_activity_fb=[int(bool(social_activity[5]))],
        company_type="For Profit"                               # The rest will be hardcoded!
    ))

    X_processed = preprocess_features(X_pred)

    y_pred = app.state.model.predict(X_processed)

    response = {
        "Success Probability": float(y_pred)
    }

    return response


@app.get("/")
def root():
    response = {
        "Connection": "Success"
    }
    return response
