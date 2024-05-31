import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

from predictor.params import *
from predictor.modules.preprocess import preprocessor
from predictor.modules.registry import load_model
from predictor.interface.main import pred

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
        social_activity: bool,      # List of boolean values
    ) -> dict:
    """
        Makes a single prediction that returns a probability ratio between 0 and 1.
    """
    social_activity=[social_activity for _ in range(6)]

    X_pred = pd.DataFrame(dict(
        funding_status=["Early Stage Venture"],
        state=[str(location)],
        revenue_range=["$1M to $10M"],
        no_employees=[str(company_size)],
        no_founders=[1.0],
        industry_groups=[str(industry)],
        website=[int(bool(social_activity[0]))],
        phone=[int(bool(social_activity[1]))],
        email=[int(bool(social_activity[2]))],
        linkedin=[int(bool(social_activity[3]))],
        twitter=[int(bool(social_activity[4]))],
        facebook=[int(bool(social_activity[5]))],
        founded_year=[int(founded_year)],
        no_investors=[7.0],
        no_fund_rounds=[8.0],
        private_ipo=[1],
        company_type=[1],
        operting_status=[1],
        no_lead_investors=[4.0],
        no_sub_orgs=[0.0],
        has_preseed=[0],
        has_seed=[0],
        has_series_a=[1],
        has_series_b=[1],
        has_series_c=[0],
        has_series_d=[0],
        has_series_e=[0],
        has_angel=[0],
        has_debt_financing=[0],
        has_grant=[0],
        has_corporate_round=[0],
        has_series_x=[0],
        has_ico=[0]
    ))

    total_funding = [float(total_funding)]

    X_processed = preprocessor(X_pred, fit_tranform=False)

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
