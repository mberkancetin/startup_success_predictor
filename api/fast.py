import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

from predictor.params import *
from predictor.modules.preprocess import preprocessor
# from predictor.modules.registry import load_model
# from predictor.interface.main import pred

from tensorflow import keras

app = FastAPI()

# app.state.model = load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/predict")
def predict(
        months_since_founded: int,          # Year between 2009-2024
        lat: float,              # Region in Germany
        lon: float,
        company_size: str,          # Company size in range
        industry_groups: str,              # Industry of the company
        funding_status: str,
        revenue_range: str,
        total_funding: float,       # Total funding amount in USD
    ) -> dict:
    """
        Makes a single prediction that returns a probability ratio between 0 and 1.
    """
    has_preseed=[0]
    has_seed=[0]
    has_series_a=[0]
    has_series_b=[0]
    has_series_c=[0]
    has_debt_financing=[0]
    has_grant=[0]
    has_corporate_round=[0]
    has_series_x=[0]

    if funding_status == "Preseed":
        has_preseed = [1]
    elif funding_status == "Seed":
        has_seed=[1]
        has_preseed = [np.random.randint(2)]
    elif funding_status == "Series A":
        has_series_a=[1]
        has_seed=[np.random.randint(2)]
        has_preseed = [np.random.randint(2)]
    elif funding_status == "Series B":
        has_series_b=[1]
        has_series_a=[np.random.randint(2)]
        has_seed=[np.random.randint(2)]
        has_preseed = [np.random.randint(2)]
    elif funding_status == "Series A":
        has_series_c=[1]
        has_series_b=[np.random.randint(2)]
        has_series_a=[np.random.randint(2)]
        has_seed=[np.random.randint(2)]
        has_preseed = [np.random.randint(2)]


    X_pred = pd.DataFrame(dict(
        funding_status=[str(funding_status)],
        lat=[float(lat)],
        lon=[float(lon)],
        revenue_range=[str(revenue_range)],
        no_employees=[str(company_size)],
        no_founders=[0.0],
        industry_groups=[str(industry_groups)],
        months_since_founded=[int(months_since_founded)],
        no_investors=[0.0],
        no_fund_rounds=[0.0],
        private_ipo=[0],
        company_type=[0],
        operting_status=[0],
        no_lead_investors=[0.0],
        no_sub_orgs=[0.0],
        has_preseed=has_preseed,
        has_seed=has_seed,
        has_series_a=has_series_a,
        has_series_b=has_series_b,
        has_series_c=has_series_c,
        has_debt_financing=[0],
        has_grant=[0],
        has_corporate_round=[0],
    ))

    total_funding = [float(total_funding)]

    X_processed = preprocessor(X_pred, fit_tranform=False)

    # y_pred = app.state.model.predict(X_processed)
    latest_model = keras.models.load_model("models/palantir_v4.keras")
    y_pred = latest_model.predict(X_processed)

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


prediction = predict(2020,          # Year between 2009-2024
        52.7565,              # Region in Germany
        10.43442,
        "101-250",          # Company size in range
        "Other",              # Industry of the company
        "Seed",
        "Less than $1M",
        10000000000,       # Total funding amount in USD
        )

# print(prediction)
