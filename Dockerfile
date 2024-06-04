FROM python:3.8.12-slim

WORKDIR /prod

COPY models models
COPY predictor predictor
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY raw_data/X_y_data3.csv raw_data/X_y_data3.csv

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

COPY Makefile Makefile

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
