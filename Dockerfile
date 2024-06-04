FROM python:3.8.12-slim
FROM tensorflow/tensorflow:2.10.0

WORKDIR /prod

COPY predictor predictor
COPY setup.py setup.py
COPY requirements.txt requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

COPY Makefile Makefile

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
