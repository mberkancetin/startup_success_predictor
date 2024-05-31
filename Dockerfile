# FROM python:3.8.12-slim

# WORKDIR /prod

# COPY startup_success_predictor startup_success_predictor
# COPY setup.py setup.py
# COPY requirements.txt requirements.txt

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# RUN pip install .

# COPY Makefile Makefile

# CMD uvicorn startup_success_predictor.api.fast:app --host 0.0.0.0 --port $PORT
