FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Clean_Dataset.csv .
COPY *.py .

RUN mkdir -p output
