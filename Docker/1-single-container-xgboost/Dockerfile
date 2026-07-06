FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY xgboost_docker.py .
COPY ai_student_impact_dataset.csv .

CMD ["python", "xgboost_docker.py"]