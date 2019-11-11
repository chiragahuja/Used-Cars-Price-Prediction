FROM python:3
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY usedCarsPricePrediction.py ./
CMD ["python", "usedCarsPricePrediction.py", "/data/autos.csv"]
