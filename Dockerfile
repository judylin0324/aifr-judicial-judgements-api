FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/Data && \
    curl -L -o "/app/Data/108_地方法院_刑事訴訟.csv" "https://github.com/judylin0324/aifr-judicial-judgements-api/raw/main/Data/108_地方法院_刑事訴訟.csv" && \
    curl -L -o "/app/Data/108_地方法院_民事訴訟.csv" "https://github.com/judylin0324/aifr-judicial-judgements-api/raw/main/Data/108_地方法院_民事訴訟.csv" && \
    curl -L -o "/app/Data/108_地方法院_民事非訟.csv" "https://github.com/judylin0324/aifr-judicial-judgements-api/raw/main/Data/108_地方法院_民事非訟.csv" && \
    curl -L -o "/app/Data/108_地方法院_家事訴訟.csv" "https://github.com/judylin0324/aifr-judicial-judgements-api/raw/main/Data/108_地方法院_家事訴訟.csv"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]