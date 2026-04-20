FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/judylin0324/aifr-judicial-judgements-api.git /tmp/repo && \
    cd /tmp/repo && git lfs pull && \
    mkdir -p /app/Data && cp /tmp/repo/Data/*.csv /app/Data/ && \
    rm -rf /tmp/repo

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]