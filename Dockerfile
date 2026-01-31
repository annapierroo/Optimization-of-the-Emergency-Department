FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    graphviz \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["python", "src/ingest_data.py"]

CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
