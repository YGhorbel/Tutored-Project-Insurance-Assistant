FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY workers /app/workers
COPY utils /app/utils
COPY config /app/config

# Create logs directory
RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/workers/ingestor.py"]