FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements
COPY requirements.txt .

# Install packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY api /app/api
COPY utils /app/utils
COPY config /app/config

# Create directories
RUN mkdir -p /app/logs /tmp

ENV PYTHONUNBUFFERED=1

EXPOSE 8000 7860

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]