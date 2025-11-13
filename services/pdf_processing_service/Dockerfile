# Use Python 3.11 base image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and supporting modules
COPY main.py .
COPY model_cache_manager.py .
COPY cloud_storage_manager.py .

# Create models directory for marker cache with proper permissions
RUN mkdir -p /app/models /app/models/torch /app/models/huggingface /app/models/transformers /app/models/hf_home && \
    chmod -R 755 /app/models

# Set cache environment variables
ENV MARKER_CACHE_DIR=/app/models \
    TORCH_HOME=/app/models/torch \
    HUGGINGFACE_HUB_CACHE=/app/models/huggingface \
    TRANSFORMERS_CACHE=/app/models/transformers \
    HF_HOME=/app/models/hf_home \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0

EXPOSE 8080

CMD ["python", "main.py"]