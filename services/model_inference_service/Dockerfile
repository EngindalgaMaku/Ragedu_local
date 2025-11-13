# Use Python slim image as base image for CPU compatibility
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including curl for Ollama installation
RUN apt-get update && apt-get install -y curl procps && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the content of the local src directory to the working directory
COPY main.py .

# Download models during build time to avoid runtime delays
RUN ollama serve & \
    sleep 10 && \
    ollama pull llama3:8b && \
    ollama pull mistral:7b && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Specify the command to run on container startup
# Start Ollama server in background and then start FastAPI
CMD /bin/sh -c "ollama serve & sleep 5 && python main.py"