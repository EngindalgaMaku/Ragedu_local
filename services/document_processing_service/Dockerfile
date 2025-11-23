FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from service directory
COPY services/document_processing_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire modular architecture
COPY services/document_processing_service/ .

# Copy the src directory for advanced semantic chunking
COPY src ./src

# Expose port 8080
EXPOSE 8080

# Run the NEW modular application
CMD ["python", "main_new.py"]