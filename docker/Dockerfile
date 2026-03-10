FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

# Create necessary directories
RUN mkdir -p logs models data/synthetic

# Default command (can be overridden)
CMD ["python", "--version"]
