# docker/Dockerfile

# Multi-stage build for development and production
FROM python:3.10-slim as base

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8

# Copy source code
COPY src/ src/
RUN pip install -e .

# Expose ports
EXPOSE 8888 6006

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Install package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 lmpro && \
    chown -R lmpro:lmpro /workspace
USER lmpro

# Default command for production
CMD ["python", "scripts/train.py", "--help"]