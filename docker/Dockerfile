FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; swap the index URL for a CUDA build if needed.
COPY requirements.txt setup.py pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -e ".[export,notebooks]"

# Copy the rest of the project (configs, scripts, notebooks). .dockerignore excludes .git, logs, checkpoints.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.torch

RUN mkdir -p logs checkpoints data/synthetic

CMD ["python", "scripts/train.py", "--help"]
