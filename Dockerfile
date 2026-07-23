FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all model artifacts & benchmarking script
COPY models/ ./models/
COPY benchmark_edge.py .

# Environment variable to force unbuffered stdout/stderr
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "benchmark_edge.py"]
