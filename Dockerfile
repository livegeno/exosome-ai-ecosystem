# Dockerfile for Exosome-AI Seven-Module Computational Ecosystem
# Academic Reproducibility Package v2.1.0

FROM python:3.11-slim

LABEL maintainer="Wei Lian <lianwubio@163.com>"
LABEL version="2.1.0"
LABEL description="Exosome-AI Computational Ecosystem - Academic Reproducibility Package"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy validation scripts
COPY validation_*.py ./
COPY utils.py ./
COPY statistical_tests.py ./
COPY run_all_validations.py ./
COPY generate_all_figures.py ./

# Create output directories
RUN mkdir -p results figures supplementary

# Set environment
ENV PYTHONUNBUFFERED=1
ENV SEED=42

# Default command: run all validations
CMD ["python", "run_all_validations.py", "--output", "results"]
