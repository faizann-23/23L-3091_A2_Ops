# Task 6.2: Dockerfile for VisionGuard Web App
# Base image: Python 3.11 slim variant for lightweight, secure deployment
FROM python:3.11-slim

# Avoid writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=0
ENV PORT=5000

WORKDIR /app

# Install system dependencies (curl needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer Cache Optimization:
# Copy requirements first so Docker reuses cached pip layers when code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY app.py inference.py database.py champion_mobilenetv2_best.keras ./
COPY templates/ ./templates/

# Security: Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Document exposed port
EXPOSE 5000

# Task 6.2: Health Check instruction
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start Flask application with Gunicorn or direct python
CMD ["python", "app.py"]
