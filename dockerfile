FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/diskcache /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=False
ENV PORT=5000
ENV CACHE_TTL=3600
ENV PRELOAD_TOP_CITIES=12
ENV MAX_IMAGE_WORKERS=8

# Expose port
EXPOSE $PORT

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]