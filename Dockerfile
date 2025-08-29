# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including git for GitHub dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --force-reinstall git+https://github.com/sadmanca/nano-graphrag.git@nano-to-genkg#egg=nano-graphrag
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run will set PORT env variable)
EXPOSE 8080

# Use gunicorn for production WSGI server
# Updated to fix genkg import issue
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 web_app:app