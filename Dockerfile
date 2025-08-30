# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    wget \
    curl \
    git \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pkg-config \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p input_files extracted_images vectorstore_faiss temp_docling

# Add a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
ENTRYPOINT ["python"]
CMD ["main.py", "--reprocess"]
