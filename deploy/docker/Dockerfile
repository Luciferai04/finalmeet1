# Production Dockerfile for Real-Time Video Translator
# Base stage with all dependencies
FROM python:3.9-slim as base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libcanberra-gtk3-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    doxygen \
    redis-server \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements_production.txt .
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements_production.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional ML/AI libraries
RUN pip3 install transformers accelerate optimum[onnxruntime] sentence-transformers

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/temp /app/uploads

# Set up permissions
RUN chmod +x scripts/*.sh || true
RUN chown -R 1000:1000 /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port
EXPOSE 7860

# Switch to non-root user
USER appuser

# Production stage - final optimized image
FROM base as production

# Copy only necessary files for production
COPY --from=base /app /app
COPY --from=base /etc/supervisor/conf.d/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY --from=base /entrypoint.sh /entrypoint.sh

# Ensure proper permissions
USER root
RUN chmod +x /entrypoint.sh && chown -R appuser:appuser /app
USER appuser

# Set working directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python3", "app_production.py"]
