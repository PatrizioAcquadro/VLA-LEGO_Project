# ============================================================================
# Dockerfile for WorldSim Training
# Multi-stage build for smaller final image
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base CUDA image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Indiana/Indianapolis

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Python environment
# ---------------------------------------------------------------------------
FROM base AS python

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Stage 3: Install dependencies
# ---------------------------------------------------------------------------
FROM python AS dependencies

WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml ./
COPY README.md ./

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    wandb>=0.15.0 \
    tqdm>=4.65.0 \
    einops>=0.6.0 \
    safetensors>=0.3.0 \
    accelerate>=0.20.0 \
    deepspeed>=0.10.0

# Dev dependencies (for testing inside container)
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0

# ---------------------------------------------------------------------------
# Stage 4: Final image
# ---------------------------------------------------------------------------
FROM dependencies AS final

WORKDIR /app

# Build arguments for versioning
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown

# Labels
LABEL org.opencontainers.image.title="WorldSim Training"
LABEL org.opencontainers.image.description="Training environment for World Simulation Model"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

# Copy project code
COPY . .

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Store build info
RUN echo "GIT_COMMIT=${GIT_COMMIT}" > /app/.build_info \
    && echo "BUILD_DATE=${BUILD_DATE}" >> /app/.build_info

# Default command
CMD ["python", "-m", "train.trainer", "--help"]

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
