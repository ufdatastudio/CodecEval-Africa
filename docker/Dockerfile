# Use CUDA base image for GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y sox ffmpeg git curl python3.11 python3.11-pip && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /workspace

# Copy requirements and install with uv (much faster than pip)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .
ENTRYPOINT ["bash"]
