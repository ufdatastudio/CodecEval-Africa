#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name codec-eval
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=5:00:00
#SBATCH --mem=50GB
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200
#SBATCH --output=logs/codec_eval_%j.out
#SBATCH --error=logs/codec_eval_%j.err

set -euo pipefail

echo "===== GPU Info ====="
nvidia-smi || true

# CUDA setup
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH

# Paths
BASE_DIR="/orange/ufdatastudios/c.okocha/CodecEval-Africa"
cd "$BASE_DIR"

# Use /orange for model caches to avoid home quota
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load required modules
module load sox/14.4.2 ffmpeg/4.3.1

# Activate virtual environment
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "===== Environment Setup ====="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU 0: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU 1: $(python -c 'import torch; print(torch.cuda.get_device_name(1))')"
fi

echo "===== Starting CodecEval Pipeline ====="
echo "Current directory: $(pwd)"
echo "Virtual env: $VIRTUAL_ENV"

# Run the pipeline
make run

echo "===== Pipeline Completed Successfully ====="
