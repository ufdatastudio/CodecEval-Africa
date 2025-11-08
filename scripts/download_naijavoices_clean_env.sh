#!/bin/bash
# Script to download NaijaVoices dataset using a clean environment
# This avoids TorchCodec compatibility issues
#
# IMPORTANT: This dataset is gated. You must:
# 1. Login to HuggingFace: huggingface-cli login
# 2. Accept the dataset terms at: https://huggingface.co/datasets/naijavoices/naijavoices-dataset

set -e

echo "Creating clean environment for NaijaVoices dataset download..."
echo "================================================================================"

# Create a temporary clean environment
CLEAN_ENV_DIR=".venv_naijavoices_clean"
BASE_DIR="/orange/ufdatastudios/c.okocha/CodecEval-Africa"

cd "$BASE_DIR"

# Create virtual environment
echo "Creating virtual environment: $CLEAN_ENV_DIR"
python -m venv "$CLEAN_ENV_DIR"

# Activate and install minimal dependencies
source "$CLEAN_ENV_DIR/bin/activate"
echo "Installing dependencies..."
pip install --upgrade pip
pip install datasets[audio] huggingface-hub pandas pyarrow tqdm soundfile

echo "✓ Clean environment ready"
echo ""
echo "IMPORTANT: Before running the download script:"
echo "  1. Login to HuggingFace: huggingface-cli login"
echo "  2. Accept dataset terms: https://huggingface.co/datasets/naijavoices/naijavoices-dataset"
echo ""
echo "Now run the download script:"
echo "  source $CLEAN_ENV_DIR/bin/activate"
echo "  python scripts/download_naijavoices_direct.py"
echo ""
echo "Or run it directly:"
echo "  $CLEAN_ENV_DIR/bin/python scripts/download_naijavoices_direct.py"

