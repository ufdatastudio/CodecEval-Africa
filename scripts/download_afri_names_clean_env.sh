#!/bin/bash
# Script to download Afri-Names dataset using a clean environment
# This avoids TorchCodec compatibility issues

set -e

echo "Creating clean environment for dataset download..."
echo "="*80

# Create a temporary clean environment
CLEAN_ENV_DIR=".venv_afri_names_clean"
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
echo "Now run the download script:"
echo "  source $CLEAN_ENV_DIR/bin/activate"
echo "  python scripts/download_afri_names_direct.py"
echo ""
echo "Or run it directly:"
echo "  $CLEAN_ENV_DIR/bin/python scripts/download_afri_names_direct.py"

