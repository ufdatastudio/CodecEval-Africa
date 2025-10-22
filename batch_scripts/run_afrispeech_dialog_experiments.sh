#!/bin/bash
#SBATCH --job-name=afrispeech_dialog_eval
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "AFRISPEECH-DIALOG CODEC EVALUATION"
echo "=========================================="

# Load required modules
module load sox/14.4.2 ffmpeg/4.3.1

# Set up environment
BASE_DIR="/orange/ufdatastudios/c.okocha/CodecEval-Africa"
cd "$BASE_DIR"

# Activate virtual environment
source .venv/bin/activate

# Set cache directories for faster processing
export HF_HOME="/orange/ufdatastudios/c.okocha/.cache/huggingface"
export TRANSFORMERS_CACHE="/orange/ufdatastudios/c.okocha/.cache/transformers"
export TORCH_HOME="/orange/ufdatastudios/c.okocha/.cache/torch"

# Performance settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

echo "Environment setup complete"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "CUDA: $(nvcc --version | grep release)"

# Step 1: Create manifests
echo ""
echo "Step 1: Creating AfriSpeech-Dialog manifests..."
python scripts/create_afrispeech_dialog_manifest.py

if [ $? -ne 0 ]; then
    echo "ERROR: Manifest creation failed"
    exit 1
fi

echo "✓ Manifests created successfully"

# Step 2: Skip validation test (manifests already created)
echo ""
echo "Step 2: Skipping validation test - proceeding to full evaluation"

# Step 3: Full AfriSpeech-Dialog evaluation
echo ""
echo "Step 3: Running full AfriSpeech-Dialog evaluation..."
echo "This will evaluate 6 codecs × 5 bitrates × ~25 samples = ~750 evaluations"

# Run the encode/decode stage
python -m code.pipeline --config configs/afrispeech_dialog_only.yml --stage encode_decode

if [ $? -ne 0 ]; then
    echo "ERROR: AfriSpeech-Dialog encoding/decoding failed"
    exit 1
fi

echo "✓ AfriSpeech-Dialog encode/decode completed"

# Step 4: Run metrics on AfriSpeech-Dialog
echo ""
echo "Step 4: Computing metrics for AfriSpeech-Dialog..."
python -m code.pipeline --config configs/afrispeech_dialog_only.yml --stage metrics

if [ $? -ne 0 ]; then
    echo "ERROR: AfriSpeech-Dialog metrics computation failed"
    exit 1
fi

echo "✓ AfriSpeech-Dialog metrics computed"

# Step 5: Generate summary report
echo ""
echo "Step 5: Generating summary report..."
python scripts/analyze_results.py --results_dir results/afrispeech_dialog --output_dir results/afrispeech_dialog/reports

echo ""
echo "=========================================="
echo "AFRISPEECH-DIALOG EXPERIMENTS COMPLETED"
echo "=========================================="
echo ""
echo "Results available in:"
echo "- results/afrispeech_dialog/"
echo "- results/afrispeech_dialog/reports/"
echo ""
echo "Check results/afrispeech_dialog/reports/ for analysis and visualizations"

echo "Job completed successfully!"
