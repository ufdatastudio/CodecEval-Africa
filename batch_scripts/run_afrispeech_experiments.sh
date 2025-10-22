#!/bin/bash
#SBATCH --job-name=afrispeech_codec_eval
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "AFRISPEECH CODEC EVALUATION EXPERIMENTS"
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
echo "Step 1: Creating AfriSpeech manifests..."
python scripts/create_afrispeech_manifests.py

if [ $? -ne 0 ]; then
    echo "ERROR: Manifest creation failed"
    exit 1
fi

echo "✓ Manifests created successfully"

# Step 2: Quick validation test
echo ""
echo "Step 2: Quick validation test with tiny manifests..."
python -m code.pipeline --config configs/afrispeech_benchmark.yml --stage encode_decode --datasets data/manifests/afrispeech_dialog_tiny.yaml data/manifests/afrispeech_200_tiny.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: Validation test failed"
    exit 1
fi

echo "✓ Validation test passed"

# Step 3: Full AfriSpeech-Dialog evaluation
echo ""
echo "Step 3: Running full AfriSpeech-Dialog evaluation..."
echo "This will evaluate 6 codecs × 5 bitrates × ~50 samples = ~1,500 evaluations"

# Create a temporary config for AfriSpeech-Dialog only
cat > configs/afrispeech_dialog_only.yml << EOF
bitrate_kbps: [3, 6, 12, 18, 24]
codecs:
  - name: encodec_24khz
    runner: code.codecs.encodec_runner:EncodecRunner
    args: { bandwidth_kbps: "\${bitrate_kbps}", causal: true, sr: 24000 }
  - name: soundstream_impl
    runner: code.codecs.soundstream_runner:SoundStreamRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: unicodec
    runner: code.codecs.unicodec_runner:UniCodecRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: dac
    runner: code.codecs.dac_runner:DACRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: sematicodec
    runner: code.codecs.sematicodec_runner:SemantiCodecRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: apcodec
    runner: code.codecs.apcodec_runner:APCodecRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
datasets:
  - data/manifests/afrispeech_dialog_balanced.yaml
metrics:
  - visqol
  - nisqa
  - dnsmos
  - asr_wer
  - speaker_cosine
  - prosody_f0_rmse
outputs:
  dir: results/afrispeech_dialog
seed: 2025
EOF

# Run the evaluation
python -m code.pipeline --config configs/afrispeech_dialog_only.yml --stage encode_decode

if [ $? -ne 0 ]; then
    echo "ERROR: AfriSpeech-Dialog evaluation failed"
    exit 1
fi

echo "✓ AfriSpeech-Dialog evaluation completed"

# Step 4: Run metrics on AfriSpeech-Dialog
echo ""
echo "Step 4: Computing metrics for AfriSpeech-Dialog..."
python -m code.pipeline --config configs/afrispeech_dialog_only.yml --stage metrics

if [ $? -ne 0 ]; then
    echo "ERROR: AfriSpeech-Dialog metrics computation failed"
    exit 1
fi

echo "✓ AfriSpeech-Dialog metrics computed"

# Step 5: AfriSpeech-200 evaluation (if data is available)
echo ""
echo "Step 5: Checking AfriSpeech-200 data availability..."
if [ -d "data/afrispeech_full" ]; then
    echo "AfriSpeech-200 data found. Running evaluation..."
    
    # Create config for AfriSpeech-200 only
    cat > configs/afrispeech_200_only.yml << EOF
bitrate_kbps: [3, 6, 12, 18, 24]
codecs:
  - name: encodec_24khz
    runner: code.codecs.encodec_runner:EncodecRunner
    args: { bandwidth_kbps: "\${bitrate_kbps}", causal: true, sr: 24000 }
  - name: soundstream_impl
    runner: code.codecs.soundstream_runner:SoundStreamRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: unicodec
    runner: code.codecs.unicodec_runner:UniCodecRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: dac
    runner: code.codecs.dac_runner:DACRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: sematicodec
    runner: code.codecs.sematicodec_runner:SemantiCodecRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
  - name: apcodec
    runner: code.codecs.apcodec_runner:APCodecRunner
    args: { bitrate_kbps: "\${bitrate_kbps}", sr: 24000 }
datasets:
  - data/manifests/afrispeech_200_balanced.yaml
metrics:
  - visqol
  - nisqa
  - dnsmos
  - asr_wer
  - speaker_cosine
  - prosody_f0_rmse
outputs:
  dir: results/afrispeech_200
seed: 2025
EOF
    
    python -m code.pipeline --config configs/afrispeech_200_only.yml --stage encode_decode
    python -m code.pipeline --config configs/afrispeech_200_only.yml --stage metrics
    
    echo "✓ AfriSpeech-200 evaluation completed"
else
    echo "AfriSpeech-200 data not found. Skipping this evaluation."
    echo "To run AfriSpeech-200 evaluation, first run: python afrispeech.py"
fi

# Step 6: Generate summary report
echo ""
echo "Step 6: Generating summary report..."
python scripts/analyze_results.py --results_dir results --output_dir results/reports

echo ""
echo "=========================================="
echo "AFRISPEECH EXPERIMENTS COMPLETED"
echo "=========================================="
echo ""
echo "Results available in:"
echo "- results/afrispeech_dialog/"
echo "- results/afrispeech_200/"
echo "- results/reports/"
echo ""
echo "Check results/reports/ for analysis and visualizations"

# Clean up temporary config files
rm -f configs/afrispeech_dialog_only.yml configs/afrispeech_200_only.yml

echo "Job completed successfully!"
