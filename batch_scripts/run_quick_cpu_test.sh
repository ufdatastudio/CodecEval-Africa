#!/bin/bash
#SBATCH --job-name=quick_cpu_test
#SBATCH --output=quick_cpu_test_%j.out
#SBATCH --error=quick_cpu_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=hpg-turin
#SBATCH --mem=8G

echo "=== QUICK CPU-ONLY TEST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate

# Test all 6 codecs (CPU-only)
python -c "
import sys
sys.path.append('.')
from code.codecs.codec_registry import get_codec_runner
import os

# Test with a shorter audio file
test_file = 'data/afrispeech_dialog/data/247554f8-f233-4861-bc1a-8fc327b5d5df_2b500b633e5d5ecce35433cbbb859ddc_8bW4oSXn.wav'
output_dir = 'quick_cpu_test_output'

os.makedirs(output_dir, exist_ok=True)

# Test all 6 codecs
codecs_to_test = [
    'encodec_24khz',
    'unicodec', 
    'dac',
    'sematicodec',
    'apcodec',
    'languagecodec'
]

print('=== TESTING ALL 6 CODECS (CPU-ONLY) ===')
for codec_name in codecs_to_test:
    print(f'Testing {codec_name}...')
    try:
        if 'encodec' in codec_name:
            runner = get_codec_runner(codec_name, bandwidth_kbps=6.0, causal=True, sr=16000, device='cpu')
        else:
            runner = get_codec_runner(codec_name, bitrate_kbps=6.0, sr=16000, device='cpu')
        
        output_file = f'{output_dir}/{codec_name}_cpu.wav'
        runner.run(test_file, output_file)
        print(f'✅ {codec_name}: SUCCESS')
    except Exception as e:
        print(f'❌ {codec_name}: FAILED - {e}')

print('=== ALL 6 CODECS TEST COMPLETE ===')
"

echo "=== QUICK CPU TEST COMPLETE ==="
echo "Date: $(date)"
