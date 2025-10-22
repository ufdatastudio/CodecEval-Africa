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

# Test CPU-only approach with shorter audio
python -c "
import sys
sys.path.append('.')
from code.codecs.encodec_runner import EncodecRunner
from code.codecs.soundstream_runner import SoundStreamRunner
import os

# Test with a shorter audio file
test_file = 'data/afrispeech_dialog/data/247554f8-f233-4861-bc1a-8fc327b5d5df_2b500b633e5d5ecce35433cbbb859ddc_8bW4oSXn.wav'
output_dir = 'quick_cpu_test_output'

os.makedirs(output_dir, exist_ok=True)

print('Testing EnCodec (CPU-only)...')
try:
    runner1 = EncodecRunner(bandwidth_kbps=6, causal=True, sr=16000)
    runner1.run(test_file, f'{output_dir}/encodec_cpu.wav')
    print('✅ EnCodec CPU: SUCCESS')
except Exception as e:
    print(f'❌ EnCodec CPU: FAILED - {e}')

print('Testing SoundStream (CPU-only)...')
try:
    runner2 = SoundStreamRunner(bitrate_kbps=6, sr=16000)
    runner2.run(test_file, f'{output_dir}/soundstream_cpu.wav')
    print('✅ SoundStream CPU: SUCCESS')
except Exception as e:
    print(f'❌ SoundStream CPU: FAILED - {e}')

print('=== CPU-ONLY TEST COMPLETE ===')
"

echo "=== QUICK CPU TEST COMPLETE ==="
echo "Date: $(date)"
