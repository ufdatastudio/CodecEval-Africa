#!/bin/bash
# Simple batch script to run codec evaluation
# Usage: ./run_codec_batch.sh

echo "=========================================="
echo "CODEC EVALUATION BATCH SCRIPT"
echo "=========================================="
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Change to project directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set environment variables
export PYTHONPATH="/orange/ufdatastudios/c.okocha/CodecEval-Africa/code:$PYTHONPATH"

# Check if test script exists
if [ ! -f "test_codecs_batch.py" ]; then
    echo "ERROR: test_codecs_batch.py not found!"
    exit 1
fi

# Check if input file exists
INPUT_FILE="/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data/0adaefab-c0fa-4d55-9564-100d2bd5bd93_86a60667f1b75930c7844e37494b97f7_UxiL1B07.wav"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input audio file not found: $INPUT_FILE"
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo "File size: $(du -h "$INPUT_FILE" | cut -f1)"

# Create output directory
mkdir -p codec_test_output

# Run the evaluation
echo "Starting codec evaluation..."
python test_codecs_batch.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ BATCH JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Output directory: $(pwd)/codec_test_output"
    echo ""
    echo "Generated files:"
    ls -la codec_test_output/
    echo ""
    echo "Summary:"
    echo "- 6 codecs tested (EnCodec, LanguageCodec, DAC, SemantiCodec, UniCodec, APCodec)"
    echo "- 5 bitrates per codec (1.5, 3, 6, 12, 24 kbps)"
    echo "- 30 total output files generated"
    echo "- Results saved to codec_test_output/test_results.txt"
else
    echo ""
    echo "=========================================="
    echo "❌ BATCH JOB FAILED"
    echo "=========================================="
    echo "Check the output above for error details"
    exit 1
fi

