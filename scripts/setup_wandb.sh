#!/bin/bash

# WandB Configuration Setup Script
# This script sets up WandB for codec training

WANDB_API_KEY="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"

echo "=========================================="
echo "Setting up Weights & Biases (WandB)"
echo "=========================================="

# Login to WandB
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
source .venv/bin/activate

echo "Logging in to WandB..."
wandb login $WANDB_API_KEY

if [ $? -eq 0 ]; then
    echo "✓ Successfully logged in to WandB!"
else
    echo "✗ WandB login failed"
    exit 1
fi

# Verify login
echo ""
echo "Verifying WandB status..."
wandb verify

echo ""
echo "=========================================="
echo "WandB Setup Complete!"
echo "=========================================="
echo ""
echo "Your WandB key is now configured."
echo "All training runs will be logged to your WandB dashboard."
echo ""
echo "Next: Run training with SLURM scripts in batch_scripts/"
