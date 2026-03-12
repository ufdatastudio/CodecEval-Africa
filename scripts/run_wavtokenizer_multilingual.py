#!/usr/bin/env python3
"""
Wrapper script to run WavTokenizer on multilingual dataset
"""
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))

from wavtokenizer_runner import compress_folder

if __name__ == "__main__":
    # Configuration
    base_dir = "/orange/ufdatastudios/c.okocha/CodecEval-Africa"
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/WavTokenizer_outputs"
    
    # Run compression with all models
    compress_folder(
        input_dir=input_folder,
        output_root=output_folder,
        base_dir=base_dir,
        model_configs=None,  # None = use all models
        device="cuda"
    )

