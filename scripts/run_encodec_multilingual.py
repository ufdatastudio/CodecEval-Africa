#!/usr/bin/env python3
"""
Wrapper script to run EnCodec on multilingual dataset
"""
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))

from encodec_runner import encodec_folder

if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/Encodec_outputs"

    encodec_folder(
        input_dir=input_folder,
        output_root=output_folder,
        bitrates=(3.0, 6.0, 12.0, 24.0),
        sr=24000,
        causal=True
    )

