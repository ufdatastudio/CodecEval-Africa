#!/usr/bin/env python3
"""
Wrapper script to run DAC on multilingual dataset
"""
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))

from dac_runner import compress_folder

if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/DAC_outputs"

    compress_folder(input_folder, output_folder)

