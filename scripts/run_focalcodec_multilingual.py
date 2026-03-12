#!/usr/bin/env python3
"""Run FocalCodec on afrispeech_multilingual dataset."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))
from focalcodec_runner import focalcodec_folder, DEFAULT_FOCALCODEC_CONFIGS


if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/FocalCodec_outputs"
    focalcodec_folder(input_folder, output_folder, configs=DEFAULT_FOCALCODEC_CONFIGS)
