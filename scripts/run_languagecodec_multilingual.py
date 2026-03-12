#!/usr/bin/env python3
"""
Wrapper script to run LanguageCodec on multilingual dataset
"""
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))

from languagecodec_runner import run_languagecodec_folder

if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/LanguageCodec_outputs"
    
    # LanguageCodec paths
    config_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/Languagecodec/configs/languagecodec_mm.yaml"
    ckpt_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/Languagecodec/pretrained/languagecodec_paper.ckpt"
    
    # LanguageCodec supports bandwidth_ids 0-3 (4 different bandwidth embeddings at ~6.6 kbps)
    bandwidth_ids = [0, 1, 2, 3]
    
    run_languagecodec_folder(input_folder, output_folder, config_path, ckpt_path, bandwidth_ids, device='cpu')

