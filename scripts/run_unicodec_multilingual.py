#!/usr/bin/env python3
"""
Wrapper script to run UniCodec on multilingual dataset
"""
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))

from unicodec_runner import compress_folder

if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/UniCodec_outputs"
    
    # UniCodec model paths
    config_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/UniCodec/configs/unicodec_frame75_10s_nq1_code16384_dim512_finetune.yaml"
    model_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/UniCodec/checkpoints/unicode.ckpt"
    
    # Run compression with bandwidth settings
    compress_folder(
        input_dir=input_folder,
        output_root=output_folder,
        config_path=config_path,
        model_path=model_path,
        bandwidth_ids=(0,),
        domain="2",
        device="cuda"
    )

