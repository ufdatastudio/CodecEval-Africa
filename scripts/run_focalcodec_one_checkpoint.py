#!/usr/bin/env python3
"""Run one FocalCodec checkpoint on one dataset."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))
from focalcodec_runner import focalcodec_folder


DATASET_MAP = {
    "afrispeech_dialog": {
        "input": "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data",
        "output": "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_dialog/FocalCodec_outputs",
    },
    "afrinames": {
        "input": "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afri_names_150_flat",
        "output": "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrinames/FocalCodec_outputs",
    },
    "afrispeech_multilingual": {
        "input": "/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav",
        "output": "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/FocalCodec_outputs",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run one FocalCodec checkpoint on one dataset")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_MAP.keys()))
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    input_folder = DATASET_MAP[args.dataset]["input"]
    output_folder = DATASET_MAP[args.dataset]["output"]

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")

    focalcodec_folder(
        input_dir=input_folder,
        output_root=output_folder,
        config=args.checkpoint,
        variant_name=args.checkpoint.split("/")[-1],
        skip_existing=True,
    )


if __name__ == "__main__":
    main()
