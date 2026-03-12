#!/usr/bin/env python3
"""
Wrapper script to run SemantiCodec on multilingual dataset
"""
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'codecs'))

# Import SemantiCodec components
import torch
from semanticodec import SemantiCodec
import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    # Device configuration
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Input & Output Paths
    input_dir = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"
    output_root = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/SemantiCodec_outputs"
    os.makedirs(output_root, exist_ok=True)
    
    # Bitrate Configurations
    configs = [
        {"token_rate": 25,  "semantic_vocab_size": 4096,   "label": "0.31kbps"},
        {"token_rate": 50,  "semantic_vocab_size": 4096,   "label": "0.63kbps"},
        {"token_rate": 100, "semantic_vocab_size": 4096,   "label": "1.25kbps"},
        {"token_rate": 25,  "semantic_vocab_size": 8192,   "label": "0.33kbps"},
        {"token_rate": 50,  "semantic_vocab_size": 16384,  "label": "0.68kbps"},
        {"token_rate": 100, "semantic_vocab_size": 32768,  "label": "1.40kbps"},
    ]
    
    # Gather Audio Files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Run SemantiCodec at 6 Bitrates
    for cfg in configs:
        label = cfg["label"]
        out_dir = os.path.join(output_root, label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- Running SemantiCodec @ {label} ---")
        semanticodec = SemantiCodec(
            token_rate=cfg["token_rate"],
            semantic_vocab_size=cfg["semantic_vocab_size"]
        ).to(device)

        for audio_path in tqdm(audio_files, desc=f"{label}"):
            filename = os.path.basename(audio_path)
            out_path = os.path.join(out_dir, filename)
            
            # Skip if already processed
            if os.path.exists(out_path):
                continue
            
            try:
                # Encode
                tokens = semanticodec.encode(audio_path)
                # Decode
                waveform = semanticodec.decode(tokens)

                # Save reconstruction
                sf.write(out_path, waveform[0, 0], 16000)
            except Exception as e:
                print(f" Error processing {filename}: {e}")

        # Optional: print approximate bitrate
        bitrate_kbps = (
            cfg["token_rate"] *
            (torch.log2(torch.tensor(cfg["semantic_vocab_size"])).item())
        ) / 1000
        print(f"Approximate bitrate: {bitrate_kbps:.2f} kbps")

    print("\n All six bitrate reconstructions complete!")

