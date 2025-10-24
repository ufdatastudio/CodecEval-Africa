"""
SemantiCodec Six-Bitrate Evaluation Script
------------------------------------------
Author: Chibuzor Okocha
Purpose: Encode & decode local audio at six preset bitrates using SemantiCodec
"""

import os
import torch
import soundfile as sf
from semanticodec import SemantiCodec

# ==============================
# 1. Device configuration
# ==============================
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ==============================
# 2. Input audio
# ==============================
filepath = "test/test.wav"   # 🔸 Change to your own file path
assert os.path.exists(filepath), f"Audio file not found: {filepath}"

# ==============================
# 3. Define six bitrate configs
# ==============================
configs = [
    {"token_rate": 25, "semantic_vocab_size": 4096,  "label": "0.31kbps"},
    {"token_rate": 50, "semantic_vocab_size": 4096,  "label": "0.63kbps"},
    {"token_rate": 100, "semantic_vocab_size": 4096, "label": "1.25kbps"},
    {"token_rate": 25, "semantic_vocab_size": 8192,  "label": "0.33kbps"},
    {"token_rate": 50, "semantic_vocab_size": 16384, "label": "0.68kbps"},
    {"token_rate": 100, "semantic_vocab_size": 32768, "label": "1.40kbps"},
]

# ==============================
# 4. Loop through configs
# ==============================
for cfg in configs:
    print(f"\n--- Running SemantiCodec @ {cfg['label']} ---")

    semanticodec = SemantiCodec(
        token_rate=cfg["token_rate"],
        semantic_vocab_size=cfg["semantic_vocab_size"]
    ).to(device)

    # Encode
    tokens = semanticodec.encode(filepath)
    print(f"Encoded tokens shape: {tokens.shape}")

    # Decode
    waveform = semanticodec.decode(tokens)
    print(f"Decoded waveform shape: {waveform.shape}")

    # Save reconstruction
    output_path = f"output_{cfg['label']}.wav"
    sf.write(output_path, waveform[0, 0], 16000)
    print(f"✅ Saved: {output_path}")

    # Compute and print approximate bitrate
    bitrate_kbps = (
        cfg["token_rate"] *
        (torch.log2(torch.tensor(cfg["semantic_vocab_size"])).item())
    ) / 1000
    print(f"Approximate bitrate: {bitrate_kbps:.2f} kbps")

print("\nAll six reconstructions complete!")
