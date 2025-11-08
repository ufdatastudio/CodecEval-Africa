"""
Encodec Multi-Bitrate Batch Evaluation
--------------------------------------
Author: Chibuzor Okocha
Purpose: Encode & decode all audio files in a folder at multiple bitrates using Encodec
"""

import os
import time
import torch
import numpy as np
import soundfile as sf
import torchaudio
from tqdm import tqdm
from encodec import EncodecModel
from encodec.utils import convert_audio

# ------------------------------------------------------------
# 1. Load & Save Utilities
# ------------------------------------------------------------
def load_wav(path, target_sr):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # convert to mono
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav

def save_wav(tensor, path, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.squeeze().cpu().numpy()
    sf.write(path, tensor, sr)

# ------------------------------------------------------------
# 2. Chunked Encodec Run (to handle long files safely)
# ------------------------------------------------------------
def encodec_reconstruct(model, wav, chunk_sec=10.0, device='cuda'):
    sr = model.sample_rate
    chunk_len = int(chunk_sec * sr)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(wav), chunk_len):
            end = min(start + chunk_len, len(wav))
            x = wav[start:end].unsqueeze(0).unsqueeze(0).to(device)
            encoded = model.encode(x)
            decoded = model.decode(encoded)
            chunks.append(decoded.squeeze().cpu())
    return torch.cat(chunks)

# ------------------------------------------------------------
# 3. Batch Runner
# ------------------------------------------------------------
def encodec_folder(
    input_dir: str,
    output_root: str,
    bitrates=(3.0, 6.0, 12.0, 24.0),
    sr: int = 24000,
    causal: bool = True,
    chunk_sec: float = 10.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Using device: {device}")

    # Load model once
    model = EncodecModel.encodec_model_24khz()
    try:
        model.set_causal(causal)
    except Exception:
        pass
    model.to(device).eval()

    # Collect audio files recursively
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))
    print(f"🎧 Found {len(audio_files)} .wav files in {input_dir}")

    # Run over multiple bitrates
    for kbps in bitrates:
        print(f"\n--- Running Encodec @ {kbps} kbps ---")
        out_dir = os.path.join(output_root, f"out_{int(kbps)}kbps")
        os.makedirs(out_dir, exist_ok=True)

        # Set bitrate
        model.set_target_bandwidth(kbps)

        for audio_path in tqdm(audio_files, desc=f"{kbps} kbps"):
            try:
                filename = os.path.basename(audio_path)
                wav = load_wav(audio_path, sr)

                # Convert & resample for model input
                x = convert_audio(
                    wav.unsqueeze(0).unsqueeze(0),
                    sr,
                    model.sample_rate,
                    model.channels
                )

                t0 = time.time()
                y = encodec_reconstruct(model, x[0, 0], chunk_sec=chunk_sec, device=device)
                elapsed = time.time() - t0

                out_path = os.path.join(out_dir, filename)
                save_wav(y, out_path, model.sample_rate)

                # Optional logging
                num_seconds = len(wav) / sr
                print(f"{filename}: RTF={elapsed / num_seconds:.3f}")
            except Exception as e:
                print(f" Error with {audio_path}: {e}")

    print("\n All bitrate reconstructions complete!")

# ------------------------------------------------------------
# 4. Run Example
# ------------------------------------------------------------
if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afri_names_150_flat"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrinames/Encodec_outputs"

    encodec_folder(
        input_dir=input_folder,
        output_root=output_folder,
        bitrates=(3.0, 6.0, 12.0, 24.0),  # Adjust as needed
        sr=24000,
        causal=True
    )
