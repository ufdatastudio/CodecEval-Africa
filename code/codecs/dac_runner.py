import os
import torch
import soundfile as sf
import torchaudio
import numpy as np
from tqdm import tqdm
import dac
from dac import DAC

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_audio(path, target_sr):
    """Load and resample mono audio."""
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav, target_sr

def save_audio(tensor, path, sr):
    """Save tensor to WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.squeeze().cpu().numpy()
    sf.write(path, tensor, sr)

def reconstruct_chunked(model, wav, chunk_sec=10.0, overlap_sec=0.25):
    """Encode+decode long audio in overlapping chunks."""
    sr = model.sample_rate
    hop = int((chunk_sec - overlap_sec) * sr)
    win = int(chunk_sec * sr)
    wav = wav.cpu().numpy()
    rec = np.zeros_like(wav)
    weight = np.zeros_like(wav)

    for start in range(0, len(wav), hop):
        end = min(start + win, len(wav))
        chunk = wav[start:end]
        x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(model.device)
        with torch.no_grad():
            # DAC encode returns a tuple/dict, we need to extract the codes
            z, codes, latents, _, _ = model.encode(x)
            # DAC decode expects the latent z
            y = model.decode(z).squeeze().cpu().numpy()
        rec[start:end] += y[: end - start]
        weight[start:end] += 1

    rec /= np.maximum(weight, 1e-6)
    return torch.from_numpy(rec).float()

# ---------------------------------------------------------
# Main Batch Compression Function
# ---------------------------------------------------------
def compress_folder(
    input_dir: str,
    output_root: str,
    bitrates=("8kbps", "16kbps", "24kbps"),
    chunk_sec=10.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Using device: {device}")

    # Download and load DAC models
    model_types = {
        "8kbps": "16khz",  # DAC 16kHz model operates at ~8kbps
        "16kbps": "24khz", # DAC 24kHz model operates at ~16kbps
        "24kbps": "44khz"  # DAC 44kHz model operates at ~24kbps
    }
    
    models = {}
    for br, model_type in model_types.items():
        print(f"Downloading DAC model: {model_type}...")
        model_path = dac.utils.download(model_type=model_type)
        print(f"Loading DAC model from: {model_path}")
        models[br] = DAC.load(model_path).eval().to(device)

    # Create output folders
    out_dirs = {br: os.path.join(output_root, f"out_{br}") for br in bitrates}
    for path in out_dirs.values():
        os.makedirs(path, exist_ok=True)

    # Collect .wav files recursively
    files = []
    for root, _, fs in os.walk(input_dir):
        for f in fs:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))
    print(f"\n Found {len(files)} audio files in {input_dir}")

    # Process files
    for in_path in tqdm(files, desc="Compressing audio files"):
        filename = os.path.basename(in_path)

        for br, model in models.items():
            sr = model.sample_rate
            wav, _ = load_audio(in_path, sr)
            y = reconstruct_chunked(model, wav, chunk_sec=chunk_sec)
            out_path = os.path.join(out_dirs[br], filename)
            save_audio(y, out_path, sr)

    print("\n Compression completed!")
    for br, out_dir in out_dirs.items():
        print(f"   • {br} → {out_dir}")


if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/DAC_outputs"
    compress_folder(input_folder, output_folder)
