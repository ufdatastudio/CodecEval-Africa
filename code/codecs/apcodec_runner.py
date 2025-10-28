"""
APCodec Batch Evaluation Script
-------------------------------
Author: Chibuzor Okocha
Purpose: Encode & decode all audio files in a folder using pretrained APCodec (YangAi520/APCodec)
"""

import os
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm
from model.apcodec import APCodecModel  # from YangAi520/APCodec repo

def run_apcodec_folder(input_dir, output_root, ckpt_path, sr=24000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pretrained checkpoint
    model = APCodecModel.load_from_checkpoint(ckpt_path)
    model = model.eval().to(device)

    os.makedirs(output_root, exist_ok=True)

    # Gather all .wav files
    wav_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(input_dir)
        for f in fs if f.lower().endswith(".wav")
    ]
    print(f"Found {len(wav_files)} audio files in {input_dir}")

    for path in tqdm(wav_files, desc="APCodec Inference"):
        filename = os.path.basename(path)
        out_path = os.path.join(output_root, filename)

        try:
            wav, file_sr = sf.read(path)
            if file_sr != sr:
                wav = torchaudio.functional.resample(torch.tensor(wav).unsqueeze(0), file_sr, sr).squeeze(0)
            if wav.ndim > 1:
                wav = wav.mean(dim=0)

            x = wav.unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                z = model.encode(x)
                y = model.decode(z)

            sf.write(out_path, y.squeeze().cpu().numpy(), sr)
        except Exception as e:
            print(f" Error processing {filename}: {e}")

    print(f"\n Completed reconstruction → {output_root}")

# Example
if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/APCodec_outputs"
    checkpoint = "/orange/ufdatastudios/c.okocha/APCodec/weights/apcodec_24khz.ckpt"  # update path

    run_apcodec_folder(input_folder, output_folder, ckpt_path=checkpoint, sr=24000)
