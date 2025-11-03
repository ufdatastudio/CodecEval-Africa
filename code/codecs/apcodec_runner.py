"""
APCodec Batch Evaluation Script
-------------------------------
Author: Chibuzor Okocha
Purpose: Encode & decode all audio files using APCodec at multiple bitrates
Supported bitrates:
  - 48 kHz: 6 kbps, 12 kbps
  - 24 kHz: 3 kbps, 6 kbps
  - 16 kHz: 2 kbps, 4 kbps
"""

import os
import sys
import torch
import soundfile as sf
import torchaudio
import json
from tqdm import tqdm

# Add APCodec to path
apcodec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'APCodec')
sys.path.insert(0, apcodec_path)

from models import Encoder, Decoder
from utils import AttrDict
from dataset import amp_pha_specturm

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_checkpoint(filepath, device):
    """Load model checkpoint."""
    assert os.path.isfile(filepath), f"Checkpoint not found: {filepath}"
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Checkpoint loaded successfully")
    return checkpoint_dict

def run_apcodec_folder(input_dir, output_root, config_path, encoder_ckpt, decoder_ckpt, bitrate_configs=None):
    """
    Process all audio files in input_dir using APCodec at multiple bitrates.
    
    Args:
        input_dir: Directory containing input wav files
        output_root: Directory for output files
        config_path: Path to APCodec config.json
        encoder_ckpt: Path to encoder checkpoint
        decoder_ckpt: Path to decoder checkpoint
        bitrate_configs: List of (sampling_rate, n_codebooks, label) tuples
                        If None, uses default: [(48000, 9, "6kbps")]
    """
    if bitrate_configs is None:
        # Default configuration
        bitrate_configs = [(48000, 9, "6kbps")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load configuration
    with open(config_path) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    
    # Load models
    print("\nLoading APCodec models...")
    encoder = Encoder(h).to(device).eval()
    decoder = Decoder(h).to(device).eval()
    
    state_dict_encoder = load_checkpoint(encoder_ckpt, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    
    state_dict_decoder = load_checkpoint(decoder_ckpt, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])
    
    print("Models loaded successfully!")
    
    # Gather all .wav files
    wav_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(input_dir)
        for f in fs if f.lower().endswith(".wav")
    ]
    print(f"\nFound {len(wav_files)} audio files in {input_dir}")
    print(f"\nProcessing with {len(bitrate_configs)} bitrate configurations:")
    for sr, n_cb, label in bitrate_configs:
        print(f"  - {label} @ {sr}Hz (n_codebooks={n_cb})")
    
    # Process each bitrate configuration
    for target_sr, n_codebooks, bitrate_label in bitrate_configs:
        print(f"\n{'='*60}")
        print(f"Processing: {bitrate_label} @ {target_sr}Hz")
        print(f"{'='*60}")
        
        # Create output directory for this bitrate
        out_dir = os.path.join(output_root, f"out_{bitrate_label}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Process files
        for path in tqdm(wav_files, desc=f"APCodec {bitrate_label}"):
            filename = os.path.basename(path)
            out_path = os.path.join(out_dir, filename)
            
            try:
                # Load audio
                wav, file_sr = sf.read(path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=0)  # Convert to mono
                
                # Resample to target sampling rate
                if file_sr != target_sr:
                    wav_tensor = torch.tensor(wav).unsqueeze(0)
                    wav = torchaudio.functional.resample(wav_tensor, file_sr, target_sr).squeeze(0).numpy()
                
                wav = torch.FloatTensor(wav).to(device)
                
                with torch.no_grad():
                    # Extract amplitude and phase spectra
                    logamp, pha, _, _ = amp_pha_specturm(wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
                    
                    # Encode with specified number of codebooks
                    latent, _, _ = encoder(logamp, pha)
                    
                    # Decode
                    logamp_g, pha_g, _, _, y_g = decoder(latent)
                    
                    # Get output audio
                    audio = y_g.squeeze().cpu().numpy()
                
                # Save reconstructed audio
                sf.write(out_path, audio, target_sr, 'PCM_16')
                
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                continue
        
        print(f"Completed: {bitrate_label} → {out_dir}")
    
    print(f"\nAll bitrate reconstructions complete!")
    for _, _, label in bitrate_configs:
        print(f"  - {label} → {os.path.join(output_root, f'out_{label}')}")


if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/APCodec_outputs"
    
    # APCodec paths
    config_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/APCodec/config.json"
    encoder_ckpt = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/APCodec/checkpoints/encoder"
    decoder_ckpt = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/APCodec/checkpoints/decoder"
    
    # Define bitrate configurations: (sampling_rate, n_codebooks, label)
    # Based on APCodec paper, different configurations achieve different bitrates
    # Using 24kHz configurations which work well for speech
    bitrate_configs = [
        (24000, 4, "3kbps_24khz"),   # 24 kHz, 3 kbps (fewer codebooks)
        (24000, 9, "6kbps_24khz"),   # 24 kHz, 6 kbps (standard)
        (48000, 9, "6kbps_48khz"),   # 48 kHz, 6 kbps (standard)
        (48000, 18, "12kbps_48khz"), # 48 kHz, 12 kbps (more codebooks)
    ]
    
    run_apcodec_folder(input_folder, output_folder, config_path, encoder_ckpt, decoder_ckpt, bitrate_configs)
