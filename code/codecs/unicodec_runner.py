
"""
UniCodec Batch Evaluation Script
---------------------------------
Author: Chibuzor Okocha
Purpose: Encode & decode audio files using UniCodec at multiple bandwidth settings
"""

import os
import sys
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm
from encodec.utils import convert_audio

# Add UniCodec to path
unicodec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'UniCodec')
sys.path.insert(0, unicodec_path)

from decoder.pretrained import Unicodec

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_audio(path, target_sr):
    """Load and resample audio."""
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # Convert to mono
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav, target_sr

def save_audio(tensor, path, sr):
    """Save tensor to WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.squeeze().cpu().numpy()
    sf.write(path, tensor, sr)

# ---------------------------------------------------------
# Main Batch Compression Function
# ---------------------------------------------------------
def compress_folder(
    input_dir: str,
    output_root: str,
    config_path: str,
    model_path: str,
    bandwidth_ids=(0, 1, 2, 3),
    domain: str = "2",
    device: str = "cuda"
):
    """
    Process all audio files in input_dir using UniCodec.
    
    Args:
        input_dir: Directory containing input wav files
        output_root: Root directory for outputs
        config_path: Path to UniCodec config file
        model_path: Path to UniCodec checkpoint
        bandwidth_ids: Tuple of bandwidth IDs to use (0-3, corresponding to different quality levels)
        domain: Domain setting for encoder
        device: Device to run on ('cuda' or 'cpu')
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Resolve kbps labels from config if available
    kbps_list = None
    try:
        # quick load of bandwidths from config yaml
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        kbps_list = cfg["model"]["init_args"]["feature_extractor"]["init_args"].get("bandwidths")
    except Exception:
        kbps_list = None

    # Create output folders labeled by kbps when possible
    out_dirs = {}
    for bw_id in bandwidth_ids:
        if kbps_list and bw_id < len(kbps_list):
            label = f"out_{kbps_list[bw_id]}kbps"
        else:
            label = f"out_bw{bw_id}"
        out_dirs[bw_id] = os.path.join(output_root, label)
    for path in out_dirs.values():
        os.makedirs(path, exist_ok=True)
    
    # Collect .wav files recursively
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))
    
    print(f"\nFound {len(audio_files)} audio files in {input_dir}")
    print(f"Processing with bandwidth IDs: {bandwidth_ids}")
    
    # Load model once (reloading on GPU causes OOM)
    print("\nLoading UniCodec model...")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {model_path}")
    codec = Unicodec.from_pretrained0802(config_path, model_path)
    codec = codec.to(device).eval()
    print("Model loaded successfully!")
    
    # Process files - single model load with cleanup between files
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        filename = os.path.basename(audio_path)
        
        # Check if already processed
        all_exist = True
        for bw_id in bandwidth_ids:
            out_path = os.path.join(out_dirs[bw_id], filename)
            if not os.path.exists(out_path):
                all_exist = False
                break
        
        if all_exist:
            continue  # Skip if all bandwidth outputs exist
        
        try:
            # Load audio at 24kHz (UniCodec's native sample rate)
            wav, sr = torchaudio.load(audio_path)
            wav = convert_audio(wav, sr, 24000, 1)  # Convert to 24kHz mono
            
            # Check duration and determine if chunking is needed
            sample_rate = 24000
            duration_seconds = wav.shape[-1] / sample_rate
            chunk_duration_seconds = 480  # 8 minutes - safe chunk size for GPU
            needs_chunking = duration_seconds > chunk_duration_seconds
            
            # Process with each bandwidth setting
            for bw_id in bandwidth_ids:
                out_path = os.path.join(out_dirs[bw_id], filename)
                if os.path.exists(out_path):
                    continue  # Skip if this bandwidth output exists
                
                bandwidth_tensor = torch.tensor([bw_id]).to(device)
                
                if needs_chunking:
                    # Process in chunks for long files
                    chunk_samples = int(chunk_duration_seconds * sample_rate)
                    total_samples = wav.shape[-1]
                    chunks = []
                    
                    print(f"\nProcessing {filename} in chunks ({duration_seconds:.1f}s, {int((total_samples + chunk_samples - 1) / chunk_samples)} chunks)...")
                    
                    for chunk_start in range(0, total_samples, chunk_samples):
                        chunk_end = min(chunk_start + chunk_samples, total_samples)
                        wav_chunk = wav[..., chunk_start:chunk_end].to(device)
                        
                        try:
                            with torch.no_grad():
                                # Encode chunk
                                features, discrete_code = codec.encode_infer(wav_chunk, domain, bandwidth_id=bandwidth_tensor)
                                
                                # Decode chunk
                                audio_out = codec.decode(features, bandwidth_id=bandwidth_tensor)
                                
                                # Store chunk result
                                chunks.append(audio_out.cpu())
                                
                                # Cleanup chunk tensors
                                del features, discrete_code, audio_out, wav_chunk
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                    
                        except Exception as chunk_error:
                            print(f"\nError processing chunk {chunk_start//chunk_samples + 1} of {filename}: {chunk_error}")
                            del wav_chunk
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            raise chunk_error
                    
                    # Concatenate all chunks
                    if chunks:
                        audio_out = torch.cat(chunks, dim=-1)
                        del chunks
                        
                        # Save reconstructed audio
                        save_audio(audio_out, out_path, sample_rate)
                        del audio_out
                        
                else:
                    # Process entire file at once (short files)
                    wav_device = wav.to(device)
                    
                    try:
                        with torch.no_grad():
                            # Encode
                            features, discrete_code = codec.encode_infer(wav_device, domain, bandwidth_id=bandwidth_tensor)

                            # Decode
                            audio_out = codec.decode(features, bandwidth_id=bandwidth_tensor)
                            
                            # Save reconstructed audio
                            save_audio(audio_out, out_path, sample_rate)
                            
                            # Cleanup tensors after each bandwidth
                            del features, discrete_code, audio_out
                    finally:
                        del wav_device
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
            
            # Cleanup after processing all bandwidths for this file
            if 'wav' in locals():
                del wav
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()
                        
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            # Cleanup on error
            if 'wav' in locals():
                del wav
            if 'bandwidth_tensor' in locals():
                del bandwidth_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue
    
    print("\nCompression completed!")
    for bw_id, out_dir in out_dirs.items():
        print(f"  Bandwidth ID {bw_id} → {out_dir}")


if __name__ == "__main__":
    # Configuration
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afri_names_150_flat"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrinames/UniCodec_outputs"
    
    # UniCodec model paths
    config_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/UniCodec/configs/unicodec_frame75_10s_nq1_code16384_dim512_finetune.yaml"
    model_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/UniCodec/checkpoints/unicode.ckpt"
    
    # Run compression with 4 bandwidth settings
    compress_folder(
        input_dir=input_folder,
        output_root=output_folder,
        config_path=config_path,
        model_path=model_path,
        bandwidth_ids=(0,),  # Single bandwidth-id; effective ~0.35 kbps with current checkpoint
        domain="2",
        device="cuda"  # Use GPU if available, falls back to CPU
    )

