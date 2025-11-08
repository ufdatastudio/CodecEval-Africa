#!/usr/bin/env python3
"""
WavTokenizer Batch Evaluation Script
------------------------------------
Author: Chibuzor Okocha
Purpose: Encode & decode audio files using WavTokenizer at multiple model configurations
"""

import os
import sys
import torch
import torchaudio
import gc
from tqdm import tqdm
from pathlib import Path

# Add WavTokenizer to path
wavtokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'WavTokenizer')
sys.path.insert(0, wavtokenizer_path)

from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer

# ---------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------
MODEL_CONFIGS = [
    {
        "name": "small-600-24k-4096",
        "config": "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "checkpoint": "wavtokenizer_small-600-24k-4096.ckpt",
        "bandwidth_id": [0],
        "description": "Small model, 40 tokens/sec"
    },
    {
        "name": "small-320-24k-4096",
        "config": "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "checkpoint": "wavtokenizer_small-320-24k-4096.ckpt",
        "bandwidth_id": [0],
        "description": "Small model, 75 tokens/sec"
    },
    {
        "name": "large-unify-40token",
        "config": "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",  # Use compatible config
        "checkpoint": "large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt",
        "bandwidth_id": [0],
        "description": "Large unified model, 40 tokens/sec"
    },
    {
        "name": "large-speech-75token",
        "config": "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",  # Use compatible config
        "checkpoint": "large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt",
        "bandwidth_id": [0],
        "description": "Large speech model, 75 tokens/sec"
    },
    {
        "name": "medium-speech-75token",
        "config": "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "checkpoint": "medium-speech-75token/wavtokenizer_medium_speech_320_24k_v2.ckpt",
        "bandwidth_id": [0],
        "description": "Medium speech model, 75 tokens/sec"
    },
    {
        "name": "medium-music-audio-75token",
        "config": "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "checkpoint": "medium-music-audio-75token/wavtokenizer_medium_music_audio_320_24k_v2.ckpt",
        "bandwidth_id": [0],
        "description": "Medium music/audio model, 75 tokens/sec"
    },
]

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_audio_files(input_dir):
    """Get all audio files from directory."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, f))
    return sorted(audio_files)

def save_audio(tensor, path, sample_rate=24000):
    """Save tensor to WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.cpu()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    torchaudio.save(
        path,
        tensor.unsqueeze(0) if tensor.dim() == 1 else tensor,
        sample_rate=sample_rate,
        encoding='PCM_S',
        bits_per_sample=16
    )

# ---------------------------------------------------------
# Main Compression Function
# ---------------------------------------------------------
def compress_folder(
    input_dir: str,
    output_root: str,
    base_dir: str,
    model_configs: list = None,
    device: str = "cuda"
):
    """
    Compress all audio files in a folder using WavTokenizer.
    
    Args:
        input_dir: Directory containing input audio files
        output_root: Root directory for output files
        base_dir: Base directory of the project (for finding WavTokenizer paths)
        model_configs: List of model configurations to use (default: all)
        device: Device to use ('cuda' or 'cpu')
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if model_configs is None:
        model_configs = MODEL_CONFIGS
    
    # Get all audio files
    audio_files = get_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"\nFound {len(audio_files)} audio files to process")
    
    # Process each model configuration
    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        print(f"\n{'='*80}")
        print(f"Processing with model: {model_name}")
        print(f"Description: {model_cfg['description']}")
        print(f"{'='*80}")
        
        # Set up paths
        config_path = os.path.join(base_dir, "WavTokenizer", "configs", model_cfg["config"])
        checkpoint_path = os.path.join(base_dir, "WavTokenizer", "checkpoints", model_cfg["checkpoint"])
        out_dir = os.path.join(output_root, f"WavTokenizer_{model_name}")
        
        # Verify paths exist
        if not os.path.exists(config_path):
            print(f"Warning: Config not found: {config_path}")
            print(f"  Skipping model: {model_name}")
            continue
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping model: {model_name}")
            continue
        
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output: {out_dir}")
        
        # Load model once for this configuration
        print(f"\nLoading WavTokenizer model: {model_name}...")
        try:
            wavtokenizer = WavTokenizer.from_pretrained0802(config_path, checkpoint_path)
            wavtokenizer = wavtokenizer.to(device).eval()
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Process all audio files
        bandwidth_id = torch.tensor(model_cfg["bandwidth_id"]).to(device)
        sample_rate = 24000
        
        for audio_path in tqdm(audio_files, desc=f"{model_name}"):
            filename = os.path.basename(audio_path)
            # Preserve extension or change to .wav
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(out_dir, f"{base_name}.wav")
            
            # Skip if already processed
            if os.path.exists(out_path):
                continue
            
            try:
                # Load audio
                wav, sr = torchaudio.load(audio_path)
                
                # Convert to 24kHz mono (WavTokenizer requirement)
                wav = convert_audio(wav, sr, sample_rate, 1)
                
                # Check duration for chunking long files
                duration_seconds = wav.shape[-1] / sample_rate
                chunk_duration_seconds = 480  # 8 minutes - safe chunk size
                needs_chunking = duration_seconds > chunk_duration_seconds
                
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
                                features, discrete_code = wavtokenizer.encode_infer(wav_chunk, bandwidth_id=bandwidth_id)
                                
                                # Decode chunk
                                audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
                                
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
                        save_audio(audio_out, out_path, sample_rate)
                        del audio_out
                
                else:
                    # Process entire file at once (short files)
                    wav_device = wav.to(device)
                    
                    try:
                        with torch.no_grad():
                            # Encode
                            features, discrete_code = wavtokenizer.encode_infer(wav_device, bandwidth_id=bandwidth_id)
                            
                            # Decode
                            audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
                            
                            # Save reconstructed audio
                            save_audio(audio_out, out_path, sample_rate)
                            
                            # Cleanup
                            del features, discrete_code, audio_out
                    finally:
                        del wav_device
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                # Cleanup after processing file
                del wav
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                # Cleanup on error
                if 'wav' in locals():
                    del wav
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Cleanup model after processing all files for this configuration
        del wavtokenizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print(f"\nCompleted processing with model: {model_name}")
    
    print("\n" + "="*80)
    print("All models processed!")
    print("="*80)


if __name__ == "__main__":
    # Configuration
    base_dir = "/orange/ufdatastudios/c.okocha/CodecEval-Africa"
    input_folder = os.path.join(base_dir, "data", "afri_names_150_flat")
    output_folder = os.path.join(base_dir, "outputs", "afrinames", "WavTokenizer_outputs")
    
    # You can specify which models to use (default: all)
    # Example: use only small models
    # selected_models = [MODEL_CONFIGS[0], MODEL_CONFIGS[1]]
    
    # Run compression with all models
    compress_folder(
        input_dir=input_folder,
        output_root=output_folder,
        base_dir=base_dir,
        model_configs=None,  # None = use all models
        device="cuda"  # Use GPU if available, falls back to CPU
    )

