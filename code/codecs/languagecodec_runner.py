"""
LanguageCodec Runner - Batch processing for AfriSpeech dataset
Processes audio files at multiple bandwidth IDs
"""

import os
import sys
import time
import torch
import torchaudio
import numpy as np
import json
import soundfile as sf
from pathlib import Path

# Add LanguageCodec to path
languagecodec_path = os.path.join(os.path.dirname(__file__), '../../Languagecodec')
sys.path.insert(0, languagecodec_path)

from languagecodec_decoder.pretrained import Vocos

def resample_audio(wav, orig_sr, target_sr, target_channels=1):
    """Pure PyTorch audio resampling without torchcodec dependency"""
    # Convert to torch tensor if needed
    if not isinstance(wav, torch.Tensor):
        wav = torch.from_numpy(wav).float()
    
    print(f"      resample_audio: input shape={wav.shape}, orig_sr={orig_sr}, target_sr={target_sr}")
    
    # Ensure proper shape: (batch, channels, time) for torchaudio.functional.resample
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)  # (time,) -> (1, 1, time)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(0)  # (channels, time) -> (1, channels, time)
    
    print(f"      resample_audio: after unsqueeze shape={wav.shape}")
    
    # Resample using torchaudio - expects (batch, channels, time)
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
        print(f"      resample_audio: after resample shape={wav.shape}")
    
    # Convert to mono if needed (average across channel dimension, not time)
    print(f"      resample_audio: checking mono conversion: wav.shape[1]={wav.shape[1]} (channels), target_channels={target_channels}")
    if wav.shape[1] > target_channels:
        wav = wav.mean(dim=1, keepdim=True)
        print(f"      resample_audio: after mono conversion shape={wav.shape}")
    else:
        print(f"      resample_audio: skipping mono conversion")
    
    result = wav.squeeze(0)  # Remove batch dimension to get (channels, time)
    print(f"      resample_audio: final result shape={result.shape}")
    return result


def run_languagecodec_folder(input_folder, output_folder, config_path, ckpt_path, bandwidth_ids, device='cuda'):
    """
    Run LanguageCodec on all WAV files in input_folder at multiple bandwidth IDs.
    
    Args:
        input_folder: Path to folder containing input WAV files
        output_folder: Path to folder for output WAV files
        config_path: Path to LanguageCodec config YAML
        ckpt_path: Path to LanguageCodec checkpoint
        bandwidth_ids: List of bandwidth IDs to test (e.g., [0, 1, 2, 3])
        device: 'cuda' or 'cpu'
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load LanguageCodec model
    print(f"Loading LanguageCodec from config: {config_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    
    model = Vocos.from_pretrained0802(config_path, ckpt_path)
    model = model.to(device)
    model.eval()
    
    print("✓ LanguageCodec model loaded successfully")
    
    # Get all WAV files
    wav_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Process each bandwidth ID
    for bandwidth_id in bandwidth_ids:
        print(f"\n{'='*80}")
        print(f"Processing with bandwidth_id={bandwidth_id}")
        print(f"{'='*80}")
        
        # Create output directory for this bandwidth
        bw_output_dir = os.path.join(output_folder, f"bandwidth_{bandwidth_id}")
        os.makedirs(bw_output_dir, exist_ok=True)
        
        # Track metrics
        results = []
        total_input_sec = 0.0
        total_proc_sec = 0.0
        
        for i, wav_file in enumerate(wav_files):
            input_path = os.path.join(input_folder, wav_file)
            output_path = os.path.join(bw_output_dir, wav_file)
            
            try:
                # Load audio using soundfile to avoid torchcodec issues
                wav, sr = sf.read(input_path)
                print(f"    Loaded audio: shape={wav.shape}, sr={sr}")
                
                # Convert to 24kHz mono using pure PyTorch
                print(f"    Before resample: wav.shape={wav.shape}, sr={sr}")
                wav = resample_audio(wav, sr, 24000, 1)
                print(f"    After resample: shape={wav.shape}")
                wav = wav.to(device)
                
                # Create bandwidth_id tensor
                bw_tensor = torch.tensor([bandwidth_id], device=device)
                print(f"    About to encode with bandwidth_id={bandwidth_id}")
                
                # Measure processing time
                t0 = time.time()
                
                with torch.no_grad():
                    # Encode and decode
                    print(f"    Calling encode_infer...")
                    features, discrete_code = model.encode_infer(wav, bandwidth_id=bw_tensor)
                    print(f"    Encode successful: features={features.shape}, discrete_code={discrete_code.shape}")
                    audio_out = model.decode(features, bandwidth_id=bw_tensor)
                    print(f"    Decode successful: audio_out={audio_out.shape}")
                
                dt = time.time() - t0
                
                # Save output
                torchaudio.save(
                    output_path,
                    audio_out.cpu(),
                    sample_rate=24000,
                    encoding='PCM_S',
                    bits_per_sample=16
                )
                
                # Calculate metrics
                num_seconds = wav.shape[-1] / 24000.0
                rtf = dt / max(1e-9, num_seconds)
                
                total_input_sec += num_seconds
                total_proc_sec += dt
                
                result = {
                    "file": wav_file,
                    "bandwidth_id": bandwidth_id,
                    "duration_sec": float(num_seconds),
                    "processing_time_sec": float(dt),
                    "rtf": float(rtf),
                    "discrete_code_shape": list(discrete_code.shape)
                }
                results.append(result)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(wav_files):
                    print(f"  Processed {i+1}/{len(wav_files)} files | "
                          f"Last RTF: {rtf:.4f} | "
                          f"Avg RTF: {total_proc_sec/total_input_sec:.4f}")
                
            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")
                continue
        
        # Save results
        avg_rtf = total_proc_sec / total_input_sec if total_input_sec > 0 else 0.0
        
        summary = {
            "bandwidth_id": bandwidth_id,
            "total_files": len(wav_files),
            "successful_files": len(results),
            "total_audio_duration_sec": float(total_input_sec),
            "total_processing_time_sec": float(total_proc_sec),
            "average_rtf": float(avg_rtf),
            "device": str(device),
            "sample_rate": 24000,
            "per_file_results": results
        }
        
        # Save JSON results
        json_path = os.path.join(bw_output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Bandwidth ID {bandwidth_id} complete:")
        print(f"  - Processed: {len(results)}/{len(wav_files)} files")
        print(f"  - Total audio: {total_input_sec:.2f}s")
        print(f"  - Processing time: {total_proc_sec:.2f}s")
        print(f"  - Average RTF: {avg_rtf:.4f}")
        print(f"  - Output: {bw_output_dir}")
        print(f"  - Results: {json_path}")


if __name__ == "__main__":
    # Paths
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/LanguageCodec_outputs"
    
    # LanguageCodec paths
    config_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/Languagecodec/configs/languagecodec_mm.yaml"
    ckpt_path = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/Languagecodec/pretrained/languagecodec_paper.ckpt"
    
    # LanguageCodec supports bandwidth_ids 0-3 (4 different bandwidth embeddings at ~6.6 kbps)
    bandwidth_ids = [0, 1, 2, 3]
    
    run_languagecodec_folder(input_folder, output_folder, config_path, ckpt_path, bandwidth_ids)