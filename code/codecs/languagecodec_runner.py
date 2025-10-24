"""
LanguageCodec Runner - Language-specific neural audio codec
Based on: https://github.com/jishengpeng/Languagecodec
Uses the actual LanguageCodec inference API
"""

import os
import sys
import time
import torch
import torchaudio
import numpy as np
import soundfile as sf
import yaml
from typing import Dict, Any, Optional

# Add LanguageCodec to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Languagecodec'))

def _load_wav(path: str, sr: int) -> np.ndarray:
    """Load audio file and convert to mono at target sample rate."""
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # Convert to mono
    if orig_sr != sr:
        # Simple resampling (for production, use proper resampling)
        ratio = sr / orig_sr
        new_length = int(len(wav) * ratio)
        wav = np.interp(np.linspace(0, len(wav), new_length), np.arange(len(wav)), wav)
    return wav

def _save_wav(wav: np.ndarray, path: str, sr: int):
    """Save audio file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    sf.write(path, wav, sr)

class LanguageCodecRunner:
    """
    LanguageCodec runner using the actual LanguageCodec inference API.
    
    This uses the real LanguageCodec implementation with discrete token representation.
    """
    
    def __init__(self, bitrate_kbps: float = 6.0, sr: int = 24000, device: Optional[str] = None, **model_kwargs):
        self.sr = sr
        self.bitrate_kbps = bitrate_kbps
        self.device = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Map bitrate to bandwidth_id for LanguageCodec
        # LanguageCodec supports multiple bandwidth levels
        self.bandwidth_id = self._map_bitrate_to_bandwidth(bitrate_kbps)
        
        # Initialize LanguageCodec model
        print(f"LanguageCodec runner initialized (real implementation) at {bitrate_kbps} kbps (bandwidth_id={self.bandwidth_id})")
        self.model = self._load_languagecodec_model()
    
    def _map_bitrate_to_bandwidth(self, bitrate_kbps):
        """Map bitrate to LanguageCodec bandwidth_id"""
        # LanguageCodec typically supports bandwidth IDs from 0-7
        # Map bitrate ranges to bandwidth IDs
        if bitrate_kbps <= 2.0:
            return 0  # Lowest quality
        elif bitrate_kbps <= 4.0:
            return 1
        elif bitrate_kbps <= 6.0:
            return 2
        elif bitrate_kbps <= 9.0:
            return 3
        elif bitrate_kbps <= 12.0:
            return 4
        elif bitrate_kbps <= 18.0:
            return 5
        elif bitrate_kbps <= 24.0:
            return 6
        else:
            return 7  # Highest quality
        
    def _load_languagecodec_model(self):
        """Load the actual LanguageCodec model using the correct API pattern."""
        try:
            # Import the correct LanguageCodec model (Vocos)
            from languagecodec_decoder.pretrained import Vocos
            
            print("Loading LanguageCodec model...")
            
            # Use the correct API pattern from the README
            config_path = "Languagecodec/configs/languagecodec_mm.yaml"
            ckpt_path = "Languagecodec/pretrained/languagecodec_paper.ckpt"
            
            print(f"Loading from config: {config_path}")
            print(f"Loading from checkpoint: {ckpt_path}")
            
            # Use the correct Vocos API
            model = Vocos.from_pretrained0802(config_path, ckpt_path)
            model = model.to(self.device)
            model.eval()
            
            print("✓ LanguageCodec model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error: Could not load LanguageCodec model: {e}")
            raise RuntimeError(f"Failed to load LanguageCodec model: {e}")
    
    
    def run(self, in_wav_path: str, out_wav_path: str) -> Dict[str, Any]:
        """Run LanguageCodec encoding/decoding using the actual API."""
        try:
            # Load audio
            wav = _load_wav(in_wav_path, self.sr)
            
            # Convert to tensor and ensure proper dimensions for LanguageCodec
            x = torch.from_numpy(wav).float().to(self.device)
            
            # LanguageCodec expects (B, T) format - batch and time dimensions only
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension: (1, T)
            elif x.dim() == 2:
                if x.shape[0] > x.shape[1]:  # If first dim is larger, it's likely (T, C)
                    x = x.transpose(0, 1)  # Convert to (C, T)
                x = x.mean(dim=0, keepdim=True)  # Convert to mono: (1, T)
            
            # Ensure we have (B, T) format
            if x.dim() == 3:
                x = x.squeeze(1)  # Remove channel dimension if present
            
            t0 = time.time()
            
            # Use LanguageCodec direct forward pass (most reliable method)
            with torch.no_grad():
                # Direct forward pass with bitrate-specific bandwidth_id parameter
                y = self.model(x, bandwidth_id=torch.tensor([self.bandwidth_id]))
            
            dt = time.time() - t0
            
            # Convert back to numpy
            y = y.squeeze().cpu().numpy()
            if y.ndim == 2:
                y = y[0]  # Remove batch dimension
            
            # Save output
            _save_wav(y, out_wav_path, self.sr)
            
            # Calculate metrics
            num_seconds = wav.shape[-1] / float(self.sr)
            rtf = dt / max(1e-9, num_seconds)
            
            return {
                "codec": "languagecodec",
                "kbps": float(self.bitrate_kbps),
                "sr_in": int(self.sr),
                "sr_out": int(self.sr),
                "elapsed_sec": float(dt),
                "rtf": float(rtf),
                "device": str(self.device),
                "model_type": "language_specific_discrete",
            }
            
        except Exception as e:
            print(f"LanguageCodec error: {e}")
            raise RuntimeError(f"LanguageCodec processing failed: {e}")
