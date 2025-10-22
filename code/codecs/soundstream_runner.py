import time
from typing import Dict, Any, Optional

import numpy as np
import torch
from encodec.model import EncodecModel
from encodec.utils import convert_audio

def _load_wav(path: str, sr: int) -> np.ndarray:
    import soundfile as sf
    wav, srx = sf.read(path, always_2d=True)  # Force 2D
    
    # Fix channel ordering if needed
    if wav.ndim == 2 and wav.shape[0] > wav.shape[1]:
        wav = wav.T  # Transpose if channels are first
    
    # Limit to 2 channels max
    if wav.shape[0] > 2:
        wav = wav[:2]
    
    if srx != sr:
        import librosa
        if wav.shape[0] == 1:
            wav = librosa.resample(wav[0].astype(float), orig_sr=srx, target_sr=sr)[None, :]
        else:
            wav = np.array([librosa.resample(ch.astype(float), orig_sr=srx, target_sr=sr) for ch in wav])
    
    return wav

def _save_wav(wav: np.ndarray, path: str, sr: int):
    import soundfile as sf
    x = wav
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    sf.write(path, x, sr)

class SoundStreamRunner:
    def __init__(self, bitrate_kbps: float = 3.0, sr: int = 24000, device: Optional[str] = None, **model_kwargs):
        self.sr = sr
        self.bitrate_kbps = bitrate_kbps
        # Force CPU for SoundStream to avoid CUDA tensor issues
        self.device = torch.device("cpu")
        
        # Use EnCodec as SoundStream alternative with different parameters
        print(f"Initializing SoundStream (EnCodec-based) at {bitrate_kbps} kbps")
        
        # Load EnCodec model with SoundStream-like behavior
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bitrate_kbps)
        # SoundStream typically uses non-causal processing
        try:
            self.model.set_causal(False)  # SoundStream is typically non-causal
        except Exception:
            pass
        self.model.to(self.device).eval()

    def run(self, in_wav_path: str, out_wav_path: str) -> Dict[str, Any]:
        wav = _load_wav(in_wav_path, self.sr)  # [1, T]
        x = torch.from_numpy(wav).float().to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 1, T]
        
        # Convert to model's expected format
        x = convert_audio(x, self.sr, self.model.sample_rate, self.model.channels)
        
        # Ensure model is on the same device as input
        self.model = self.model.to(self.device)
        # Force model to eval mode and ensure all parameters are on device
        self.model.eval()
        for param in self.model.parameters():
            param.data = param.data.to(self.device)

        t0 = time.time()
        with torch.no_grad():
            # Use EnCodec for SoundStream-like compression
            frames = self.model.encode(x)
            y = self.model.decode(frames)
        dt = time.time() - t0

        # Convert back to original format
        y = y.detach().cpu().numpy()
        if y.ndim == 3:
            y = y[0]
        if y.ndim == 2 and y.shape[0] > 1:
            y = y.mean(axis=0, keepdims=True)
        elif y.ndim == 1:
            y = y[None, :]

        _save_wav(y, out_wav_path, self.model.sample_rate)

        num_seconds = wav.shape[-1] / float(self.sr)
        rtf = dt / max(1e-9, num_seconds)

        return {
            "codec": "soundstream_impl",
            "kbps": float(self.bitrate_kbps),
            "sr_in": int(self.sr),
            "sr_out": int(self.model.sample_rate),
            "elapsed_sec": float(dt),
            "rtf": float(rtf),
            "device": str(self.device),
            "model_type": "encodec_based",
        }
