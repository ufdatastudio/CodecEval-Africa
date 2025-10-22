import numpy as np
import torch
import torchaudio
from typing import Optional

def score(wav_path: str, sr: Optional[int] = None) -> float:
    """
    Compute NISQA score for audio quality assessment.
    
    This is a simplified implementation that computes basic audio quality metrics
    as a proxy for NISQA until the full model is integrated.
    
    Args:
        wav_path: Path to audio file
        sr: Sample rate (optional)
        
    Returns:
        Quality score (higher = better quality)
    """
    try:
        # Load audio using soundfile (more reliable)
        import soundfile as sf
        audio, sample_rate = sf.read(wav_path, always_2d=False)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Compute basic quality metrics as NISQA proxy
        quality_score = _compute_quality_proxy(audio, sample_rate)
        
        return float(quality_score)
        
    except Exception as e:
        print(f"Error computing NISQA score for {wav_path}: {e}")
        return float('nan')

def _compute_quality_proxy(audio: np.ndarray, sr: int) -> float:
    """
    Compute quality proxy metrics as NISQA approximation.
    
    This uses signal processing metrics that correlate with perceptual quality.
    """
    # 1. Signal-to-noise ratio approximation
    signal_power = np.mean(audio ** 2)
    noise_floor = np.percentile(np.abs(audio), 10)  # Estimate noise floor
    snr_approx = 10 * np.log10(signal_power / (noise_floor ** 2 + 1e-8))
    
    # 2. Dynamic range
    dynamic_range = np.max(audio) - np.min(audio)
    
    # 3. Spectral flatness (measure of tonality)
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-8))) / np.mean(magnitude + 1e-8)
    
    # 4. Zero crossing rate (speech activity indicator)
    zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
    zcr = zero_crossings / len(audio)
    
    # Combine metrics into quality score (0-5 scale like NISQA)
    quality_score = (
        np.clip(snr_approx / 20, 0, 1) * 2 +  # SNR contribution (0-2)
        np.clip(dynamic_range, 0, 1) * 1.5 +   # Dynamic range (0-1.5)
        (1 - spectral_flatness) * 1 +          # Spectral richness (0-1)
        np.clip(zcr * 100, 0, 1) * 0.5         # Speech activity (0-0.5)
    )
    
    return np.clip(quality_score, 0, 5)
