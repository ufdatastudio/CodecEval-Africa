import numpy as np
import soundfile as sf
from typing import Optional

def score(wav_path: str, sr: Optional[int] = None) -> dict:
    """
    Compute DNSMOS scores for speech quality assessment.
    
    DNSMOS provides multiple quality dimensions:
    - SIG: Signal distortion
    - BAK: Background noise
    - OVR: Overall quality
    
    Args:
        wav_path: Path to audio file
        sr: Sample rate (optional)
        
    Returns:
        Dictionary with SIG, BAK, OVR scores
    """
    try:
        # Load audio
        audio, sample_rate = sf.read(wav_path, always_2d=False)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Compute DNSMOS proxy scores
        scores = _compute_dnsmos_proxy(audio, sample_rate)
        
        return scores
        
    except Exception as e:
        print(f"Error computing DNSMOS scores for {wav_path}: {e}")
        return {
            "sig": float('nan'),
            "bak": float('nan'), 
            "ovr": float('nan')
        }

def _compute_dnsmos_proxy(audio: np.ndarray, sr: int) -> dict:
    """
    Compute DNSMOS proxy scores using signal processing metrics.
    
    DNSMOS measures:
    - SIG: Speech signal quality
    - BAK: Background noise level
    - OVR: Overall perceptual quality
    """
    # 1. Signal Quality (SIG) - measures speech distortion
    sig_score = _compute_signal_quality(audio, sr)
    
    # 2. Background Noise (BAK) - measures noise suppression
    bak_score = _compute_background_noise(audio, sr)
    
    # 3. Overall Quality (OVR) - combination of SIG and BAK
    ovr_score = _compute_overall_quality(sig_score, bak_score)
    
    return {
        "sig": float(sig_score),
        "bak": float(bak_score),
        "ovr": float(ovr_score)
    }

def _compute_signal_quality(audio: np.ndarray, sr: int) -> float:
    """Compute signal quality score (SIG)."""
    # 1. Spectral flatness (measure of signal richness)
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])
    
    # Avoid log(0)
    magnitude = magnitude + 1e-8
    
    geometric_mean = np.exp(np.mean(np.log(magnitude)))
    arithmetic_mean = np.mean(magnitude)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    # 2. Harmonic-to-noise ratio
    hnr = _compute_harmonic_noise_ratio(audio, sr)
    
    # 3. Dynamic range
    dynamic_range = np.max(audio) - np.min(audio)
    
    # Combine metrics (scale 1-5 like DNSMOS)
    sig_score = (
        (1 - spectral_flatness) * 2.0 +    # Spectral richness (0-2)
        np.clip(hnr / 20, 0, 1) * 2.0 +    # Harmonic content (0-2)
        np.clip(dynamic_range, 0, 1) * 1.0 # Dynamic range (0-1)
    )
    
    return np.clip(sig_score, 1, 5)

def _compute_background_noise(audio: np.ndarray, sr: int) -> float:
    """Compute background noise score (BAK)."""
    # 1. Estimate noise floor
    sorted_audio = np.sort(np.abs(audio))
    noise_floor = np.percentile(sorted_audio, 10)  # Bottom 10%
    
    # 2. High-frequency noise content
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(fft), d=1/sr)[:len(fft)//2]
    
    # High frequency content (above 4kHz)
    high_freq_mask = freqs > 4000
    high_freq_energy = np.sum(magnitude[high_freq_mask])
    total_energy = np.sum(magnitude)
    
    high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
    
    # 3. Spectral noise floor
    spectral_noise = np.percentile(magnitude, 5)  # Bottom 5% of spectral magnitude
    
    # Combine metrics (scale 1-5, higher = less noise)
    bak_score = (
        np.clip(1 - noise_floor * 10, 0, 1) * 2.0 +      # Low noise floor (0-2)
        np.clip(1 - high_freq_ratio * 5, 0, 1) * 2.0 +   # Low high-freq noise (0-2)
        np.clip(1 - spectral_noise * 100, 0, 1) * 1.0    # Low spectral noise (0-1)
    )
    
    return np.clip(bak_score, 1, 5)

def _compute_overall_quality(sig_score: float, bak_score: float) -> float:
    """Compute overall quality score (OVR)."""
    # OVR is typically a weighted combination of SIG and BAK
    # Weight SIG more heavily as it's more important for speech
    ovr_score = 0.7 * sig_score + 0.3 * bak_score
    
    return np.clip(ovr_score, 1, 5)

def _compute_harmonic_noise_ratio(audio: np.ndarray, sr: int) -> float:
    """Compute harmonic-to-noise ratio."""
    # Simple pitch detection using autocorrelation
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    # Find pitch period
    min_period = int(sr / 400)  # 400 Hz max
    max_period = int(sr / 80)   # 80 Hz min
    
    if max_period >= len(autocorr):
        return 0.0
    
    pitch_region = autocorr[min_period:max_period]
    if len(pitch_region) == 0:
        return 0.0
    
    pitch_period = np.argmax(pitch_region) + min_period
    
    # Estimate harmonic energy vs noise energy
    harmonic_energy = autocorr[pitch_period]
    noise_energy = np.mean(autocorr[1:pitch_period])
    
    if noise_energy == 0:
        return 20.0  # High HNR
    
    hnr = 10 * np.log10(harmonic_energy / noise_energy)
    return float(hnr)