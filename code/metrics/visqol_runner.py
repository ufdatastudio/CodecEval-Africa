import numpy as np
import soundfile as sf
from typing import Optional

def score(ref_audio: str, hyp_audio: str, sr: Optional[int] = None) -> float:
    """
    Compute ViSQOL score for perceptual quality assessment.
    
    This is a simplified implementation that computes spectral similarity metrics
    as a proxy for ViSQOL until the full model is integrated.
    
    Args:
        ref_audio: Path to reference audio file
        hyp_audio: Path to hypothesis audio file
        sr: Sample rate (optional)
        
    Returns:
        ViSQOL score (higher = better quality)
    """
    try:
        # Load both audio files
        ref_wave, ref_sr = sf.read(ref_audio, always_2d=False)
        hyp_wave, hyp_sr = sf.read(hyp_audio, always_2d=False)
        
        # Convert to mono if stereo
        if ref_wave.ndim > 1:
            ref_wave = np.mean(ref_wave, axis=1)
        if hyp_wave.ndim > 1:
            hyp_wave = np.mean(hyp_wave, axis=1)
        
        # Resample to same sample rate
        if ref_sr != hyp_sr:
            import librosa
            target_sr = max(ref_sr, hyp_sr)
            ref_wave = librosa.resample(ref_wave.astype(float), orig_sr=ref_sr, target_sr=target_sr)
            hyp_wave = librosa.resample(hyp_wave.astype(float), orig_sr=hyp_sr, target_sr=target_sr)
        
        # Ensure same length
        min_len = min(len(ref_wave), len(hyp_wave))
        ref_wave = ref_wave[:min_len]
        hyp_wave = hyp_wave[:min_len]
        
        # Compute ViSQOL proxy
        visqol_score = _compute_visqol_proxy(ref_wave, hyp_wave)
        
        return float(visqol_score)
        
    except Exception as e:
        print(f"Error computing ViSQOL score: {e}")
        return float('nan')

def _compute_visqol_proxy(ref_audio: np.ndarray, hyp_audio: np.ndarray) -> float:
    """
    Compute ViSQOL proxy using spectral similarity metrics.
    
    ViSQOL measures perceptual quality by comparing spectral characteristics.
    """
    # 1. Spectral Centroid similarity
    ref_centroid = _spectral_centroid(ref_audio)
    hyp_centroid = _spectral_centroid(hyp_audio)
    centroid_sim = 1 - abs(ref_centroid - hyp_centroid) / (ref_centroid + 1e-8)
    
    # 2. Spectral Rolloff similarity
    ref_rolloff = _spectral_rolloff(ref_audio)
    hyp_rolloff = _spectral_rolloff(hyp_audio)
    rolloff_sim = 1 - abs(ref_rolloff - hyp_rolloff) / (ref_rolloff + 1e-8)
    
    # 3. Spectral Bandwidth similarity
    ref_bandwidth = _spectral_bandwidth(ref_audio)
    hyp_bandwidth = _spectral_bandwidth(hyp_audio)
    bandwidth_sim = 1 - abs(ref_bandwidth - hyp_bandwidth) / (ref_bandwidth + 1e-8)
    
    # 4. MFCC similarity
    mfcc_sim = _mfcc_similarity(ref_audio, hyp_audio)
    
    # 5. SNR-based quality
    snr_quality = _compute_snr_quality(ref_audio, hyp_audio)
    
    # Combine metrics (scale 0-5 like ViSQOL)
    visqol_score = (
        centroid_sim * 1.0 +      # Spectral shape
        rolloff_sim * 1.0 +       # Spectral distribution
        bandwidth_sim * 1.0 +     # Spectral width
        mfcc_sim * 1.5 +          # Timbre similarity
        snr_quality * 0.5         # Noise level
    )
    
    return np.clip(visqol_score, 0, 5)

def _spectral_centroid(audio: np.ndarray) -> float:
    """Compute spectral centroid."""
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(fft), d=1)[:len(fft)//2]
    
    if np.sum(magnitude) == 0:
        return 0.0
    
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    return float(centroid)

def _spectral_rolloff(audio: np.ndarray, rolloff_threshold: float = 0.85) -> float:
    """Compute spectral rolloff."""
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(fft), d=1)[:len(fft)//2]
    
    cumsum_magnitude = np.cumsum(magnitude)
    total_magnitude = cumsum_magnitude[-1]
    
    if total_magnitude == 0:
        return 0.0
    
    rolloff_point = np.where(cumsum_magnitude >= rolloff_threshold * total_magnitude)[0]
    if len(rolloff_point) > 0:
        return float(freqs[rolloff_point[0]])
    return float(freqs[-1])

def _spectral_bandwidth(audio: np.ndarray) -> float:
    """Compute spectral bandwidth."""
    centroid = _spectral_centroid(audio)
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(fft), d=1)[:len(fft)//2]
    
    if np.sum(magnitude) == 0:
        return 0.0
    
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
    return float(bandwidth)

def _mfcc_similarity(ref_audio: np.ndarray, hyp_audio: np.ndarray) -> float:
    """Compute MFCC similarity."""
    # Simple MFCC-like features using DCT
    ref_mfcc = _simple_mfcc(ref_audio)
    hyp_mfcc = _simple_mfcc(hyp_audio)
    
    # Ensure same length
    min_len = min(len(ref_mfcc), len(hyp_mfcc))
    ref_mfcc = ref_mfcc[:min_len]
    hyp_mfcc = hyp_mfcc[:min_len]
    
    # Cosine similarity
    dot_product = np.dot(ref_mfcc, hyp_mfcc)
    norms = np.linalg.norm(ref_mfcc) * np.linalg.norm(hyp_mfcc)
    
    if norms == 0:
        return 0.0
    
    similarity = dot_product / norms
    return float(similarity)

def _simple_mfcc(audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Compute simple MFCC-like features."""
    # Apply windowing and FFT
    windowed = audio * np.hanning(len(audio))
    fft = np.fft.fft(windowed)
    magnitude = np.abs(fft[:len(fft)//2])
    
    # Log magnitude
    log_magnitude = np.log(magnitude + 1e-8)
    
    # Simple DCT (MFCC approximation) - use scipy if numpy doesn't have dct
    try:
        mfcc = np.fft.dct(log_magnitude)[:n_mfcc]
    except AttributeError:
        # Fallback to scipy DCT
        from scipy.fft import dct
        mfcc = dct(log_magnitude)[:n_mfcc]
    
    return mfcc

def _compute_snr_quality(ref_audio: np.ndarray, hyp_audio: np.ndarray) -> float:
    """Compute SNR-based quality measure."""
    # Estimate noise as difference
    noise = ref_audio - hyp_audio
    
    signal_power = np.mean(ref_audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return 1.0  # Perfect reconstruction
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Convert SNR to quality score (0-1)
    quality = np.clip(snr_db / 40, 0, 1)  # 40dB = perfect quality
    
    return float(quality)
