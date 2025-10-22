import numpy as np
import soundfile as sf
from typing import Optional

def compute_f0_rmse(ref_audio: str, hyp_audio: str, sr: Optional[int] = None) -> float:
    """
    Compute F0 RMSE between reference and hypothesis audio.
    
    Measures prosody preservation by comparing fundamental frequency (pitch) contours.
    
    Args:
        ref_audio: Path to reference audio file
        hyp_audio: Path to hypothesis audio file
        sr: Sample rate (optional)
        
    Returns:
        F0 RMSE (lower = better prosody preservation)
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
        
        # Extract F0 contours
        ref_f0 = _extract_f0(ref_wave, target_sr)
        hyp_f0 = _extract_f0(hyp_wave, target_sr)
        
        # Ensure same length
        min_len = min(len(ref_f0), len(hyp_f0))
        ref_f0 = ref_f0[:min_len]
        hyp_f0 = hyp_f0[:min_len]
        
        # Compute RMSE
        rmse = _compute_f0_rmse(ref_f0, hyp_f0)
        
        return float(rmse)
        
    except Exception as e:
        print(f"Error computing F0 RMSE: {e}")
        return float('nan')

def _extract_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract fundamental frequency (F0) contour using autocorrelation.
    
    This is a simplified F0 extraction that works reasonably well for speech.
    """
    # Parameters
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.01 * sr)     # 10ms hop
    min_f0 = 50   # 50 Hz minimum
    max_f0 = 400  # 400 Hz maximum
    
    f0_contour = []
    
    # Process in frames
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        
        # Apply window
        windowed_frame = frame * np.hanning(len(frame))
        
        # Compute autocorrelation
        autocorr = np.correlate(windowed_frame, windowed_frame, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find F0 using autocorrelation
        f0 = _find_f0_autocorr(autocorr, sr, min_f0, max_f0)
        f0_contour.append(f0)
    
    return np.array(f0_contour)

def _find_f0_autocorr(autocorr: np.ndarray, sr: int, min_f0: int, max_f0: int) -> float:
    """Find F0 from autocorrelation."""
    # Convert frequency bounds to period bounds
    min_period = int(sr / max_f0)
    max_period = int(sr / min_f0)
    
    if max_period >= len(autocorr):
        return 0.0
    
    # Find peak in period range
    search_region = autocorr[min_period:max_period]
    if len(search_region) == 0:
        return 0.0
    
    # Find local maximum
    peak_idx = np.argmax(search_region)
    period = peak_idx + min_period
    
    # Check if peak is significant
    peak_value = autocorr[period]
    baseline = np.mean(autocorr[1:min_period]) if min_period > 1 else 0
    
    # Threshold for valid F0
    if peak_value < baseline * 1.2:  # 20% above baseline
        return 0.0
    
    f0 = sr / period
    return float(f0)

def _compute_f0_rmse(ref_f0: np.ndarray, hyp_f0: np.ndarray) -> float:
    """Compute RMSE between F0 contours."""
    # Remove unvoiced frames (F0 = 0)
    voiced_mask = (ref_f0 > 0) & (hyp_f0 > 0)
    
    if np.sum(voiced_mask) == 0:
        return float('nan')  # No voiced frames
    
    ref_voiced = ref_f0[voiced_mask]
    hyp_voiced = hyp_f0[voiced_mask]
    
    # Compute RMSE
    squared_errors = (ref_voiced - hyp_voiced) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    
    return float(rmse)

def compute_prosody_metrics(ref_audio: str, hyp_audio: str) -> dict:
    """
    Compute comprehensive prosody metrics.
    
    Returns:
        Dictionary with F0 RMSE and other prosody measures
    """
    try:
        f0_rmse = compute_f0_rmse(ref_audio, hyp_audio)
        
        # Additional prosody metrics could be added here:
        # - Jitter (F0 variability)
        # - Shimmer (amplitude variability) 
        # - Speaking rate
        # - Pause duration
        
        return {
            "f0_rmse": f0_rmse,
            "prosody_score": max(0, 1 - f0_rmse / 50) if not np.isnan(f0_rmse) else 0  # Convert to 0-1 score
        }
        
    except Exception as e:
        print(f"Error computing prosody metrics: {e}")
        return {
            "f0_rmse": float('nan'),
            "prosody_score": 0.0
        }
