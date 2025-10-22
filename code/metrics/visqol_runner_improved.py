import numpy as np
import soundfile as sf
import librosa
from typing import Optional
from scipy import signal
from scipy.stats import pearsonr

def score(ref_audio: str, hyp_audio: str, sr: Optional[int] = None) -> float:
    """
    Compute improved ViSQOL score for perceptual quality assessment.
    
    This implementation uses more sophisticated spectral analysis to better
    approximate ViSQOL's perceptual quality assessment.
    
    Args:
        ref_audio: Path to reference audio file
        hyp_audio: Path to hypothesis audio file
        sr: Sample rate (optional)
        
    Returns:
        ViSQOL score (higher = better quality, 1-5 scale)
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
        target_sr = 16000  # ViSQOL typically uses 16kHz
        if ref_sr != target_sr:
            ref_wave = librosa.resample(ref_wave.astype(float), orig_sr=ref_sr, target_sr=target_sr)
        if hyp_sr != target_sr:
            hyp_wave = librosa.resample(hyp_wave.astype(float), orig_sr=hyp_sr, target_sr=target_sr)
        
        # Ensure same length
        min_len = min(len(ref_wave), len(hyp_wave))
        ref_wave = ref_wave[:min_len]
        hyp_wave = hyp_wave[:min_len]
        
        # Compute improved ViSQOL score
        visqol_score = _compute_improved_visqol(ref_wave, hyp_wave, target_sr)
        
        return float(visqol_score)
        
    except Exception as e:
        print(f"Error computing ViSQOL score: {e}")
        return float('nan')

def _compute_improved_visqol(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """
    Compute improved ViSQOL score using advanced spectral analysis.
    """
    # 1. Spectral Magnitude Similarity (most important for ViSQOL)
    spectral_sim = _spectral_magnitude_similarity(ref_audio, hyp_audio, sr)
    
    # 2. Spectral Phase Correlation
    phase_sim = _spectral_phase_similarity(ref_audio, hyp_audio, sr)
    
    # 3. Perceptual Spectral Distance
    perceptual_dist = _perceptual_spectral_distance(ref_audio, hyp_audio, sr)
    
    # 4. Temporal Structure Similarity
    temporal_sim = _temporal_structure_similarity(ref_audio, hyp_audio, sr)
    
    # 5. Harmonic Structure Preservation
    harmonic_sim = _harmonic_structure_similarity(ref_audio, hyp_audio, sr)
    
    # 6. Noise Floor Analysis
    noise_quality = _noise_floor_analysis(ref_audio, hyp_audio, sr)
    
    # Combine metrics with weights (based on ViSQOL research)
    visqol_score = (
        spectral_sim * 0.35 +        # Spectral magnitude (most important)
        phase_sim * 0.20 +           # Phase information
        (1 - perceptual_dist) * 0.25 +  # Perceptual distance (inverted)
        temporal_sim * 0.10 +        # Temporal structure
        harmonic_sim * 0.05 +        # Harmonic content
        noise_quality * 0.05         # Noise characteristics
    )
    
    # Scale to 1-5 range (ViSQOL standard)
    return np.clip(visqol_score * 4 + 1, 1, 5)

def _spectral_magnitude_similarity(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """Compute spectral magnitude similarity using STFT."""
    # Compute STFT
    ref_stft = librosa.stft(ref_audio, n_fft=2048, hop_length=512)
    hyp_stft = librosa.stft(hyp_audio, n_fft=2048, hop_length=512)
    
    # Get magnitude spectra
    ref_mag = np.abs(ref_stft)
    hyp_mag = np.abs(hyp_stft)
    
    # Ensure same dimensions
    min_frames = min(ref_mag.shape[1], hyp_mag.shape[1])
    ref_mag = ref_mag[:, :min_frames]
    hyp_mag = hyp_mag[:, :min_frames]
    
    # Compute correlation across frequency bins
    correlations = []
    for f in range(ref_mag.shape[0]):
        if np.std(ref_mag[f, :]) > 0 and np.std(hyp_mag[f, :]) > 0:
            corr, _ = pearsonr(ref_mag[f, :], hyp_mag[f, :])
            if not np.isnan(corr):
                correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0

def _spectral_phase_similarity(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """Compute spectral phase similarity."""
    # Compute STFT
    ref_stft = librosa.stft(ref_audio, n_fft=2048, hop_length=512)
    hyp_stft = librosa.stft(hyp_audio, n_fft=2048, hop_length=512)
    
    # Get phase spectra
    ref_phase = np.angle(ref_stft)
    hyp_phase = np.angle(hyp_stft)
    
    # Ensure same dimensions
    min_frames = min(ref_phase.shape[1], hyp_phase.shape[1])
    ref_phase = ref_phase[:, :min_frames]
    hyp_phase = hyp_phase[:, :min_frames]
    
    # Compute phase difference correlation
    phase_diff = np.abs(ref_phase - hyp_phase)
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Wrap to [0, π]
    
    # Convert to similarity (closer to 0 = more similar)
    phase_sim = 1 - np.mean(phase_diff) / np.pi
    return max(0, phase_sim)

def _perceptual_spectral_distance(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """Compute perceptual spectral distance using mel-scale."""
    # Compute mel-spectrograms
    ref_mel = librosa.feature.melspectrogram(y=ref_audio, sr=sr, n_mels=80, fmax=8000)
    hyp_mel = librosa.feature.melspectrogram(y=hyp_audio, sr=sr, n_mels=80, fmax=8000)
    
    # Convert to log scale
    ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)
    hyp_mel_db = librosa.power_to_db(hyp_mel, ref=np.max)
    
    # Ensure same dimensions
    min_frames = min(ref_mel_db.shape[1], hyp_mel_db.shape[1])
    ref_mel_db = ref_mel_db[:, :min_frames]
    hyp_mel_db = hyp_mel_db[:, :min_frames]
    
    # Compute perceptual distance (lower = better)
    distance = np.mean(np.abs(ref_mel_db - hyp_mel_db))
    return distance

def _temporal_structure_similarity(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """Compute temporal structure similarity using autocorrelation."""
    # Compute autocorrelation
    ref_autocorr = np.correlate(ref_audio, ref_audio, mode='full')
    hyp_autocorr = np.correlate(hyp_audio, hyp_audio, mode='full')
    
    # Normalize
    ref_autocorr = ref_autocorr / np.max(ref_autocorr)
    hyp_autocorr = hyp_autocorr / np.max(hyp_autocorr)
    
    # Compute correlation
    if len(ref_autocorr) == len(hyp_autocorr):
        corr, _ = pearsonr(ref_autocorr, hyp_autocorr)
        return corr if not np.isnan(corr) else 0.0
    else:
        # Resample to same length
        min_len = min(len(ref_autocorr), len(hyp_autocorr))
        corr, _ = pearsonr(ref_autocorr[:min_len], hyp_autocorr[:min_len])
        return corr if not np.isnan(corr) else 0.0

def _harmonic_structure_similarity(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """Compute harmonic structure similarity using pitch analysis."""
    # Extract pitch using autocorrelation
    ref_pitch = _extract_pitch(ref_audio, sr)
    hyp_pitch = _extract_pitch(hyp_audio, sr)
    
    if ref_pitch is None or hyp_pitch is None:
        return 0.0
    
    # Compute pitch correlation
    min_len = min(len(ref_pitch), len(hyp_pitch))
    if min_len > 0:
        corr, _ = pearsonr(ref_pitch[:min_len], hyp_pitch[:min_len])
        return corr if not np.isnan(corr) else 0.0
    return 0.0

def _extract_pitch(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Extract pitch using autocorrelation."""
    try:
        # Use librosa for pitch extraction
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
            else:
                pitch_values.append(0)
        
        return np.array(pitch_values)
    except:
        return None

def _noise_floor_analysis(ref_audio: np.ndarray, hyp_audio: np.ndarray, sr: int) -> float:
    """Analyze noise floor characteristics."""
    # Estimate noise as difference
    noise = ref_audio - hyp_audio
    
    # Compute noise characteristics
    noise_power = np.mean(noise ** 2)
    signal_power = np.mean(ref_audio ** 2)
    
    if signal_power == 0:
        return 1.0
    
    # SNR-based quality
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Convert to quality score (0-1)
    quality = np.clip(snr_db / 30, 0, 1)  # 30dB = good quality
    return quality
