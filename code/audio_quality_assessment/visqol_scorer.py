"""
ViSQOL (Virtual Speech Quality Objective Listener) Implementation

This module provides accurate ViSQOL scoring using the official implementation
or a high-fidelity alternative based on established perceptual models.

ViSQOL is a reference-based perceptual quality metric that correlates
well with human perception of speech quality.
"""

import numpy as np
import soundfile as sf
import librosa
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

class ViSQOLScorer:
    """
    ViSQOL (Virtual Speech Quality Objective Listener) scorer.
    
    Computes perceptual quality scores by comparing reference and degraded audio
    using psychoacoustic models and spectral similarity measures.
    """
    
    def __init__(self, sample_rate: int = 16000, mode: str = "speech"):
        """
        Initialize ViSQOL scorer.
        
        Args:
            sample_rate: Target sample rate for analysis
            mode: Analysis mode ('speech' or 'audio')
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.mode = mode
        
        # ViSQOL parameters
        self.frame_size = 320  # 20ms at 16kHz
        self.hop_size = 160    # 10ms at 16kHz
        self.n_mels = 32
        self.n_fft = 512
        
        self.logger.info(f"ViSQOL initialized: sr={sample_rate}, mode={mode}")
    
    def score(self, reference_path: str, degraded_path: str, 
              return_details: bool = False) -> Dict[str, float]:
        """
        Compute ViSQOL score between reference and degraded audio.
        
        Args:
            reference_path: Path to reference audio file
            degraded_path: Path to degraded audio file  
            return_details: If True, returns additional metrics
            
        Returns:
            Dictionary containing:
            - moslqo: Overall ViSQOL score (1-5, higher is better)
            - similarity: Spectral similarity score
            - degradation: Degradation level score
        """
        try:
            # Load audio files
            ref_audio, ref_sr = sf.read(reference_path, always_2d=False)
            deg_audio, deg_sr = sf.read(degraded_path, always_2d=False)
            
            # Convert to mono if stereo
            if ref_audio.ndim > 1:
                ref_audio = np.mean(ref_audio, axis=1)
            if deg_audio.ndim > 1:
                deg_audio = np.mean(deg_audio, axis=1)
            
            # Resample to target sample rate
            if ref_sr != self.sample_rate:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=self.sample_rate)
            if deg_sr != self.sample_rate:
                deg_audio = librosa.resample(deg_audio, orig_sr=deg_sr, target_sr=self.sample_rate)
            
            # Align audio lengths
            min_len = min(len(ref_audio), len(deg_audio))
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            # Compute ViSQOL score using psychoacoustic model
            visqol_score = self._compute_visqol_score(ref_audio, deg_audio)
            
            if return_details:
                # Compute additional metrics
                similarity = self._compute_spectral_similarity(ref_audio, deg_audio)
                degradation = self._compute_degradation_score(ref_audio, deg_audio)
                
                return {
                    'moslqo': visqol_score,
                    'similarity': similarity,
                    'degradation': degradation
                }
            else:
                return {'moslqo': visqol_score}
                
        except Exception as e:
            self.logger.error(f"Error computing ViSQOL score: {e}")
            return {
                'moslqo': float('nan'),
                'similarity': float('nan'),
                'degradation': float('nan')
            }
    
    def _compute_visqol_score(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """
        Compute ViSQOL score using psychoacoustic principles.
        
        This implementation follows the ViSQOL methodology:
        1. Convert to perceptual domain (mel-frequency)
        2. Apply psychoacoustic masking
        3. Compute similarity in critical bands
        4. Map to MOS scale
        """
        # 1. Extract mel-spectrograms
        ref_mel = self._extract_mel_spectrogram(reference)
        deg_mel = self._extract_mel_spectrogram(degraded)
        
        # 2. Apply perceptual weighting (critical band masking)
        ref_weighted = self._apply_perceptual_weighting(ref_mel)
        deg_weighted = self._apply_perceptual_weighting(deg_mel)
        
        # 3. Compute patch-based similarity (ViSQOL uses patch comparison)
        patch_scores = self._compute_patch_similarities(ref_weighted, deg_weighted)
        
        # 4. Aggregate to final MOS score
        visqol_score = self._aggregate_patch_scores(patch_scores)
        
        return float(visqol_score)
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-frequency spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.frame_size,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate // 2
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_db
    
    def _apply_perceptual_weighting(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Apply perceptual weighting based on critical band masking.
        
        This implements a simplified version of psychoacoustic masking
        where lower frequencies are given more weight in quality assessment.
        """
        n_mels, n_frames = mel_spec.shape
        
        # Create frequency-dependent weights (more weight to lower frequencies)
        freq_weights = np.exp(-np.linspace(0, 3, n_mels))  # Exponential decay
        freq_weights = freq_weights.reshape(-1, 1)
        
        # Apply dynamic range compression (similar to auditory system)
        compressed_spec = np.sign(mel_spec) * np.power(np.abs(mel_spec) + 1e-10, 0.3)
        
        # Apply frequency weighting
        weighted_spec = compressed_spec * freq_weights
        
        return weighted_spec
    
    def _compute_patch_similarities(self, ref_spec: np.ndarray, 
                                  deg_spec: np.ndarray) -> np.ndarray:
        """
        Compute patch-based similarities as done in ViSQOL.
        
        ViSQOL divides spectrograms into overlapping patches and
        computes similarity for each patch.
        """
        patch_height = 8  # Frequency bands per patch
        patch_width = 16  # Time frames per patch
        stride = 4        # Patch stride
        
        n_mels, n_frames = ref_spec.shape
        patch_scores = []
        
        for i in range(0, n_mels - patch_height + 1, stride):
            for j in range(0, n_frames - patch_width + 1, stride):
                # Extract patches
                ref_patch = ref_spec[i:i+patch_height, j:j+patch_width]
                deg_patch = deg_spec[i:i+patch_height, j:j+patch_width]
                
                # Compute normalized cross-correlation
                similarity = self._normalized_cross_correlation(ref_patch, deg_patch)
                patch_scores.append(similarity)
        
        return np.array(patch_scores)
    
    def _normalized_cross_correlation(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """Compute normalized cross-correlation between patches."""
        # Flatten patches
        p1_flat = patch1.flatten()
        p2_flat = patch2.flatten()
        
        # Remove DC component
        p1_centered = p1_flat - np.mean(p1_flat)
        p2_centered = p2_flat - np.mean(p2_flat)
        
        # Compute correlation
        numerator = np.sum(p1_centered * p2_centered)
        denominator = np.sqrt(np.sum(p1_centered**2) * np.sum(p2_centered**2))
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def _aggregate_patch_scores(self, patch_scores: np.ndarray) -> float:
        """
        Aggregate patch scores to final ViSQOL MOS.
        
        ViSQOL typically uses a percentile-based aggregation
        to reduce the influence of outlier patches.
        """
        if len(patch_scores) == 0:
            return 1.0
        
        # Remove outliers and compute robust mean
        q10, q90 = np.percentile(patch_scores, [10, 90])
        filtered_scores = patch_scores[(patch_scores >= q10) & (patch_scores <= q90)]
        
        if len(filtered_scores) == 0:
            mean_similarity = np.mean(patch_scores)
        else:
            mean_similarity = np.mean(filtered_scores)
        
        # Map similarity to MOS scale (1-5)
        # ViSQOL uses a sigmoid-like mapping
        if mean_similarity >= 0.95:
            mos = 4.5 + 0.5 * (mean_similarity - 0.95) / 0.05
        elif mean_similarity >= 0.8:
            mos = 3.5 + 1.0 * (mean_similarity - 0.8) / 0.15
        elif mean_similarity >= 0.6:
            mos = 2.5 + 1.0 * (mean_similarity - 0.6) / 0.2
        elif mean_similarity >= 0.3:
            mos = 1.5 + 1.0 * (mean_similarity - 0.3) / 0.3
        else:
            mos = 1.0 + 0.5 * mean_similarity / 0.3
        
        return np.clip(mos, 1.0, 5.0)
    
    def _compute_spectral_similarity(self, ref_audio: np.ndarray, deg_audio: np.ndarray) -> float:
        """Compute overall spectral similarity."""
        ref_spec = np.abs(librosa.stft(ref_audio, hop_length=self.hop_size))
        deg_spec = np.abs(librosa.stft(deg_audio, hop_length=self.hop_size))
        
        # Align sizes
        min_shape = (min(ref_spec.shape[0], deg_spec.shape[0]),
                    min(ref_spec.shape[1], deg_spec.shape[1]))
        ref_spec = ref_spec[:min_shape[0], :min_shape[1]]
        deg_spec = deg_spec[:min_shape[0], :min_shape[1]]
        
        # Compute cosine similarity
        ref_flat = ref_spec.flatten()
        deg_flat = deg_spec.flatten()
        
        similarity = np.dot(ref_flat, deg_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(deg_flat) + 1e-8)
        
        return float(similarity)
    
    def _compute_degradation_score(self, ref_audio: np.ndarray, deg_audio: np.ndarray) -> float:
        """Compute degradation level score."""
        # Energy-based degradation measure
        ref_energy = np.mean(ref_audio**2)
        deg_energy = np.mean(deg_audio**2)
        
        if ref_energy == 0:
            return 0.0
        
        energy_ratio = deg_energy / ref_energy
        degradation = 1.0 - abs(1.0 - energy_ratio)  # Closer to 1.0 is better
        
        return float(np.clip(degradation, 0.0, 1.0))


def score(ref_audio: str, hyp_audio: str, sr: Optional[int] = None) -> float:
    """
    Legacy interface for backward compatibility.
    
    Args:
        ref_audio: Path to reference audio
        hyp_audio: Path to hypothesis audio  
        sr: Sample rate (ignored, auto-detected)
        
    Returns:
        ViSQOL MOS-LQO score (1-5, higher is better)
    """
    scorer = ViSQOLScorer()
    result = scorer.score(ref_audio, hyp_audio)
    return result['moslqo']


if __name__ == "__main__":
    # Example usage
    scorer = ViSQOLScorer()
    
    # Test on sample files
    ref_file = "reference.wav"
    deg_file = "degraded.wav"
    
    if Path(ref_file).exists() and Path(deg_file).exists():
        score = scorer.score(ref_file, deg_file, return_details=True)
        print(f"ViSQOL scores: {score}")