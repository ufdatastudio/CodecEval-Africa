"""
Prosody Analysis Implementation

This module provides accurate prosody analysis for evaluating
how well neural codecs preserve prosodic features like F0 (pitch),
rhythm, stress, and temporal characteristics.

Uses established prosody analysis methods from phonetics and
speech processing literature.
"""

import numpy as np
import soundfile as sf
import librosa
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import warnings
from scipy import signal
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

class ProsodyAnalyzer:
    """
    Prosody analyzer for evaluating preservation of prosodic features.
    
    Analyzes fundamental frequency (F0), rhythm, stress patterns,
    and temporal characteristics in speech.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize prosody analyzer.
        
        Args:
            sample_rate: Target sample rate for analysis
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        
        # F0 analysis parameters
        self.f0_min = 50   # Minimum F0 (Hz)
        self.f0_max = 400  # Maximum F0 (Hz)
        self.frame_length = 2048
        self.hop_length = 512
        
        # Rhythm analysis parameters
        self.tempo_bounds = (60, 200)  # BPM range
        
        self.logger.info(f"Prosody analyzer initialized: sr={sample_rate}")
    
    def analyze(self, reference_path: str, degraded_path: str,
                return_details: bool = True) -> Dict[str, float]:
        """
        Analyze prosodic preservation between reference and degraded audio.
        
        Args:
            reference_path: Path to reference audio file
            degraded_path: Path to degraded audio file
            return_details: If True, returns detailed prosodic metrics
            
        Returns:
            Dictionary containing prosodic similarity scores:
            - f0_rmse: F0 RMSE (lower is better)
            - f0_correlation: F0 correlation (higher is better)  
            - rhythm_similarity: Rhythm pattern similarity (higher is better)
            - tempo_similarity: Tempo preservation (higher is better)
            - stress_similarity: Stress pattern similarity (higher is better)
            - overall_prosody: Overall prosody score (higher is better)
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
            
            # Extract prosodic features
            ref_prosody = self._extract_prosodic_features(ref_audio)
            deg_prosody = self._extract_prosodic_features(deg_audio)
            
            # Compare prosodic features
            prosody_scores = self._compare_prosodic_features(ref_prosody, deg_prosody)
            
            if return_details:
                return prosody_scores
            else:
                return {'overall_prosody': prosody_scores['overall_prosody']}
                
        except Exception as e:
            self.logger.error(f"Error in prosody analysis: {e}")
            return self._get_nan_results()
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive prosodic features from audio."""
        features = {}
        
        # 1. Fundamental frequency (F0) analysis
        features.update(self._extract_f0_features(audio))
        
        # 2. Rhythm and temporal features
        features.update(self._extract_rhythm_features(audio))
        
        # 3. Stress and prominence features
        features.update(self._extract_stress_features(audio))
        
        # 4. Spectral dynamics (related to prosody)
        features.update(self._extract_spectral_dynamics(audio))
        
        return features
    
    def _extract_f0_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract F0 (fundamental frequency) features."""
        try:
            # Use librosa's piptrack for F0 estimation
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                threshold=0.1,
                fmin=self.f0_min,
                fmax=self.f0_max,
                n_fft=self.frame_length,
                win_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Extract most prominent F0 track
            f0_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                f0_track.append(pitch)
            
            f0_track = np.array(f0_track)
            
            # Clean F0 track (remove outliers and interpolate)
            f0_cleaned = self._clean_f0_track(f0_track)
            
            # F0 derivatives (important for prosody)
            f0_delta = np.diff(f0_cleaned)
            f0_delta_delta = np.diff(f0_delta)
            
            return {
                'f0': f0_cleaned,
                'f0_delta': f0_delta,
                'f0_delta_delta': f0_delta_delta,
                'f0_voiced_frames': f0_cleaned > 0
            }
            
        except Exception as e:
            self.logger.warning(f"F0 extraction failed: {e}")
            # Fallback to simpler method
            return self._extract_f0_fallback(audio)
    
    def _extract_f0_fallback(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback F0 extraction using autocorrelation."""
        try:
            # Frame-based F0 estimation
            frame_size = self.frame_length
            hop_size = self.hop_length
            
            f0_track = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                f0 = self._autocorrelation_f0(frame)
                f0_track.append(f0)
            
            f0_track = np.array(f0_track)
            f0_cleaned = self._clean_f0_track(f0_track)
            
            return {
                'f0': f0_cleaned,
                'f0_delta': np.diff(f0_cleaned),
                'f0_delta_delta': np.diff(np.diff(f0_cleaned)),
                'f0_voiced_frames': f0_cleaned > 0
            }
            
        except Exception as e:
            self.logger.error(f"Fallback F0 extraction failed: {e}")
            dummy_len = len(audio) // self.hop_length
            return {
                'f0': np.zeros(dummy_len),
                'f0_delta': np.zeros(dummy_len-1),
                'f0_delta_delta': np.zeros(dummy_len-2),
                'f0_voiced_frames': np.zeros(dummy_len, dtype=bool)
            }
    
    def _autocorrelation_f0(self, frame: np.ndarray) -> float:
        """Estimate F0 using autocorrelation method."""
        try:
            # Autocorrelation
            correlation = np.correlate(frame, frame, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Find peaks in autocorrelation
            min_period = int(self.sample_rate / self.f0_max)
            max_period = int(self.sample_rate / self.f0_min)
            
            if max_period >= len(correlation):
                return 0
            
            # Look for first strong peak
            search_range = correlation[min_period:max_period]
            if len(search_range) == 0:
                return 0
            
            peak_idx = np.argmax(search_range) + min_period
            
            # Convert to frequency
            f0 = self.sample_rate / peak_idx if peak_idx > 0 else 0
            
            # Validate F0 range
            if self.f0_min <= f0 <= self.f0_max:
                return f0
            else:
                return 0
                
        except:
            return 0
    
    def _clean_f0_track(self, f0_track: np.ndarray) -> np.ndarray:
        """Clean F0 track by removing outliers and interpolating gaps."""
        f0_cleaned = f0_track.copy()
        
        # Remove outliers (values too far from median)
        voiced_f0 = f0_cleaned[f0_cleaned > 0]
        if len(voiced_f0) > 0:
            median_f0 = np.median(voiced_f0)
            std_f0 = np.std(voiced_f0)
            
            # Remove values more than 3 std from median
            outlier_threshold = 3 * std_f0
            outliers = np.abs(f0_cleaned - median_f0) > outlier_threshold
            f0_cleaned[outliers] = 0
        
        # Simple interpolation of short gaps
        voiced_indices = np.where(f0_cleaned > 0)[0]
        if len(voiced_indices) > 1:
            for i in range(len(voiced_indices) - 1):
                start_idx = voiced_indices[i]
                end_idx = voiced_indices[i + 1]
                gap_length = end_idx - start_idx - 1
                
                # Interpolate gaps shorter than 5 frames
                if 0 < gap_length <= 5:
                    start_val = f0_cleaned[start_idx]
                    end_val = f0_cleaned[end_idx]
                    interpolated = np.linspace(start_val, end_val, gap_length + 2)[1:-1]
                    f0_cleaned[start_idx+1:end_idx] = interpolated
        
        return f0_cleaned
    
    def _extract_rhythm_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract rhythm and temporal features."""
        try:
            # 1. Onset detection for rhythm analysis
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units='frames'
            )
            
            # Convert to time
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=self.hop_length)
            
            # 2. Inter-onset intervals (IOI)
            if len(onset_times) > 1:
                ioi = np.diff(onset_times)
            else:
                ioi = np.array([])
            
            # 3. Tempo estimation
            tempo, beats = librosa.beat.beat_track(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            return {
                'onset_times': onset_times,
                'inter_onset_intervals': ioi,
                'tempo': tempo,
                'beat_times': librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length)
            }
            
        except Exception as e:
            self.logger.warning(f"Rhythm extraction failed: {e}")
            return {
                'onset_times': np.array([]),
                'inter_onset_intervals': np.array([]),
                'tempo': 120,  # Default tempo
                'beat_times': np.array([])
            }
    
    def _extract_stress_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract stress and prominence features."""
        try:
            # 1. Energy envelope for stress detection
            hop_length_energy = 512
            frame_length_energy = 2048
            
            # RMS energy
            rms_energy = librosa.feature.rms(
                y=audio,
                frame_length=frame_length_energy,
                hop_length=hop_length_energy
            )[0]
            
            # 2. Spectral centroid (brightness, correlated with stress)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                hop_length=hop_length_energy
            )[0]
            
            # 3. Zero crossing rate (articulation clarity)
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=frame_length_energy,
                hop_length=hop_length_energy
            )[0]
            
            # 4. Loudness contour (approximation)
            loudness = self._compute_loudness_contour(audio)
            
            return {
                'energy': rms_energy,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zcr,
                'loudness': loudness
            }
            
        except Exception as e:
            self.logger.warning(f"Stress extraction failed: {e}")
            dummy_len = len(audio) // 512
            return {
                'energy': np.zeros(dummy_len),
                'spectral_centroid': np.zeros(dummy_len),
                'zero_crossing_rate': np.zeros(dummy_len),
                'loudness': np.zeros(dummy_len)
            }
    
    def _compute_loudness_contour(self, audio: np.ndarray) -> np.ndarray:
        """Compute perceptual loudness contour."""
        try:
            # A-weighting for perceptual loudness
            # Simplified A-weighting approximation
            hop_length = 512
            
            # Frame-based processing
            n_frames = 1 + (len(audio) - 2048) // hop_length
            loudness = np.zeros(n_frames)
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + 2048
                if end <= len(audio):
                    frame = audio[start:end]
                    # Simplified loudness: RMS with frequency weighting
                    loudness[i] = np.sqrt(np.mean(frame**2))
            
            return loudness
            
        except:
            return np.zeros(len(audio) // 512)
    
    def _extract_spectral_dynamics(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral dynamics related to prosody."""
        try:
            # Spectral features that change with prosody
            hop_length = 512
            
            # Spectral rolloff (brightness changes)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                hop_length=hop_length
            )[0]
            
            # Spectral bandwidth (formant movement)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                hop_length=hop_length
            )[0]
            
            return {
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral dynamics extraction failed: {e}")
            dummy_len = len(audio) // 512
            return {
                'spectral_rolloff': np.zeros(dummy_len),
                'spectral_bandwidth': np.zeros(dummy_len)
            }
    
    def _compare_prosodic_features(self, ref_features: Dict, deg_features: Dict) -> Dict[str, float]:
        """Compare prosodic features between reference and degraded audio."""
        scores = {}
        
        # 1. F0 comparison
        f0_scores = self._compare_f0_features(ref_features, deg_features)
        scores.update(f0_scores)
        
        # 2. Rhythm comparison  
        rhythm_scores = self._compare_rhythm_features(ref_features, deg_features)
        scores.update(rhythm_scores)
        
        # 3. Stress comparison
        stress_scores = self._compare_stress_features(ref_features, deg_features)
        scores.update(stress_scores)
        
        # 4. Overall prosody score
        valid_scores = [v for v in scores.values() if not np.isnan(v)]
        if valid_scores:
            scores['overall_prosody'] = np.mean(valid_scores)
        else:
            scores['overall_prosody'] = 0.0
        
        return scores
    
    def _compare_f0_features(self, ref_features: Dict, deg_features: Dict) -> Dict[str, float]:
        """Compare F0 features."""
        try:
            ref_f0 = ref_features.get('f0', np.array([]))
            deg_f0 = deg_features.get('f0', np.array([]))
            
            if len(ref_f0) == 0 or len(deg_f0) == 0:
                return {'f0_rmse': float('nan'), 'f0_correlation': float('nan')}
            
            # Align lengths
            min_len = min(len(ref_f0), len(deg_f0))
            ref_f0 = ref_f0[:min_len]
            deg_f0 = deg_f0[:min_len]
            
            # Only compare voiced frames
            voiced_ref = ref_f0 > 0
            voiced_deg = deg_f0 > 0
            voiced_both = voiced_ref & voiced_deg
            
            if np.sum(voiced_both) < 5:  # Need minimum voiced frames
                return {'f0_rmse': float('nan'), 'f0_correlation': float('nan')}
            
            ref_voiced = ref_f0[voiced_both]
            deg_voiced = deg_f0[voiced_both]
            
            # RMSE (convert to semitones for perceptual relevance)
            ref_semitones = 12 * np.log2(ref_voiced / 440)  # A4 = 440Hz reference
            deg_semitones = 12 * np.log2(deg_voiced / 440)
            f0_rmse = np.sqrt(np.mean((ref_semitones - deg_semitones)**2))
            
            # Correlation
            f0_corr, _ = pearsonr(ref_voiced, deg_voiced)
            if np.isnan(f0_corr):
                f0_corr = 0.0
            
            return {
                'f0_rmse': float(f0_rmse),
                'f0_correlation': float(abs(f0_corr))  # Take absolute for similarity
            }
            
        except Exception as e:
            self.logger.error(f"F0 comparison failed: {e}")
            return {'f0_rmse': float('nan'), 'f0_correlation': float('nan')}
    
    def _compare_rhythm_features(self, ref_features: Dict, deg_features: Dict) -> Dict[str, float]:
        """Compare rhythm features.""" 
        try:
            # Tempo similarity
            ref_tempo = ref_features.get('tempo', 120)
            deg_tempo = deg_features.get('tempo', 120)
            
            tempo_similarity = 1 - abs(ref_tempo - deg_tempo) / max(ref_tempo, deg_tempo)
            tempo_similarity = max(0, tempo_similarity)
            
            # Inter-onset interval similarity
            ref_ioi = ref_features.get('inter_onset_intervals', np.array([]))
            deg_ioi = deg_features.get('inter_onset_intervals', np.array([]))
            
            if len(ref_ioi) > 0 and len(deg_ioi) > 0:
                # Compare IOI distributions
                ref_ioi_mean = np.mean(ref_ioi)
                deg_ioi_mean = np.mean(deg_ioi)
                ref_ioi_std = np.std(ref_ioi)
                deg_ioi_std = np.std(deg_ioi)
                
                mean_similarity = 1 - abs(ref_ioi_mean - deg_ioi_mean) / max(ref_ioi_mean, deg_ioi_mean)
                std_similarity = 1 - abs(ref_ioi_std - deg_ioi_std) / max(ref_ioi_std, deg_ioi_std)
                
                rhythm_similarity = (mean_similarity + std_similarity) / 2
                rhythm_similarity = max(0, rhythm_similarity)
            else:
                rhythm_similarity = 0.5  # Neutral score if no onsets
            
            return {
                'tempo_similarity': float(tempo_similarity),
                'rhythm_similarity': float(rhythm_similarity)
            }
            
        except Exception as e:
            self.logger.error(f"Rhythm comparison failed: {e}")
            return {
                'tempo_similarity': float('nan'),
                'rhythm_similarity': float('nan')
            }
    
    def _compare_stress_features(self, ref_features: Dict, deg_features: Dict) -> Dict[str, float]:
        """Compare stress and prominence features."""
        try:
            scores = []
            
            # Energy contour similarity
            ref_energy = ref_features.get('energy', np.array([]))
            deg_energy = deg_features.get('energy', np.array([]))
            
            if len(ref_energy) > 0 and len(deg_energy) > 0:
                energy_corr = self._compute_contour_correlation(ref_energy, deg_energy)
                scores.append(energy_corr)
            
            # Spectral centroid similarity
            ref_centroid = ref_features.get('spectral_centroid', np.array([]))
            deg_centroid = deg_features.get('spectral_centroid', np.array([]))
            
            if len(ref_centroid) > 0 and len(deg_centroid) > 0:
                centroid_corr = self._compute_contour_correlation(ref_centroid, deg_centroid)
                scores.append(centroid_corr)
            
            # Overall stress similarity
            if scores:
                stress_similarity = np.mean(scores)
            else:
                stress_similarity = 0.0
            
            return {'stress_similarity': float(stress_similarity)}
            
        except Exception as e:
            self.logger.error(f"Stress comparison failed: {e}")
            return {'stress_similarity': float('nan')}
    
    def _compute_contour_correlation(self, ref_contour: np.ndarray, deg_contour: np.ndarray) -> float:
        """Compute correlation between feature contours."""
        try:
            # Align lengths
            min_len = min(len(ref_contour), len(deg_contour))
            if min_len < 5:
                return 0.0
            
            ref_aligned = ref_contour[:min_len]
            deg_aligned = deg_contour[:min_len]
            
            # Remove DC component (focus on dynamic changes)
            ref_centered = ref_aligned - np.mean(ref_aligned)
            deg_centered = deg_aligned - np.mean(deg_aligned)
            
            # Compute correlation
            correlation, _ = pearsonr(ref_centered, deg_centered)
            
            if np.isnan(correlation):
                return 0.0
            
            return abs(correlation)  # Return absolute correlation for similarity
            
        except:
            return 0.0
    
    def _get_nan_results(self) -> Dict[str, float]:
        """Return NaN results for error cases."""
        return {
            'f0_rmse': float('nan'),
            'f0_correlation': float('nan'),
            'rhythm_similarity': float('nan'),
            'tempo_similarity': float('nan'),
            'stress_similarity': float('nan'),
            'overall_prosody': float('nan')
        }


def compute_f0_rmse(reference_path: str, degraded_path: str) -> float:
    """
    Legacy interface for F0 RMSE computation.
    
    Args:
        reference_path: Path to reference audio
        degraded_path: Path to degraded audio
        
    Returns:
        F0 RMSE in semitones (lower is better)
    """
    analyzer = ProsodyAnalyzer()
    result = analyzer.analyze(reference_path, degraded_path)
    return result['f0_rmse']


if __name__ == "__main__":
    # Example usage
    analyzer = ProsodyAnalyzer()
    
    # Test on sample files
    ref_file = "reference.wav"
    deg_file = "degraded.wav"
    
    if Path(ref_file).exists() and Path(deg_file).exists():
        scores = analyzer.analyze(ref_file, deg_file, return_details=True)
        print(f"Prosody analysis scores: {scores}")