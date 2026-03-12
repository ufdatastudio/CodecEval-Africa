"""
STOI (Short-Time Objective Intelligibility) Implementation

This module provides STOI scoring for speech intelligibility evaluation.
STOI is a measure that correlates highly with human intelligibility judgments.

STOI compares a reference (clean) signal with a degraded signal and produces
a score between 0 and 1, where higher scores indicate better intelligibility.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
from pystoi import stoi
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class STOIScorer:
    """
    STOI (Short-Time Objective Intelligibility) scorer.
    
    Evaluates speech intelligibility by comparing reference and degraded audio.
    Supports both standard STOI and extended STOI.
    """
    
    def __init__(self, extended: bool = False):
        """
        Initialize STOI scorer.
        
        Args:
            extended: If True, use extended STOI which is more suitable for 
                     intelligibility prediction with noise suppression
        """
        self.extended = extended
        logger.info(f"STOI scorer initialized (extended: {extended})")
    
    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate using high-quality polyphase filtering.
        
        Args:
            audio: Input audio signal
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio signal
        """
        if orig_sr == target_sr:
            return audio
        
        # Use polyphase resampling for better quality
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        
        resampled = signal.resample_poly(audio, up, down)
        return resampled
    
    def score(self, ref_path: str, deg_path: str, return_details: bool = False) -> Dict:
        """
        Compute STOI score between reference and degraded audio.
        
        Args:
            ref_path: Path to reference (clean) audio file
            deg_path: Path to degraded (codec output) audio file
            return_details: If True, return additional information
            
        Returns:
            Dictionary with STOI score:
                - 'stoi': STOI intelligibility score (0-1)
                - 'sample_rate': Sample rate used (if return_details=True)
                - 'duration': Audio duration in seconds (if return_details=True)
        """
        try:
            # Load audio files
            ref_rate, ref = wavfile.read(ref_path)
            deg_rate, deg = wavfile.read(deg_path)
            
            # Resample to matching sample rate if needed
            if ref_rate != deg_rate:
                logger.debug(f"Resampling: ref={ref_rate}Hz, deg={deg_rate}Hz -> target={ref_rate}Hz")
                # Resample degraded to match reference
                deg = self.resample_audio(deg, deg_rate, ref_rate)
                deg_rate = ref_rate
            
            target_sr = ref_rate
            
            # Convert to float32 if needed
            if ref.dtype == np.int16:
                ref = ref.astype(np.float32) / 32768.0
            if deg.dtype == np.int16:
                deg = deg.astype(np.float32) / 32768.0
            
            # Align lengths (use minimum length)
            min_len = min(len(ref), len(deg))
            ref = ref[:min_len]
            deg = deg[:min_len]
            
            # Compute STOI score
            stoi_score = stoi(ref, deg, target_sr, extended=self.extended)
            
            result = {
                'stoi': float(stoi_score)
            }
            
            if return_details:
                result['sample_rate'] = target_sr
                result['duration'] = min_len / target_sr
                result['extended'] = self.extended
            
            return result
            
        except Exception as e:
            logger.error(f"STOI scoring failed: {e}")
            raise
    
    def score_batch(self, ref_dir: str, deg_dir: str) -> Dict:
        """
        Compute average STOI scores for all matching files in directories.
        
        Args:
            ref_dir: Directory containing reference audio files
            deg_dir: Directory containing degraded audio files
            
        Returns:
            Dictionary with average STOI score
        """
        from glob import glob
        from tqdm import tqdm
        
        ref_dir = Path(ref_dir)
        deg_dir = Path(deg_dir)
        
        # Find all degraded files
        deg_files = sorted(glob(str(deg_dir / "*.wav")))
        
        if len(deg_files) == 0:
            raise RuntimeError(f"No .wav files found in {deg_dir}")
        
        stoi_scores = []
        
        for deg_path in tqdm(deg_files, desc="Computing STOI"):
            # Find matching reference file
            ref_path = ref_dir / Path(deg_path).name
            
            if not ref_path.exists():
                logger.warning(f"Reference file not found: {ref_path}")
                continue
            
            try:
                scores = self.score(str(ref_path), deg_path)
                stoi_scores.append(scores['stoi'])
            except Exception as e:
                logger.error(f"Failed to score {deg_path}: {e}")
                continue
        
        if len(stoi_scores) == 0:
            raise RuntimeError("No files were successfully scored")
        
        result = {
            'stoi': float(np.mean(stoi_scores)),
            'stoi_std': float(np.std(stoi_scores)),
            'num_files': len(stoi_scores)
        }
        
        logger.info(f"Average STOI: {result['stoi']:.4f} (±{result['stoi_std']:.4f})")
        
        return result
