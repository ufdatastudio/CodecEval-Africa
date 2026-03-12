"""
PESQ (Perceptual Evaluation of Speech Quality) Implementation

This module provides PESQ scoring for speech quality evaluation.
PESQ is an ITU-T standard (P.862) for objective speech quality assessment.

PESQ compares a reference (original) audio with a degraded audio and provides
a score from -0.5 to 4.5, where higher scores indicate better quality.
"""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from pesq import pesq
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PESQScorer:
    """
    PESQ (Perceptual Evaluation of Speech Quality) scorer.
    
    Evaluates speech quality by comparing reference and degraded audio files.
    Supports both narrow-band (NB, 8kHz) and wide-band (WB, 16kHz) modes.
    """
    
    def __init__(self, mode: str = 'wb'):
        """
        Initialize PESQ scorer.
        
        Args:
            mode: PESQ mode - 'nb' (narrow-band, 8kHz) or 'wb' (wide-band, 16kHz)
        """
        self.mode = mode
        self.wb_sr = 16000
        self.nb_sr = 8000
        logger.info(f"PESQ scorer initialized (mode: {mode})")
    
    def score(self, ref_path: str, deg_path: str, return_details: bool = False) -> Dict:
        """
        Compute PESQ score between reference and degraded audio.
        
        Args:
            ref_path: Path to reference (original) audio file
            deg_path: Path to degraded (codec output) audio file
            return_details: If True, return both NB and WB scores
            
        Returns:
            Dictionary with PESQ scores:
                - 'pesq': Main PESQ score (based on mode)
                - 'nb_pesq': Narrow-band PESQ (if return_details=True)
                - 'wb_pesq': Wide-band PESQ (if return_details=True)
        """
        try:
            # Load audio files
            ref_rate, ref = wavfile.read(ref_path)
            deg_rate, deg = wavfile.read(deg_path)
            
            # Convert to float32 if needed
            if ref.dtype == np.int16:
                ref = ref.astype(np.float32) / 32768.0
            if deg.dtype == np.int16:
                deg = deg.astype(np.float32) / 32768.0
            
            # Prepare WB (16kHz)
            ref_wb = ref if ref_rate == self.wb_sr else signal.resample_poly(ref, self.wb_sr, ref_rate)
            deg_wb = deg if deg_rate == self.wb_sr else signal.resample_poly(deg, self.wb_sr, deg_rate)
            min_len_wb = min(len(ref_wb), len(deg_wb))
            ref_wb = ref_wb[:min_len_wb]
            deg_wb = deg_wb[:min_len_wb]

            # Prepare NB (8kHz)
            ref_nb = ref if ref_rate == self.nb_sr else signal.resample_poly(ref, self.nb_sr, ref_rate)
            deg_nb = deg if deg_rate == self.nb_sr else signal.resample_poly(deg, self.nb_sr, deg_rate)
            min_len_nb = min(len(ref_nb), len(deg_nb))
            ref_nb = ref_nb[:min_len_nb]
            deg_nb = deg_nb[:min_len_nb]
            
            # Compute PESQ scores
            result = {}
            
            if return_details or self.mode == 'nb':
                nb_score = pesq(self.nb_sr, ref_nb, deg_nb, 'nb')
                result['nb_pesq'] = float(nb_score)
            
            if return_details or self.mode == 'wb':
                wb_score = pesq(self.wb_sr, ref_wb, deg_wb, 'wb')
                result['wb_pesq'] = float(wb_score)
            
            # Set main score based on mode
            if self.mode == 'nb':
                result['pesq'] = result['nb_pesq']
            else:
                result['pesq'] = result['wb_pesq']
            
            return result
            
        except Exception as e:
            logger.error(f"PESQ scoring failed: {e}")
            raise

    def _score_subprocess(self, ref_path: str, deg_path: str, mode: str) -> float:
        """Compute PESQ in isolated subprocess to avoid native crashes taking down parent process."""
        score_script = f"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from pesq import pesq

ref_path = r'''{ref_path}'''
deg_path = r'''{deg_path}'''
mode = '{mode}'
target_sr = 16000 if mode == 'wb' else 8000

ref_rate, ref = wavfile.read(ref_path)
deg_rate, deg = wavfile.read(deg_path)

if ref.dtype == np.int16:
    ref = ref.astype(np.float32) / 32768.0
else:
    ref = ref.astype(np.float32)
if deg.dtype == np.int16:
    deg = deg.astype(np.float32) / 32768.0
else:
    deg = deg.astype(np.float32)

if ref_rate != target_sr:
    ref = resample_poly(ref, target_sr, ref_rate).astype(np.float32)
if deg_rate != target_sr:
    deg = resample_poly(deg, target_sr, deg_rate).astype(np.float32)

min_len = min(len(ref), len(deg))
ref = np.ascontiguousarray(ref[:min_len])
deg = np.ascontiguousarray(deg[:min_len])

score = pesq(target_sr, ref, deg, mode)
print(float(score))
"""
        proc = subprocess.run(
            [sys.executable, "-c", score_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"subprocess failed ({proc.returncode})")
        return float(proc.stdout.strip().splitlines()[-1])
    
    def score_batch(self, ref_dir: str, deg_dir: str) -> Dict:
        """
        Compute average PESQ scores for all matching files in directories.
        
        Args:
            ref_dir: Directory containing reference audio files
            deg_dir: Directory containing degraded audio files
            
        Returns:
            Dictionary with average PESQ scores
        """
        from glob import glob
        from tqdm import tqdm
        
        ref_dir = Path(ref_dir)
        deg_dir = Path(deg_dir)
        
        # Find all degraded files
        deg_files = sorted(glob(str(deg_dir / "*.wav")))
        
        if len(deg_files) == 0:
            raise RuntimeError(f"No .wav files found in {deg_dir}")
        
        scores_main = []
        
        for deg_path in tqdm(deg_files, desc="Computing PESQ"):
            # Find matching reference file
            ref_path = ref_dir / Path(deg_path).name
            
            if not ref_path.exists():
                logger.warning(f"Reference file not found: {ref_path}")
                continue
            
            try:
                score_val = self._score_subprocess(str(ref_path), deg_path, self.mode)
                scores_main.append(score_val)
            except Exception as e:
                logger.error(f"Failed to score {deg_path}: {e}")
                continue
        
        if len(scores_main) == 0:
            raise RuntimeError("No files were successfully scored")
        
        result = {
            'pesq': float(np.mean(scores_main)),
            'num_files': len(scores_main)
        }

        if self.mode == 'wb':
            result['wb_pesq'] = result['pesq']
            logger.info(f"Average WB PESQ: {result['wb_pesq']:.3f}")
        else:
            result['nb_pesq'] = result['pesq']
            logger.info(f"Average NB PESQ: {result['nb_pesq']:.3f}")
        
        return result
