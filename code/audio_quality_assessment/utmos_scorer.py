"""
UTMOS scorer wrapper for batch evaluation.

Uses the UTMOS implementation vendored in this repository under either:
- WavTokenizer/metrics/UTMOS.py
- Languagecodec/metrics/UTMOS.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio


class UTMOSScorer:
    """Compute UTMOS scores for WAV files."""

    def __init__(self, device: Optional[str] = None, max_duration_seconds: float = 10.0):
        self.logger = logging.getLogger(__name__)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_duration_seconds = max_duration_seconds
        self.model = self._load_utmos_model()
        self.logger.info("UTMOS initialized on %s", self.device)

    def _load_module_from_file(self, module_path: Path):
        spec = importlib.util.spec_from_file_location("local_utmos_module", str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_utmos_model(self):
        project_root = Path(__file__).resolve().parents[2]
        candidates = [
            project_root / "WavTokenizer" / "metrics" / "UTMOS.py",
            project_root / "Languagecodec" / "metrics" / "UTMOS.py",
        ]

        last_error = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                module = self._load_module_from_file(candidate)
                if not hasattr(module, "UTMOSScore"):
                    raise AttributeError(f"UTMOSScore not found in {candidate}")
                self.logger.info("Using UTMOS implementation from %s", candidate)
                return module.UTMOSScore(device=str(self.device))
            except Exception as exc:
                last_error = exc
                self.logger.warning("Failed loading UTMOS from %s: %s", candidate, exc)

        raise RuntimeError(
            "Could not initialize UTMOS model. Ensure dependencies are installed "
            "(fairseq, pytorch_lightning, requests) and checkpoint download is allowed."
        ) from last_error

    def score(self, audio_path: str) -> float:
        wav, sr = torchaudio.load(audio_path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.to(self.device)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)

        if self.max_duration_seconds and self.max_duration_seconds > 0:
            max_samples = int(16000 * self.max_duration_seconds)
            if wav.shape[-1] > max_samples:
                wav = wav[..., :max_samples]

        with torch.no_grad():
            try:
                # Expected shape for UTMOS implementation: [B, 1, T]
                score_tensor = self.model.score(wav.unsqueeze(1))
                score = float(score_tensor[0].item())
            except RuntimeError as exc:
                # Retry with a shorter clip if memory spikes on very long/complex utterances
                fallback_samples = 16000 * 5
                if wav.shape[-1] > fallback_samples:
                    wav_short = wav[..., :fallback_samples]
                    score_tensor = self.model.score(wav_short.unsqueeze(1))
                    score = float(score_tensor[0].item())
                    self.logger.warning("UTMOS retry succeeded on shorter clip for %s: %s", audio_path, exc)
                else:
                    raise

        return score

    def score_batch(self, audio_dir: str) -> Dict:
        wav_files = sorted([p for p in Path(audio_dir).glob("*.wav") if p.is_file()])

        results: List[Dict] = []
        scores: List[float] = []

        for wav_path in wav_files:
            try:
                score = self.score(str(wav_path))
                results.append({"filename": wav_path.name, "utmos": score})
                scores.append(score)
            except Exception as exc:
                results.append({"filename": wav_path.name, "utmos": float("nan"), "error": str(exc)})

        valid_scores = [s for s in scores if not np.isnan(s)]

        if valid_scores:
            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores))
            min_score = float(np.min(valid_scores))
            max_score = float(np.max(valid_scores))
        else:
            mean_score = float("nan")
            std_score = float("nan")
            min_score = float("nan")
            max_score = float("nan")

        return {
            "metric": "utmos",
            "audio_dir": os.path.abspath(audio_dir),
            "num_files": len(wav_files),
            "num_scored_files": len(valid_scores),
            "utmos": mean_score,
            "utmos_std": std_score,
            "utmos_min": min_score,
            "utmos_max": max_score,
            "per_file": results,
        }
