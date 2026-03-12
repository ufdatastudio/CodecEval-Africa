"""Official NISQA scorer wrapper (no fallback model)."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

class NISQAScorer:
    """
    NISQA (Non-Intrusive Speech Quality Assessment) scorer.
    
    Uses the official NISQA model to predict MOS (Mean Opinion Score) 
    for speech quality assessment.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        ms_channel: Optional[int] = None,
        ms_max_segments: Optional[int] = None,
        allow_fallback: bool = False,
    ):
        """
        Initialize NISQA scorer.
        
        Args:
            model_path: Path to official NISQA model weights.
            device: Device to run model on ('cuda' or 'cpu'). Auto-detected if None.
            batch_size: Batch size for official NISQA inference.
            num_workers: Number of workers for official NISQA data loader.
            ms_channel: Optional channel index for stereo files.
            ms_max_segments: Max mel-segment bins for long files (NISQA arg `ms_max_segments`).
            allow_fallback: Deprecated. Must be False in production.
        """
        self.logger = logging.getLogger(__name__)

        if allow_fallback:
            raise ValueError("Fallback mode is disabled. Use official NISQA inference only.")

        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = str(device)

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.ms_channel = ms_channel
        if ms_max_segments is None:
            ms_max_segments = int(os.environ.get("NISQA_MS_MAX_SEGMENTS", "50000"))
        self.ms_max_segments = int(ms_max_segments)

        self.repo_root = Path(__file__).parent.parent.parent / "NISQA"
        if not self.repo_root.exists():
            raise FileNotFoundError(f"NISQA repository not found at {self.repo_root}")

        self.model_path = Path(model_path) if model_path else self.repo_root / "weights" / "nisqa.tar"
        if not self.model_path.exists():
            raise FileNotFoundError(f"NISQA model not found at {self.model_path}")

        self.logger.info(f"NISQA using device: {self.device}")

    def _import_nisqa_model(self):
        import sys

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        from nisqa.NISQA_model import nisqaModel

        return nisqaModel

    @staticmethod
    def _row_to_scores(row: pd.Series) -> Dict[str, float]:
        def _value(column: str) -> float:
            val = row.get(column)
            if pd.isna(val):
                return float("nan")
            return float(val)

        return {
            "mos": _value("mos_pred"),
            "noi": _value("noi_pred"),
            "dis": _value("dis_pred"),
            "col": _value("col_pred"),
            "loud": _value("loud_pred"),
        }

    def _predict_dataframe(self, args: Dict[str, Any]) -> pd.DataFrame:
        nisqaModel = self._import_nisqa_model()
        predictor = nisqaModel(args)
        return predictor.predict()
    
    def score(self, audio_path: str, return_details: bool = False) -> Dict[str, float]:
        """
        Compute NISQA score for audio file.
        
        Args:
            audio_path: Path to audio file
            return_details: If True, returns additional metrics
            
        Returns:
            Dictionary containing NISQA scores:
            - mos: Overall Mean Opinion Score (1-5, higher is better)
            - noi: Noisiness score  
            - dis: Distortion score
            - col: Coloration score
            - loud: Loudness score
        """
        audio_path = str(Path(audio_path).resolve())
        args = {
            "mode": "predict_file",
            "pretrained_model": str(self.model_path.resolve()),
            "deg": audio_path,
            "output_dir": "",
            "tr_bs_val": self.batch_size,
            "tr_num_workers": self.num_workers,
            "tr_device": self.device,
            "ms_channel": self.ms_channel,
            "ms_max_segments": self.ms_max_segments,
        }

        try:
            df = self._predict_dataframe(args)
        except Exception as error:
            message = str(error)
            if "n_wins" in message and "max_length" in message:
                import re

                match = re.search(r"n_wins\s+(\d+)\s+>\s+max_length\s+(\d+)", message)
                if match:
                    required = int(match.group(1)) + 256
                    if required > self.ms_max_segments:
                        args["ms_max_segments"] = required
                        self.logger.warning(
                            "Retrying NISQA score with larger ms_max_segments=%d for %s",
                            required,
                            audio_path,
                        )
                        df = self._predict_dataframe(args)
                    else:
                        raise
                else:
                    raise
            else:
                raise
        if df.empty:
            raise RuntimeError(f"Official NISQA returned no prediction for {audio_path}")

        scores = self._row_to_scores(df.iloc[0])
        return scores if return_details else {"mos": scores["mos"]}
    
    def batch_score(self, audio_paths: List[str], return_details: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compute NISQA scores for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            return_details: If True, returns additional metrics
            
        Returns:
            Dictionary mapping file paths to NISQA scores
        """
        resolved_paths = [str(Path(path).resolve()) for path in audio_paths]
        if not resolved_paths:
            return {}

        with tempfile.TemporaryDirectory(prefix="nisqa_predict_") as tmp_dir:
            csv_path = Path(tmp_dir) / "files.csv"
            pd.DataFrame({"deg": resolved_paths}).to_csv(csv_path, index=False)

            args = {
                "mode": "predict_csv",
                "pretrained_model": str(self.model_path.resolve()),
                "data_dir": str(Path(tmp_dir).resolve()),
                "csv_file": csv_path.name,
                "csv_deg": "deg",
                "output_dir": "",
                "tr_bs_val": self.batch_size,
                "tr_num_workers": self.num_workers,
                "tr_device": self.device,
                "ms_channel": self.ms_channel,
                "ms_max_segments": self.ms_max_segments,
            }

            df = self._predict_dataframe(args)

        if len(df) != len(resolved_paths):
            self.logger.warning(
                "NISQA returned %d predictions for %d inputs", len(df), len(resolved_paths)
            )

        results = {}
        by_deg = {
            str(Path(row["deg"]).resolve()): self._row_to_scores(row)
            for _, row in df.iterrows()
        }
        for path in resolved_paths:
            values = by_deg.get(
                path,
                {
                    "mos": float("nan"),
                    "noi": float("nan"),
                    "dis": float("nan"),
                    "col": float("nan"),
                    "loud": float("nan"),
                },
            )
            results[path] = values if return_details else {"mos": values["mos"]}

        return results


def score(wav_path: str, sr: Optional[int] = None) -> float:
    """
    Legacy interface for backward compatibility.
    
    Args:
        wav_path: Path to audio file
        sr: Sample rate (ignored, auto-detected)
        
    Returns:
        NISQA MOS score (1-5, higher is better)
    """
    scorer = NISQAScorer()
    result = scorer.score(wav_path)
    return result['mos']


if __name__ == "__main__":
    # Example usage
    scorer = NISQAScorer()
    
    # Test on a sample file
    test_file = "test_audio.wav"
    if Path(test_file).exists():
        score = scorer.score(test_file, return_details=True)
        print(f"NISQA scores: {score}")