"""F0 RMSE scorer for intrusive codec evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from code.metrics.prosody_f0_rmse import compute_f0_rmse


class F0Scorer:
    """Compute F0 RMSE over matched reference/degraded file pairs."""

    def score_batch(self, ref_dir: str, deg_dir: str, num_samples: int = -1) -> Dict:
        ref_map = {p.name: p for p in Path(ref_dir).glob("*.wav")}
        deg_files = sorted([p for p in Path(deg_dir).glob("*.wav") if p.is_file()])
        if num_samples is not None and num_samples > 0:
            deg_files = deg_files[:num_samples]

        per_file: List[Dict] = []
        f0_values: List[float] = []
        matched = 0

        for deg_path in deg_files:
            ref_path = ref_map.get(deg_path.name)
            if ref_path is None:
                per_file.append({
                    "filename": deg_path.name,
                    "f0_rmse": float("nan"),
                    "error": "reference file not found",
                })
                continue

            matched += 1
            try:
                value = float(compute_f0_rmse(str(ref_path), str(deg_path)))
                per_file.append({"filename": deg_path.name, "f0_rmse": value})
                if not np.isnan(value):
                    f0_values.append(value)
            except Exception as exc:
                per_file.append({
                    "filename": deg_path.name,
                    "f0_rmse": float("nan"),
                    "error": str(exc),
                })

        if f0_values:
            agg = {
                "f0_rmse": float(np.mean(f0_values)),
                "f0_rmse_std": float(np.std(f0_values)),
                "f0_rmse_min": float(np.min(f0_values)),
                "f0_rmse_max": float(np.max(f0_values)),
            }
        else:
            agg = {
                "f0_rmse": float("nan"),
                "f0_rmse_std": float("nan"),
                "f0_rmse_min": float("nan"),
                "f0_rmse_max": float("nan"),
            }

        return {
            "metric": "f0_rmse",
            "reference_dir": os.path.abspath(ref_dir),
            "degraded_dir": os.path.abspath(deg_dir),
            "num_ref_files": len(ref_map),
            "num_deg_files": len(deg_files),
            "num_matched_files": matched,
            "num_scored_files": len(f0_values),
            **agg,
            "per_file": per_file,
        }
