#!/usr/bin/env python3
"""Summarize ASV JSON results into codec leaderboards.

Outputs:
- asv_variant_table.csv
- asv_codec_leaderboard_by_dataset.csv
- asv_codec_leaderboard_overall.csv
- asv_summary.md
"""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


DATASETS = ["afrispeech_dialog", "afrinames", "afrispeech_multilingual"]
CODECS = ["DAC", "Encodec", "FocalCodec", "LanguageCodec", "SemantiCodec", "UniCodec", "WavTokenizer"]


def _parse_dataset_and_suffix(stem: str):
    for dataset in DATASETS:
        prefix = f"{dataset}_"
        if stem.startswith(prefix) and stem.endswith("_asv"):
            return dataset, stem[len(prefix):-4]
    return None, None


def _parse_codec_variant(suffix: str):
    for codec in CODECS:
        prefix = f"{codec}_"
        if suffix.startswith(prefix):
            return codec, suffix[len(prefix):]
    return "UNKNOWN", suffix


def _safe(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _variant_flag(row):
    eer = row.get("eer_percent")
    eer_ci_u = row.get("eer_percent_ci95_upper")
    overlap = row.get("has_overlap")

    if eer is None:
        return "invalid"
    if overlap is True:
        return "overlap_detected"
    if eer == 0.0:
        if eer_ci_u is None:
            return "zero_eer_no_ci"
        if eer_ci_u > 0.5:
            return "zero_eer_fragile"
        return "zero_eer_robust"
    return "nonzero_eer"


def _mean(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def _std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return pstdev(vals)


def _write_csv(path: Path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize ASV results and create leaderboards")
    parser.add_argument("--input-dir", default="results/ASV/metrics", help="Directory containing *_asv.json files")
    parser.add_argument("--output-dir", default="results/ASV/analysis", help="Directory for summary outputs")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = []
    pattern = str(in_dir / "*_asv.json")
    for path in sorted(glob.glob(pattern)):
        p = Path(path)
        stem = p.stem
        dataset, suffix = _parse_dataset_and_suffix(stem)
        if dataset is None:
            continue
        codec, variant = _parse_codec_variant(suffix)

        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        metrics = data.get("metrics", {})
        diagnostics = data.get("diagnostics", {})
        eer_ci95 = metrics.get("eer_percent_ci95")

        row = {
            "dataset": dataset,
            "codec": codec,
            "variant": variant,
            "file": str(p),
            "eer_percent": _safe(metrics.get("eer_percent")),
            "min_dcf": _safe(metrics.get("min_dcf")),
            "eer_percent_ci95_lower": _safe(eer_ci95[0]) if isinstance(eer_ci95, list) and len(eer_ci95) == 2 else None,
            "eer_percent_ci95_upper": _safe(eer_ci95[1]) if isinstance(eer_ci95, list) and len(eer_ci95) == 2 else None,
            "has_overlap": diagnostics.get("has_overlap"),
            "margin": _safe(diagnostics.get("separation_margin_targetMin_minus_impostorMax")),
            "num_trials": data.get("num_trials"),
            "model_type": data.get("model_type"),
        }
        row["quality_flag"] = _variant_flag(row)
        variant_rows.append(row)

    variant_headers = [
        "dataset", "codec", "variant", "eer_percent", "min_dcf",
        "eer_percent_ci95_lower", "eer_percent_ci95_upper",
        "has_overlap", "margin", "num_trials", "model_type", "quality_flag", "file",
    ]
    _write_csv(out_dir / "asv_variant_table.csv", variant_rows, variant_headers)

    grouped = defaultdict(list)
    for row in variant_rows:
        grouped[(row["dataset"], row["codec"])].append(row)

    by_dataset_rows = []
    for (dataset, codec), rows in sorted(grouped.items()):
        eers = [r["eer_percent"] for r in rows]
        dcfs = [r["min_dcf"] for r in rows]
        ci_us = [r["eer_percent_ci95_upper"] for r in rows]

        by_dataset_rows.append({
            "dataset": dataset,
            "codec": codec,
            "num_variants": len(rows),
            "mean_eer_percent": _mean(eers),
            "std_eer_percent": _std(eers),
            "mean_min_dcf": _mean(dcfs),
            "std_min_dcf": _std(dcfs),
            "mean_eer_ci95_upper": _mean(ci_us),
            "num_overlap_variants": sum(1 for r in rows if r.get("has_overlap") is True),
            "num_zero_eer_variants": sum(1 for r in rows if r.get("eer_percent") == 0.0),
            "num_fragile_zero_variants": sum(1 for r in rows if r.get("quality_flag") == "zero_eer_fragile"),
        })

    by_dataset_rows.sort(key=lambda r: (r["dataset"], r["mean_eer_percent"] if r["mean_eer_percent"] is not None else 1e9))
    _write_csv(
        out_dir / "asv_codec_leaderboard_by_dataset.csv",
        by_dataset_rows,
        [
            "dataset", "codec", "num_variants",
            "mean_eer_percent", "std_eer_percent",
            "mean_min_dcf", "std_min_dcf",
            "mean_eer_ci95_upper",
            "num_overlap_variants", "num_zero_eer_variants", "num_fragile_zero_variants",
        ],
    )

    overall_grouped = defaultdict(list)
    for row in by_dataset_rows:
        overall_grouped[row["codec"]].append(row)

    overall_rows = []
    for codec, rows in sorted(overall_grouped.items()):
        overall_rows.append({
            "codec": codec,
            "datasets_covered": len(rows),
            "overall_mean_eer_percent": _mean([r["mean_eer_percent"] for r in rows]),
            "overall_mean_min_dcf": _mean([r["mean_min_dcf"] for r in rows]),
            "overall_mean_eer_ci95_upper": _mean([r["mean_eer_ci95_upper"] for r in rows]),
            "total_overlap_variants": sum(r["num_overlap_variants"] for r in rows),
            "total_zero_eer_variants": sum(r["num_zero_eer_variants"] for r in rows),
            "total_fragile_zero_variants": sum(r["num_fragile_zero_variants"] for r in rows),
        })

    overall_rows.sort(key=lambda r: r["overall_mean_eer_percent"] if r["overall_mean_eer_percent"] is not None else 1e9)
    _write_csv(
        out_dir / "asv_codec_leaderboard_overall.csv",
        overall_rows,
        [
            "codec", "datasets_covered",
            "overall_mean_eer_percent", "overall_mean_min_dcf", "overall_mean_eer_ci95_upper",
            "total_overlap_variants", "total_zero_eer_variants", "total_fragile_zero_variants",
        ],
    )

    md_path = out_dir / "asv_summary.md"
    with md_path.open("w") as f:
        f.write("# ASV Summary\n\n")
        f.write(f"Variants processed: {len(variant_rows)}\n\n")

        f.write("## Overall codec ranking (lower is better)\n\n")
        f.write("| Rank | Codec | Mean EER% | Mean minDCF | Mean EER CI upper | Overlap variants |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for idx, row in enumerate(overall_rows, 1):
            f.write(
                f"| {idx} | {row['codec']} | "
                f"{(row['overall_mean_eer_percent'] if row['overall_mean_eer_percent'] is not None else float('nan')):.3f} | "
                f"{(row['overall_mean_min_dcf'] if row['overall_mean_min_dcf'] is not None else float('nan')):.4f} | "
                f"{(row['overall_mean_eer_ci95_upper'] if row['overall_mean_eer_ci95_upper'] is not None else float('nan')):.3f} | "
                f"{row['total_overlap_variants']} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- `quality_flag=zero_eer_fragile` means EER is zero but CI upper bound is not tight.\n")
        f.write("- `has_overlap=true` indicates target/impostor score overlap was observed in that variant.\n")

    print(f"Wrote: {out_dir / 'asv_variant_table.csv'}")
    print(f"Wrote: {out_dir / 'asv_codec_leaderboard_by_dataset.csv'}")
    print(f"Wrote: {out_dir / 'asv_codec_leaderboard_overall.csv'}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
