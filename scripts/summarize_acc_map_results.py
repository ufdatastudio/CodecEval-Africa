#!/usr/bin/env python3
"""Summarize ACC/mAP JSON results into leaderboard artifacts."""

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
        if stem.startswith(prefix) and stem.endswith("_acc_map"):
            return dataset, stem[len(prefix):-8]
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


def _quality_flag(acc, map_score):
    if acc is None or map_score is None:
        return "invalid"
    if acc >= 99.0 and map_score >= 99.0:
        return "excellent"
    if acc >= 95.0 and map_score >= 97.0:
        return "strong"
    if acc >= 80.0 and map_score >= 90.0:
        return "moderate"
    return "weak"


def main():
    parser = argparse.ArgumentParser(description="Summarize ACC/mAP results and create leaderboards")
    parser.add_argument("--input-dir", default="results/ACC_MAP/metrics", help="Directory containing *_acc_map.json files")
    parser.add_argument("--output-dir", default="results/ACC_MAP/analysis", help="Directory for summary outputs")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = []
    for path in sorted(glob.glob(str(in_dir / "*_acc_map.json"))):
        p = Path(path)
        dataset, suffix = _parse_dataset_and_suffix(p.stem)
        if dataset is None:
            continue
        codec, variant = _parse_codec_variant(suffix)

        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        m = data.get("metrics", {})
        acc_ci = m.get("acc_percent_ci95")
        map_ci = m.get("map_percent_ci95")

        row = {
            "dataset": dataset,
            "codec": codec,
            "variant": variant,
            "file": str(p),
            "acc_percent": _safe(m.get("acc_percent")),
            "map_percent": _safe(m.get("map_percent")),
            "acc_percent_ci95_lower": _safe(acc_ci[0]) if isinstance(acc_ci, list) and len(acc_ci) == 2 else None,
            "acc_percent_ci95_upper": _safe(acc_ci[1]) if isinstance(acc_ci, list) and len(acc_ci) == 2 else None,
            "map_percent_ci95_lower": _safe(map_ci[0]) if isinstance(map_ci, list) and len(map_ci) == 2 else None,
            "map_percent_ci95_upper": _safe(map_ci[1]) if isinstance(map_ci, list) and len(map_ci) == 2 else None,
            "num_queries": data.get("num_queries"),
            "model_type": data.get("model_type"),
            "bootstrap_samples": data.get("bootstrap_samples"),
        }
        row["quality_flag"] = _quality_flag(row["acc_percent"], row["map_percent"])
        variant_rows.append(row)

    variant_headers = [
        "dataset", "codec", "variant", "acc_percent", "map_percent",
        "acc_percent_ci95_lower", "acc_percent_ci95_upper",
        "map_percent_ci95_lower", "map_percent_ci95_upper",
        "num_queries", "model_type", "bootstrap_samples", "quality_flag", "file",
    ]
    _write_csv(out_dir / "acc_map_variant_table.csv", variant_rows, variant_headers)

    grouped = defaultdict(list)
    for row in variant_rows:
        grouped[(row["dataset"], row["codec"])].append(row)

    by_dataset_rows = []
    for (dataset, codec), rows in sorted(grouped.items()):
        by_dataset_rows.append({
            "dataset": dataset,
            "codec": codec,
            "num_variants": len(rows),
            "mean_acc_percent": _mean([r["acc_percent"] for r in rows]),
            "std_acc_percent": _std([r["acc_percent"] for r in rows]),
            "mean_map_percent": _mean([r["map_percent"] for r in rows]),
            "std_map_percent": _std([r["map_percent"] for r in rows]),
            "mean_acc_ci95_lower": _mean([r["acc_percent_ci95_lower"] for r in rows]),
            "mean_acc_ci95_upper": _mean([r["acc_percent_ci95_upper"] for r in rows]),
            "mean_map_ci95_lower": _mean([r["map_percent_ci95_lower"] for r in rows]),
            "mean_map_ci95_upper": _mean([r["map_percent_ci95_upper"] for r in rows]),
            "num_excellent_variants": sum(1 for r in rows if r["quality_flag"] == "excellent"),
            "num_weak_variants": sum(1 for r in rows if r["quality_flag"] == "weak"),
        })

    by_dataset_rows.sort(key=lambda r: (r["dataset"], -(r["mean_map_percent"] if r["mean_map_percent"] is not None else -1e9)))
    _write_csv(
        out_dir / "acc_map_codec_leaderboard_by_dataset.csv",
        by_dataset_rows,
        [
            "dataset", "codec", "num_variants",
            "mean_acc_percent", "std_acc_percent",
            "mean_map_percent", "std_map_percent",
            "mean_acc_ci95_lower", "mean_acc_ci95_upper",
            "mean_map_ci95_lower", "mean_map_ci95_upper",
            "num_excellent_variants", "num_weak_variants",
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
            "overall_mean_acc_percent": _mean([r["mean_acc_percent"] for r in rows]),
            "overall_mean_map_percent": _mean([r["mean_map_percent"] for r in rows]),
            "overall_mean_acc_ci95_lower": _mean([r["mean_acc_ci95_lower"] for r in rows]),
            "overall_mean_acc_ci95_upper": _mean([r["mean_acc_ci95_upper"] for r in rows]),
            "overall_mean_map_ci95_lower": _mean([r["mean_map_ci95_lower"] for r in rows]),
            "overall_mean_map_ci95_upper": _mean([r["mean_map_ci95_upper"] for r in rows]),
            "total_excellent_variants": sum(r["num_excellent_variants"] for r in rows),
            "total_weak_variants": sum(r["num_weak_variants"] for r in rows),
        })

    overall_rows.sort(key=lambda r: (r["overall_mean_map_percent"] if r["overall_mean_map_percent"] is not None else -1e9), reverse=True)
    _write_csv(
        out_dir / "acc_map_codec_leaderboard_overall.csv",
        overall_rows,
        [
            "codec", "datasets_covered",
            "overall_mean_acc_percent", "overall_mean_map_percent",
            "overall_mean_acc_ci95_lower", "overall_mean_acc_ci95_upper",
            "overall_mean_map_ci95_lower", "overall_mean_map_ci95_upper",
            "total_excellent_variants", "total_weak_variants",
        ],
    )

    md = out_dir / "acc_map_summary.md"
    with md.open("w") as f:
        f.write("# ACC/mAP Summary\n\n")
        f.write(f"Variants processed: {len(variant_rows)}\n\n")
        f.write("## Overall codec ranking (higher is better)\n\n")
        f.write("| Rank | Codec | Mean ACC% | Mean mAP% | Mean ACC CI upper | Mean mAP CI upper | Weak variants |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for i, row in enumerate(overall_rows, 1):
            acc = row["overall_mean_acc_percent"]
            mapv = row["overall_mean_map_percent"]
            acc_u = row["overall_mean_acc_ci95_upper"]
            map_u = row["overall_mean_map_ci95_upper"]
            f.write(
                f"| {i} | {row['codec']} | {acc if acc is not None else float('nan'):.3f} | "
                f"{mapv if mapv is not None else float('nan'):.3f} | "
                f"{acc_u if acc_u is not None else float('nan'):.3f} | "
                f"{map_u if map_u is not None else float('nan'):.3f} | "
                f"{row['total_weak_variants']} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- This implementation is retrieval-based: each degraded file queries all references by cosine score.\n")
        f.write("- `ACC` is top-1 identification accuracy.\n")
        f.write("- `mAP` here is equivalent to mean reciprocal rank because each query has one relevant item.\n")

    print(f"Wrote: {out_dir / 'acc_map_variant_table.csv'}")
    print(f"Wrote: {out_dir / 'acc_map_codec_leaderboard_by_dataset.csv'}")
    print(f"Wrote: {out_dir / 'acc_map_codec_leaderboard_overall.csv'}")
    print(f"Wrote: {md}")


if __name__ == "__main__":
    main()
