"""
FocalCodec batch runner
-----------------------
Encode/decode WAV folders with FocalCodec and save reconstructions.
"""

import os
import time
import gc
from pathlib import Path
from typing import Optional, Sequence

import torch
import torchaudio
from tqdm import tqdm


DEFAULT_FOCALCODEC_CONFIGS = [
    "lucadellalib/focalcodec_50hz",
    "lucadellalib/focalcodec_50hz_65k_causal",
    "lucadellalib/focalcodec_50hz_4k_causal",
    "lucadellalib/focalcodec_50hz_2k_causal",
    "lucadellalib/focalcodec_25hz",
    "lucadellalib/focalcodec_12_5hz",
]


def _collect_wavs(input_dir: str):
    wavs = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".wav"):
                wavs.append(os.path.join(root, name))
    return sorted(wavs)


def _load_mono(path: str):
    sig, sr = torchaudio.load(path)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    return sig, sr


def focalcodec_folder(
    input_dir: str,
    output_root: str,
    model_repo: str = "lucadellalib/focalcodec",
    model_name: str = "focalcodec",
    config: Optional[str] = None,
    variant_name: Optional[str] = None,
    configs: Optional[Sequence[str]] = None,
    skip_existing: bool = True,
    chunk_seconds: float = 8.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_list = list(configs) if configs is not None else [config or "lucadellalib/focalcodec_50hz"]
    wav_files = _collect_wavs(input_dir)
    print(f"Found {len(wav_files)} wav files in {input_dir}")

    for cfg in cfg_list:
        current_variant = variant_name or cfg.split("/")[-1]

        print(f"Loading FocalCodec from torch hub: repo={model_repo}, config={cfg}")
        codec = torch.hub.load(
            repo_or_dir=model_repo,
            model=model_name,
            config=cfg,
            force_reload=False,
        )
        codec.eval().requires_grad_(False)
        try:
            codec = codec.to(device)
        except Exception:
            pass

        sr_in = int(codec.sample_rate_input)
        sr_out = int(codec.sample_rate_output)
        print(f"FocalCodec sample rates: input={sr_in}, output={sr_out}")

        out_dir = Path(output_root) / current_variant
        out_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = 0
        skipped = 0

        for wav_path in tqdm(wav_files, desc=f"FocalCodec ({current_variant})"):
            filename = os.path.basename(wav_path)
            out_path = out_dir / filename

            if skip_existing and out_path.exists():
                skipped += 1
                continue

            try:
                sig, sr_orig = _load_mono(wav_path)
                sig = torchaudio.functional.resample(sig, sr_orig, sr_in)
                chunk_len = max(1, int(chunk_seconds * sr_in))
                rec_chunks = []

                with torch.inference_mode():
                    for start in range(0, sig.shape[-1], chunk_len):
                        end = min(start + chunk_len, sig.shape[-1])
                        sig_chunk = sig[:, start:end].to(device)

                        toks = codec.sig_to_toks(sig_chunk)
                        rec_chunk = codec.toks_to_sig(toks)
                        rec_chunks.append(rec_chunk.detach().cpu())

                        del sig_chunk, toks, rec_chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                rec_sig = torch.cat(rec_chunks, dim=-1)
                del rec_chunks
                if rec_sig.dim() == 1:
                    rec_sig = rec_sig.unsqueeze(0)
                elif rec_sig.dim() == 3:
                    rec_sig = rec_sig.squeeze(0)

                rec_sig = torchaudio.functional.resample(rec_sig, sr_out, sr_orig)
                torchaudio.save(str(out_path), rec_sig, sr_orig)
                success += 1

                del sig, rec_sig
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as exc:
                failed += 1
                print(f"Failed: {filename} -> {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print("----------------------------------------")
        print(f"Variant: {current_variant}")
        print(f"Saved to: {out_dir}")
        print(f"Success: {success} | Failed: {failed} | Skipped: {skipped}")
        print("----------------------------------------")


if __name__ == "__main__":
    input_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afri_names_150_flat"
    output_folder = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrinames/FocalCodec_outputs"

    t0 = time.time()
    focalcodec_folder(input_folder, output_folder, configs=DEFAULT_FOCALCODEC_CONFIGS)
    print(f"Elapsed: {time.time() - t0:.1f}s")
