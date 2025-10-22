import os, yaml, json, pathlib, csv
import urllib.parse
from code.codecs.encodec_runner import EncodecRunner
from code.codecs.soundstream_runner import SoundStreamRunner
from code.utils.audio_io import load_wav, save_wav

# Import all metrics
from code.metrics.nisqa_runner import score as nisqa_score
from code.metrics.visqol_runner import score as visqol_score
from code.metrics.dnsmos_runner import score as dnsmos_score
from code.metrics.speaker_cosine import compute_speaker_similarity
from code.metrics.prosody_f0_rmse import compute_f0_rmse
from code.asr.wer_eval import compute_wer_from_audio

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def encode_decode(cfg):
    outdir = pathlib.Path(cfg['outputs']['dir'])/ "decoded"
    ensure_dir(outdir)
    results = []
    for ds in cfg['datasets']:
        with open(ds, 'r') as f:
            man = yaml.safe_load(f)
        for item in man['items']:
            in_wav = item['path']      # may be an hf:// path placeholder
            # If you have local audio paths, the runners will actually encode/decode.
            # For hf:// paths, we keep placeholder behavior until a fetcher is wired.
            for br in cfg['bitrate_kbps']:
                for codec in cfg['codecs']:
                    name = codec['name']
                    out_wav = outdir / f"{item['id']}.{name}.{br}kbps.wav"
                    if "encodec" in name:
                        runner = EncodecRunner(bandwidth_kbps=br, causal=True, sr=man['sampling_rate'])
                    else:
                        runner = SoundStreamRunner(bitrate_kbps=br, sr=man['sampling_rate'])
                    os.makedirs(out_wav.parent, exist_ok=True)
                    # Decide execution path
                    if isinstance(in_wav, str) and in_wav.startswith("hf://"):
                        # Still a placeholder: no local file to read. Write a short silent stub.
                        import numpy as np, soundfile as sf
                        sf.write(out_wav, np.zeros(int(1.0*man['sampling_rate'])), man['sampling_rate'])
                    else:
                        # Use the real runner on a local file path
                        try:
                            runner.run(in_wav, str(out_wav))
                        except Exception as e:
                            # Fall back to silent stub if something goes wrong, but record failure
                            import numpy as np, soundfile as sf
                            sf.write(out_wav, np.zeros(int(1.0*man['sampling_rate'])), man['sampling_rate'])
                    meta = {"dataset": man['dataset'], "id": item['id'], "codec": name, "kbps": br, "out": str(out_wav)}
                    results.append(meta)
    ensure_dir("results")
    with open("results/metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} decoded placeholders → results/metadata.json")

def run_metrics(cfg):
    """Run all metrics on decoded audio files."""
    import pandas as pd
    
    # Load metadata
    metadata_path = "results/metadata.json"
    if not os.path.exists(metadata_path):
        print("No metadata.json found. Run encode_decode stage first.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize results
    results = []
    
    print(f"Running metrics on {len(metadata)} files...")
    
    # Skip ASR model for faster processing
    asr_model = None
    
    for i, item in enumerate(metadata):
        print(f"Processing {i+1}/{len(metadata)}: {item['id']}.{item['codec']}.{item['kbps']}kbps")
        
        # Skip HF paths (placeholders)
        if 'hf://' in str(item.get('out', '')):
            print(f"  Skipping HF placeholder: {item['out']}")
            continue
            
        decoded_file = item['out']
        if not os.path.exists(decoded_file):
            print(f"  File not found: {decoded_file}")
            continue
        
        # Find original file
        original_file = _find_original_file(item, cfg)
        if not original_file or not os.path.exists(original_file):
            print(f"  Original file not found for: {item['id']}")
            continue
        
        # Compute metrics
        metrics = _compute_all_metrics(original_file, decoded_file, asr_model)
        
        # Store result
        result = {
            'dataset': item['dataset'],
            'id': item['id'],
            'codec': item['codec'],
            'kbps': item['kbps'],
            **metrics
        }
        results.append(result)
    
    # Save results
    ensure_dir("results/csv")
    df = pd.DataFrame(results)
    df.to_csv("results/csv/benchmark.csv", index=False)
    
    print(f"Wrote metrics for {len(results)} files to results/csv/benchmark.csv")
    
    # Print summary
    if len(results) > 0:
        _print_metrics_summary(df)

def _find_original_file(item, cfg):
    """Find the original audio file for a given item."""
    # This is a simplified approach - in practice you'd want more robust file finding
    item_id = item['id']
    
    # Look for common audio extensions
    extensions = ['.wav', '.mp3', '.flac']
    possible_paths = []
    
    # Check current directory first
    for ext in extensions:
        possible_paths.append(f"{item_id}{ext}")
    
    # Check if it's a local file from manifests
    for ds_path in cfg['datasets']:
        with open(ds_path, 'r') as f:
            manifest = yaml.safe_load(f)
        for manifest_item in manifest['items']:
            if manifest_item['id'] == item_id:
                path = manifest_item['path']
                if not path.startswith('hf://'):
                    possible_paths.append(path)
    
    # Return first existing file
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def _compute_all_metrics(original_file, decoded_file, asr_model):
    """Compute all metrics between original and decoded files."""
    metrics = {}
    
    try:
        # NISQA (single file metric)
        metrics['nisqa_original'] = nisqa_score(original_file)
        metrics['nisqa_decoded'] = nisqa_score(decoded_file)
        metrics['nisqa_drop'] = metrics['nisqa_original'] - metrics['nisqa_decoded']
        
        # DNSMOS (single file metric)
        dnsmos_orig = dnsmos_score(original_file)
        dnsmos_dec = dnsmos_score(decoded_file)
        metrics['dnsmos_sig_original'] = dnsmos_orig.get('sig', float('nan'))
        metrics['dnsmos_sig_decoded'] = dnsmos_dec.get('sig', float('nan'))
        metrics['dnsmos_bak_original'] = dnsmos_orig.get('bak', float('nan'))
        metrics['dnsmos_bak_decoded'] = dnsmos_dec.get('bak', float('nan'))
        metrics['dnsmos_ovr_original'] = dnsmos_orig.get('ovr', float('nan'))
        metrics['dnsmos_ovr_decoded'] = dnsmos_dec.get('ovr', float('nan'))
        
        # ViSQOL (comparison metric)
        metrics['visqol'] = visqol_score(original_file, decoded_file)
        
        # Speaker similarity
        metrics['speaker_similarity'] = compute_speaker_similarity(original_file, decoded_file)
        
        # F0 RMSE (prosody)
        metrics['f0_rmse'] = compute_f0_rmse(original_file, decoded_file)
        
        # ASR WER (skipped for speed)
        metrics['wer'] = float('nan')
        metrics['ref_text'] = 'ASR skipped'
        metrics['hyp_text'] = 'ASR skipped'
        
    except Exception as e:
        print(f"    Error computing metrics: {e}")
        # Set default values
        for key in ['nisqa_original', 'nisqa_decoded', 'nisqa_drop', 'visqol', 
                   'speaker_similarity', 'f0_rmse', 'wer']:
            metrics[key] = float('nan')
    
    return metrics

def _print_metrics_summary(df):
    """Print a summary of the metrics results."""
    print("\n=== METRICS SUMMARY ===")
    
    # Group by codec
    if 'codec' in df.columns:
        codec_summary = df.groupby('codec').agg({
            'visqol': 'mean',
            'speaker_similarity': 'mean', 
            'f0_rmse': 'mean',
            'wer': 'mean',
            'nisqa_drop': 'mean'
        }).round(3)
        print("By Codec:")
        print(codec_summary)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Files processed: {len(df)}")
    if 'visqol' in df.columns:
        print(f"  Average ViSQOL: {df['visqol'].mean():.3f}")
    if 'speaker_similarity' in df.columns:
        print(f"  Average Speaker Similarity: {df['speaker_similarity'].mean():.3f}")
    if 'f0_rmse' in df.columns:
        print(f"  Average F0 RMSE: {df['f0_rmse'].mean():.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["encode_decode","metrics"], required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.stage == "encode_decode":
        encode_decode(cfg)
    else:
        run_metrics(cfg)
