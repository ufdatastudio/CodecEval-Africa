#!/usr/bin/env python3
"""
Download Afri-Names dataset directly from HuggingFace
Bypasses datasets library and TorchCodec by downloading parquet files directly
"""

import os
import json
import random
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import soundfile as sf

# Configuration
DATASET_ID = "intronhealth/afri-names"
OUTPUT_DIR = "data/afri_names_150_flat"
TOTAL_TARGET = 150
SEED = 42


def sanitize_filename(text):
    """Clean accent name for use in filename"""
    return text.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


def download_audio_from_path(audio_path, dest_path):
    """Download audio file from path"""
    try:
        if os.path.exists(audio_path):
            # Copy if local file
            import shutil
            shutil.copy(audio_path, dest_path)
            return True
        else:
            # Download from HuggingFace
            downloaded_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=audio_path,
                repo_type="dataset"
            )
            import shutil
            shutil.copy(downloaded_path, dest_path)
            return True
    except Exception as e:
        print(f"Error downloading {audio_path}: {e}")
        return False


def main():
    print("=" * 80)
    print("Afri-Names Dataset Downloader (Direct Parquet Method)")
    print("=" * 80)
    
    print(f"\nDownloading dataset: {DATASET_ID}")
    print("This method downloads parquet files directly, bypassing TorchCodec")
    
    # List files in dataset
    print("\nStep 1: Listing dataset files...")
    try:
        repo_files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset")
        print(f"Found {len(repo_files)} files in repository")
        
        # Find metadata files (JSON, CSV, or parquet)
        metadata_files = [f for f in repo_files if any(f.endswith(ext) for ext in ['.json', '.jsonl', '.csv', '.parquet', '.tsv'])]
        wav_files = [f for f in repo_files if f.endswith('.wav')]
        
        print(f"Found {len(metadata_files)} metadata file(s)")
        print(f"Found {len(wav_files)} WAV file(s)")
        
        if not metadata_files:
            print("⚠️  No metadata files found. Will need to download all files and infer structure.")
        
        # Try to find metadata file
        metadata_file = None
        for mf in metadata_files:
            if 'train' in mf.lower() or 'metadata' in mf.lower() or 'info' in mf.lower():
                metadata_file = mf
                break
        
        if not metadata_files and not wav_files:
            print("❌ No data files found in dataset")
            return
        
        # Download metadata if available
        df = None
        if metadata_file:
            print(f"\nStep 2: Downloading metadata file: {metadata_file}")
            try:
                local_metadata = hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=metadata_file,
                    repo_type="dataset"
                )
                print(f"✓ Downloaded metadata to: {local_metadata}")
                
                # Read metadata based on file type
                if metadata_file.endswith('.parquet'):
                    df = pd.read_parquet(local_metadata, engine='pyarrow')
                elif metadata_file.endswith('.json') or metadata_file.endswith('.jsonl'):
                    with open(local_metadata, 'r') as f:
                        if metadata_file.endswith('.jsonl'):
                            df = pd.read_json(f, lines=True)
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                            else:
                                df = pd.DataFrame([data])
                elif metadata_file.endswith('.csv') or metadata_file.endswith('.tsv'):
                    df = pd.read_csv(local_metadata, sep='\t' if metadata_file.endswith('.tsv') else ',')
                
                if df is not None:
                    print(f"✓ Loaded {len(df)} rows from metadata")
                    print(f"Columns: {list(df.columns)}")
            except Exception as e:
                print(f"⚠️  Could not read metadata file: {e}")
                df = None
        
        # If no metadata, try to use datasets library in streaming mode to get metadata only
        if df is None:
            print("\nStep 2: No metadata file found. Trying to extract metadata from dataset...")
            print("Note: We'll use streaming mode to get metadata without loading audio")
            
            try:
                # Use datasets library in streaming mode but only access non-audio fields
                from datasets import load_dataset
                print("Loading dataset in streaming mode (metadata only)...")
                
                ds_stream = load_dataset(DATASET_ID, split="train", streaming=True)
                
                # Collect metadata (without accessing audio field)
                metadata_list = []
                print("Extracting metadata from streaming dataset...")
                for i, item in enumerate(ds_stream):
                    # Create metadata dict without audio field
                    meta = {k: v for k, v in item.items() if k != "audio"}
                    if "audio" in item:
                        # Try to get audio path if available
                        audio_obj = item["audio"]
                        if hasattr(audio_obj, "path"):
                            meta["file_path"] = audio_obj.path
                        elif isinstance(audio_obj, dict) and "path" in audio_obj:
                            meta["file_path"] = audio_obj["path"]
                        else:
                            # Construct path from dataset structure
                            # Most datasets store paths relative to data/
                            meta["file_path"] = f"data/{i:05d}_*.wav"  # Placeholder
                    metadata_list.append(meta)
                    
                    if i >= TOTAL_TARGET * 10:  # Get enough samples
                        break
                    if (i + 1) % 1000 == 0:
                        print(f"  Processed {i+1} samples...")
                
                df = pd.DataFrame(metadata_list)
                print(f"✓ Extracted metadata for {len(df)} samples")
                print(f"Columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"⚠️  Could not extract metadata from dataset: {e}")
                print("Falling back to WAV file list...")
                
                if wav_files:
                    print(f"Sample WAV files: {wav_files[:5]}")
                    # Create a simple dataframe with file paths
                    df = pd.DataFrame({'file_path': wav_files})
                    print(f"Created dataframe with {len(df)} file paths")
                else:
                    print("❌ No WAV files found")
                    return
        
        # Extract accents
        print("\nStep 3: Analyzing accents...")
        if "accent" not in df.columns:
            print("⚠️  'accent' column not found in metadata")
            print(f"Available columns: {list(df.columns)}")
            print("\nTrying to infer accents from file paths or other columns...")
            
            # Check if accent info is in filename or other columns
            accent_cols = [col for col in df.columns if 'accent' in col.lower() or 'language' in col.lower() or 'lang' in col.lower()]
            if accent_cols:
                df["accent"] = df[accent_cols[0]]
                print(f"Using '{accent_cols[0]}' column as accent")
            else:
                print("❌ Cannot determine accents from metadata. Please check dataset structure.")
                return
        
        accents = sorted(df["accent"].unique().tolist())
        print(f"Found {len(accents)} accents: {accents}")
        
    except Exception as e:
        print(f"\n❌ Error accessing dataset: {e}")
        print("\n" + "=" * 80)
        print("ALTERNATIVE SOLUTION:")
        print("=" * 80)
        print("Use HuggingFace CLI to download raw files:")
        print(f"  huggingface-cli download {DATASET_ID}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate sampling strategy
    num_accents = len(accents)
    per_accent = TOTAL_TARGET // num_accents
    remainder = TOTAL_TARGET % num_accents
    
    print(f"\nSampling strategy:")
    print(f"  Target total: {TOTAL_TARGET} files")
    print(f"  Base per accent: {per_accent} files")
    if remainder > 0:
        print(f"  Additional files: {remainder} files will be distributed to first {remainder} accents")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Download files
    counter = {}
    downloaded_files = 0
    failed_files = 0
    
    random.seed(SEED)
    
    print("\nStep 4: Downloading audio files...")
    for accent_idx, accent in enumerate(accents):
        # Adjust count for remainder distribution
        count = per_accent + (1 if accent_idx < remainder else 0)
        
        # Filter by accent
        accent_df = df[df["accent"] == accent].copy()
        
        # Shuffle
        accent_df = accent_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        
        # Limit
        count = min(len(accent_df), count)
        counter[accent] = count
        
        print(f"\nProcessing {accent} ({count} files from {len(accent_df)} available)...")
        
        accent_clean = sanitize_filename(accent)
        
        for file_idx in tqdm(range(count), desc=f"  {accent}"):
            try:
                row = accent_df.iloc[file_idx]
                
                # Get audio path - check various possible column names
                audio_path = None
                
                # Check for file path columns
                for col in ["file_path", "path", "audio_path", "file", "audio"]:
                    if col in row.index and pd.notna(row[col]):
                        val = row[col]
                        if isinstance(val, dict) and "path" in val:
                            audio_path = val["path"]
                        elif isinstance(val, str):
                            audio_path = val
                        elif isinstance(val, dict) and "array" in val:
                            # Audio data is embedded - write directly
                            new_filename = f"{accent_clean}{file_idx+1:02d}.wav"
                            dest_path = os.path.join(OUTPUT_DIR, new_filename)
                            if "sampling_rate" in val:
                                sf.write(dest_path, val["array"], val["sampling_rate"])
                                downloaded_files += 1
                            else:
                                failed_files += 1
                            continue
                        break
                
                if not audio_path:
                    # Try to construct path from filename or other columns
                    if "file_path" in row.index:
                        audio_path = row["file_path"]
                    elif any("file" in col.lower() for col in row.index):
                        file_col = [col for col in row.index if "file" in col.lower()][0]
                        audio_path = row[file_col]
                    else:
                        print(f"  Warning: No audio path found in row {file_idx+1}")
                        print(f"  Available columns: {list(row.index)}")
                        failed_files += 1
                        continue
                
                # Format filename
                new_filename = f"{accent_clean}{file_idx+1:02d}.wav"
                dest_path = os.path.join(OUTPUT_DIR, new_filename)
                
                # Download audio file from HuggingFace
                try:
                    # Download from HuggingFace repo
                    downloaded_audio = hf_hub_download(
                        repo_id=DATASET_ID,
                        filename=audio_path,
                        repo_type="dataset"
                    )
                    # Copy to destination
                    import shutil
                    shutil.copy(downloaded_audio, dest_path)
                    downloaded_files += 1
                except Exception as e:
                    print(f"  Error downloading {audio_path}: {e}")
                    failed_files += 1
                    
            except Exception as e:
                print(f"\nError processing {accent} sample {file_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                failed_files += 1
                continue
    
    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Total downloaded: {downloaded_files} files")
    if failed_files > 0:
        print(f"Failed downloads: {failed_files} files")
    print(f"\nBreakdown by accent:")
    for accent, count in counter.items():
        print(f"  {accent}: {count} files")
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

