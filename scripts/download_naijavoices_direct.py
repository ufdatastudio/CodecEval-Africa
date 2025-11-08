#!/usr/bin/env python3
"""
Download NaijaVoices dataset directly from HuggingFace
Bypasses datasets library and TorchCodec by downloading parquet files directly
Supports multilingual dataset with 3 languages: Igbo, Hausa, Yoruba
"""

import os
import json
import random
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, login
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import soundfile as sf

# Configuration
DATASET_ID = "naijavoices/naijavoices-dataset"
COMPRESSED_DATASET_ID = "naijavoices/naijavoices-dataset-compressed"  # 84GB compressed version
OUTPUT_DIR = "data/naijavoices_150_flat"
TOTAL_TARGET = 150
SEED = 42
USE_COMPRESSED = True  # Use compressed version (84GB) instead of full (500+ GB)

# Languages in the dataset
LANGUAGES = ["igbo", "hausa", "yoruba"]


def sanitize_filename(text):
    """Clean language name for use in filename"""
    return text.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def check_huggingface_login():
    """Check if user is logged in to HuggingFace"""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✓ Logged in as: {user.get('name', 'Unknown')}")
        return True
    except Exception:
        print("⚠️  Not logged in to HuggingFace")
        print("This dataset is gated and requires login.")
        print("Please run: huggingface-cli login")
        print("Or accept the terms at: https://huggingface.co/datasets/naijavoices/naijavoices-dataset")
        return False


def load_dataset_from_csv(dataset_id, split="train"):
    """Load dataset metadata from CSV file"""
    print(f"Loading {split} split from CSV...")
    
    try:
        # Download CSV file directly
        csv_path = f"split/{split}.csv"
        print(f"  Downloading CSV: {csv_path}")
        
        local_csv = hf_hub_download(
            repo_id=dataset_id,
            filename=csv_path,
            repo_type="dataset"
        )
        
        print(f"  ✓ Downloaded CSV to: {local_csv}")
        
        # Read CSV
        df = pd.read_csv(local_csv)
        print(f"  ✓ Loaded {len(df)} rows from CSV")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"  Error downloading/reading CSV: {e}")
        print("  Falling back to datasets library streaming mode...")
        
        # Fallback: use datasets library in streaming mode
        try:
            from datasets import load_dataset
            print(f"  Using datasets library (streaming mode)...")
            
            ds = load_dataset(dataset_id, split=split, streaming=True)
            
            # Collect metadata (without accessing audio field)
            metadata_list = []
            print("  Extracting metadata...")
            for i, item in enumerate(ds):
                # Create metadata dict without audio field
                meta = {k: v for k, v in item.items() if k != "audio"}
                
                # Handle audio field
                if "audio" in item:
                    audio_obj = item["audio"]
                    if hasattr(audio_obj, "path"):
                        meta["audio_path"] = audio_obj.path
                    elif isinstance(audio_obj, dict):
                        if "path" in audio_obj:
                            meta["audio_path"] = audio_obj["path"]
                        elif "array" in audio_obj:
                            # Store audio data for later
                            meta["audio_data"] = audio_obj
                            if "sampling_rate" in audio_obj:
                                meta["sampling_rate"] = audio_obj["sampling_rate"]
                
                metadata_list.append(meta)
                
                # Limit for initial exploration
                if i >= TOTAL_TARGET * 10:
                    break
                if (i + 1) % 1000 == 0:
                    print(f"    Processed {i+1} samples...")
            
            df = pd.DataFrame(metadata_list)
            print(f"  ✓ Extracted {len(df)} samples")
            return df
            
        except Exception as e2:
            print(f"  Error loading with datasets library: {e2}")
            return None


def main():
    print("=" * 80)
    print("NaijaVoices Dataset Downloader (Direct Parquet Method)")
    print("=" * 80)
    
    # Check HuggingFace login
    if not check_huggingface_login():
        print("\nPlease login to HuggingFace first:")
        print("  huggingface-cli login")
        print("Then accept the dataset terms at:")
        print(f"  https://huggingface.co/datasets/{DATASET_ID}")
        return
    
    # Select dataset version
    dataset_id = COMPRESSED_DATASET_ID if USE_COMPRESSED else DATASET_ID
    print(f"\nUsing dataset: {dataset_id}")
    if USE_COMPRESSED:
        print("  (Compressed version: 84GB)")
        print("  Note: Compressed version uses CSV splits instead of batch configs")
    else:
        print("  (Full version: 500+ GB)")
        print("  Note: Full version may have batch configs")
    
    # Load data from train split
    print("\nStep 1: Loading data from train split...")
    df = load_dataset_from_csv(dataset_id, split="train")
    
    if df is None or len(df) == 0:
        print("❌ No data loaded from dataset")
        return
    
    print(f"\n✓ Total samples loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\n⚠️  NOTE: We only loaded metadata (CSV file).")
    print(f"         We will only download {TOTAL_TARGET} audio files total ({TOTAL_TARGET // len(LANGUAGES)} per language)")
    print(f"         This is just metadata exploration - no audio downloaded yet!\n")
    
    # Extract languages
    print("Step 2: Analyzing languages...")
    if "language" not in df.columns:
        print("⚠️  'language' column not found in metadata")
        print(f"Available columns: {list(df.columns)}")
        
        # Try to infer from other columns
        lang_cols = [col for col in df.columns if 'language' in col.lower() or 'lang' in col.lower()]
        if lang_cols:
            df["language"] = df[lang_cols[0]]
            print(f"Using '{lang_cols[0]}' column as language")
        else:
            print("❌ Cannot determine languages from metadata")
            print("Please check the CSV file structure")
            return
    
    # Normalize language names
    df["language"] = df["language"].str.lower().str.strip()
    
    # Filter to only the 3 main languages
    df = df[df["language"].isin([lang.lower() for lang in LANGUAGES])]
    
    languages = sorted(df["language"].unique().tolist())
    print(f"Found {len(languages)} languages: {languages}")
    
    # Calculate sampling strategy
    num_languages = len(languages)
    per_language = TOTAL_TARGET // num_languages
    remainder = TOTAL_TARGET % num_languages
    
    print(f"\nSampling strategy:")
    print(f"  Target total: {TOTAL_TARGET} files")
    print(f"  Base per language: {per_language} files")
    if remainder > 0:
        print(f"  Additional files: {remainder} files will be distributed to first {remainder} languages")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Download files
    counter = {}
    downloaded_files = 0
    failed_files = 0
    
    random.seed(SEED)
    
    # Build a set of target audio filenames and their destination paths
    print("\nStep 3: Preparing download targets...")
    target_files = {}  # audio_filename -> (language, dest_filename, dest_path)
    
    for lang_idx, language in enumerate(languages):
        # Adjust count for remainder distribution
        count = per_language + (1 if lang_idx < remainder else 0)
        
        # Filter by language
        lang_df = df[df["language"] == language].copy()
        
        print(f"\n  Found {len(lang_df)} samples for {language}")
        print(f"  Will sample {count} files for download")
        
        # Shuffle
        lang_df = lang_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        
        # Limit
        count = min(len(lang_df), count)
        counter[language] = count
        
        lang_clean = sanitize_filename(language)
        
        for file_idx in range(count):
            row = lang_df.iloc[file_idx]
            audio_path = row.get("audio", None)
            
            if audio_path and isinstance(audio_path, str):
                new_filename = f"{lang_clean}{file_idx+1:02d}.wav"
                dest_path = os.path.join(OUTPUT_DIR, new_filename)
                target_files[audio_path] = (language, new_filename, dest_path)
    
    print(f"\n✓ Prepared {len(target_files)} files for download")
    print(f"Now downloading from dataset (this may take a while)...")
    
    # Iterate through dataset once and extract target files
    print("\nStep 4: Downloading audio files from dataset...")
    from datasets import load_dataset
    
    ds_stream = load_dataset(dataset_id, split="train", streaming=True)
    remaining = len(target_files)
    
    for item in tqdm(ds_stream, desc="  Processing dataset", total=None):
        if remaining == 0:
            break
            
        # Check if this item's audio filename matches our target
        item_audio_field = item.get("audio", "")
        # The audio field might be a string (filename) or an Audio object
        if isinstance(item_audio_field, str):
            audio_filename = item_audio_field
        else:
            # Try to get filename from Audio object
            audio_filename = getattr(item_audio_field, "path", None)
            if audio_filename:
                # Extract just the filename from path
                audio_filename = os.path.basename(audio_filename)
        
        if audio_filename and audio_filename in target_files:
            language, new_filename, dest_path = target_files[audio_filename]
            
            try:
                audio_obj = item.get("audio")
                extracted = False
                
                # Handle audio data extraction - try multiple formats
                if isinstance(audio_obj, dict):
                    if "array" in audio_obj:
                        sf.write(dest_path, audio_obj["array"], audio_obj.get("sampling_rate", 16000))
                        downloaded_files += 1
                        extracted = True
                    elif "path" in audio_obj:
                        shutil.copy(audio_obj["path"], dest_path)
                        downloaded_files += 1
                        extracted = True
                
                if not extracted and hasattr(audio_obj, "array"):
                    sf.write(dest_path, audio_obj.array, getattr(audio_obj, "sampling_rate", 16000))
                    downloaded_files += 1
                    extracted = True
                
                if not extracted and hasattr(audio_obj, "path"):
                    shutil.copy(audio_obj.path, dest_path)
                    downloaded_files += 1
                    extracted = True
                
                if not extracted:
                    print(f"  Warning: Could not extract audio from {audio_filename} (type: {type(audio_obj)})")
                    failed_files += 1
                
                remaining -= 1
                del target_files[audio_filename]  # Remove from target list
                
            except Exception as e:
                print(f"  Error extracting {audio_filename}: {e}")
                failed_files += 1
    
    # Check for any files we didn't find
    if target_files:
        print(f"\n  Warning: {len(target_files)} files were not found in dataset")
        failed_files += len(target_files)
    
    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Total downloaded: {downloaded_files} files")
    if failed_files > 0:
        print(f"Failed downloads: {failed_files} files")
    print(f"\nBreakdown by language:")
    for language, count in counter.items():
        print(f"  {language}: {count} files")
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

