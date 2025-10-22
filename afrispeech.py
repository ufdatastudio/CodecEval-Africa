#!/usr/bin/env python3
"""
Download and extract AfriSpeech-200 audio and transcript archives.
Adapted from the original afrispeech.py script to include transcripts and manifest creation.
"""

import os
import sys
import csv
import tarfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "intronhealth/afrispeech-200"
OUT_DIR = Path("data/afrispeech_full")  # local directory for extracted data

# Create output directory if not exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize API
api = HfApi()

print("=" * 80)
print("AFRISPEECH-200 DOWNLOAD AND EXTRACTION")
print("=" * 80)

# --- Step 1: List files in the repository ---
print("Getting repository file list...")
files = api.list_repo_files(repo_id=REPO_ID)

# Find audio and transcript archives
audio_archives = [f for f in files if f.startswith("audio/") and f.endswith(".tar.gz")]
transcript_archives = [f for f in files if f.startswith("transcript/") and f.endswith(".tar.gz")]

print(f"Found {len(audio_archives)} audio archives and {len(transcript_archives)} transcript archives")
print("Audio archives:", audio_archives[:3])

# --- Step 2: Download and extract each accent archive ---
all_samples = []

for file_path in audio_archives:
    accent_name = os.path.basename(file_path).replace(".tar.gz", "")
    accent_dir = OUT_DIR / accent_name
    
    print(f"\n{'='*50}")
    print(f"Processing accent: {accent_name}")
    print(f"{'='*50}")
    
    # Create accent directories
    audio_dir = accent_dir / "audio"
    transcript_dir = accent_dir / "transcript"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract audio archive
    print(f"Downloading audio archive: {file_path}")
    local_audio_tar = hf_hub_download(repo_id=REPO_ID, filename=file_path)
    
    print(f"Extracting audio to: {audio_dir}")
    with tarfile.open(local_audio_tar, "r:gz") as tar:
        tar.extractall(path=audio_dir)

    # Find corresponding transcript archive
    transcript_archive = None
    for trans_arch in transcript_archives:
        if accent_name in trans_arch:
            transcript_archive = trans_arch
            break
    
    if transcript_archive:
        print(f"Downloading transcript archive: {transcript_archive}")
        local_transcript_tar = hf_hub_download(repo_id=REPO_ID, filename=transcript_archive)
        
        print(f"Extracting transcripts to: {transcript_dir}")
        with tarfile.open(local_transcript_tar, "r:gz") as tar:
            tar.extractall(path=transcript_dir)
    else:
        print(f"⚠ No transcript archive found for {accent_name}")

    # Find test audio files and their corresponding transcripts
    test_audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        if "test" in root:
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    test_audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_audio_files)} test audio files")
    
    # Process test files (limit to 3 samples per accent for manageable size)
    accent_samples = []
    for audio_file in test_audio_files[:3]:
        try:
            filename = os.path.basename(audio_file)
            base_name = os.path.splitext(filename)[0]
            
            # Find corresponding transcript
            transcript_content = ""
            transcript_path = ""
            
            # Look for transcript in test folder
            test_transcript_dir = transcript_dir / "test" if (transcript_dir / "test").exists() else transcript_dir
            
            # Try different transcript file extensions
            for ext in ['.csv', '.txt']:
                transcript_file = test_transcript_dir / f"{base_name}{ext}"
                if transcript_file.exists():
                    transcript_path = str(transcript_file)
                    
                    # Read transcript content
                    if ext == '.csv':
                        # Read CSV transcript
                        with open(transcript_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if row.get('filename', '').replace('.wav', '') == base_name:
                                    transcript_content = row.get('transcript', '').strip()
                                    break
                    else:
                        # Read text transcript
                        with open(transcript_file, 'r', encoding='utf-8') as f:
                            transcript_content = f.read().strip()
                    break
            
            if transcript_content:
                sample = {
                    'filename': filename,
                    'accent': accent_name,
                    'audio_path': audio_file,
                    'transcript_path': transcript_path,
                    'transcript': transcript_content
                }
                accent_samples.append(sample)
                all_samples.append(sample)
                
                print(f"✓ {filename}: '{transcript_content[:50]}...'")
            else:
                print(f"⚠ No transcript found for {filename}")
                
        except Exception as e:
            print(f"✗ Error processing {audio_file}: {e}")
    
    print(f"Processed {len(accent_samples)} samples for {accent_name}")

# Create manifest CSV
manifest_file = OUT_DIR / "afrispeech_manifest.csv"
with open(manifest_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['filename', 'accent', 'audio_path', 'transcript_path', 'transcript']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_samples)

print(f"\n" + "=" * 80)
print("DOWNLOAD AND EXTRACTION COMPLETE")
print("=" * 80)
print(f"Total samples: {len(all_samples)}")
print(f"Manifest file: {manifest_file}")
print(f"Output directory: {OUT_DIR}")

# Show accent distribution
accent_counts = {}
for sample in all_samples:
    accent = sample['accent']
    accent_counts[accent] = accent_counts.get(accent, 0) + 1

print(f"\nAccent distribution:")
for accent, count in sorted(accent_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {accent}: {count} samples")

print("\nAll accent archives downloaded and extracted successfully!")
