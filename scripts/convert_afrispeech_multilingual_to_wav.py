#!/usr/bin/env python3
"""
Convert afrispeech_multilingual dataset audio files to WAV format
Supports MP3, FLAC, M4A, and other audio formats
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import torchaudio
import numpy as np

# Configuration
INPUT_DIR = "data/afrispeech_multilingual"
OUTPUT_DIR = "data/afrispeech_multilingual_wav"
TARGET_SAMPLE_RATE = 24000  # Standard sample rate for codec evaluation
TARGET_CHANNELS = 1  # Mono audio
SUPPORTED_FORMATS = ['.mp3', '.flac', '.m4a', '.wav', '.ogg', '.aac', '.wma']


def convert_audio_file(input_path, output_path, target_sr=TARGET_SAMPLE_RATE, target_channels=TARGET_CHANNELS):
    """
    Convert audio file to WAV format with specified sample rate and channels.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        target_sr: Target sample rate (default: 24000)
        target_channels: Target number of channels (default: 1 for mono)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Try using torchaudio first (better format support)
        try:
            waveform, sample_rate = torchaudio.load(input_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                if target_channels == 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                else:
                    # For stereo, take first channel or mix
                    waveform = waveform[0:1]
            elif waveform.shape[0] == 1 and target_channels == 2:
                # Duplicate for stereo
                waveform = waveform.repeat(2, 1)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as WAV
            torchaudio.save(
                output_path,
                waveform,
                target_sr,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            return True
            
        except Exception as e:
            # Fallback to soundfile + librosa for resampling
            print(f"  Warning: torchaudio failed ({e}), trying soundfile...")
            
            try:
                import librosa
                
                # Load audio with librosa (handles many formats)
                audio, sr = librosa.load(input_path, sr=None, mono=False)
                
                # Convert to numpy array
                if isinstance(audio, np.ndarray):
                    if audio.ndim == 1:
                        audio = audio.reshape(1, -1)  # Make it 2D: [channels, samples]
                    elif audio.ndim > 2:
                        audio = audio.reshape(-1, audio.shape[-1])
                
                # Convert to mono if needed
                if audio.shape[0] > 1:
                    if target_channels == 1:
                        audio = np.mean(audio, axis=0, keepdims=True)
                    else:
                        audio = audio[0:1]  # Take first channel
                elif audio.shape[0] == 1 and target_channels == 2:
                    audio = np.repeat(audio, 2, axis=0)
                
                # Resample if needed
                if sr != target_sr:
                    if audio.shape[0] == 1:
                        audio_resampled = librosa.resample(audio[0], orig_sr=sr, target_sr=target_sr)
                        audio = audio_resampled.reshape(1, -1)
                    else:
                        audio_resampled = []
                        for ch in range(audio.shape[0]):
                            audio_resampled.append(librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr))
                        audio = np.array(audio_resampled)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save as WAV
                if audio.shape[0] == 1:
                    sf.write(output_path, audio[0], target_sr, subtype='PCM_16')
                else:
                    sf.write(output_path, audio.T, target_sr, subtype='PCM_16')  # Transpose for soundfile
                
                return True
                
            except Exception as e2:
                print(f"  Error: Both torchaudio and soundfile failed: {e2}")
                return False
                
    except Exception as e:
        print(f"  Error converting {input_path}: {e}")
        return False


def find_audio_files(directory, extensions=SUPPORTED_FORMATS):
    """
    Find all audio files in directory (recursively).
    
    Args:
        directory: Root directory to search
        extensions: List of file extensions to include
    
    Returns:
        list: List of Path objects for audio files
    """
    audio_files = []
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return audio_files
    
    for ext in extensions:
        audio_files.extend(directory.rglob(f"*{ext}"))
        audio_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    return sorted(audio_files)


def main():
    print("=" * 80)
    print("AfriSpeech Multilingual Audio to WAV Converter")
    print("=" * 80)
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Target channels: {TARGET_CHANNELS} (mono)")
    print()
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist")
        print("Please make sure the dataset is in the correct location")
        return 1
    
    # Find all audio files
    print("Step 1: Scanning for audio files...")
    audio_files = find_audio_files(INPUT_DIR)
    
    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return 1
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Count by format
    format_counts = {}
    for f in audio_files:
        ext = f.suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    print("\nFiles by format:")
    for ext, count in sorted(format_counts.items()):
        print(f"  {ext}: {count} files")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert files
    print(f"\nStep 2: Converting audio files to WAV...")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    converted = 0
    failed = 0
    skipped = 0
    
    for audio_file in tqdm(audio_files, desc="Converting"):
        # Get relative path from input directory
        try:
            rel_path = audio_file.relative_to(Path(INPUT_DIR))
        except ValueError:
            # If relative path fails, use just the filename
            rel_path = Path(audio_file.name)
        
        # Create output path (change extension to .wav)
        output_path = Path(OUTPUT_DIR) / rel_path.with_suffix('.wav')
        
        # Skip if output already exists
        if output_path.exists():
            skipped += 1
            continue
        
        # Convert audio file
        if convert_audio_file(str(audio_file), str(output_path)):
            converted += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)
    print(f"Total files: {len(audio_files)}")
    print(f"Converted: {converted}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 80)
    
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

