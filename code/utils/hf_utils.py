"""
Utilities for working with Hugging Face datasets.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from datasets import load_dataset

def download_hf_audio(repo_id: str, filename: str, output_path: str) -> Optional[str]:
    """
    Download audio file from Hugging Face dataset.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "intronhealth/afri-names")
        filename: Audio filename (e.g., "en_ng_036.wav")
        output_path: Local path to save the audio file
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    try:
        # Load the dataset
        dataset = load_dataset(repo_id, split="train")
        
        # Find the item with matching filename
        item = None
        for sample in dataset:
            if sample.get('audio', {}).get('path', '').endswith(filename):
                item = sample
                break
        
        if item is None:
            print(f"Audio file {filename} not found in dataset")
            return None
        
        # Get audio data
        audio_data = item['audio']
        
        # Save to local file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write audio data
        with open(output_path, 'wb') as f:
            f.write(audio_data['bytes'])
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}")
        return None

def load_hf_dataset_sample(repo_id: str, sample_id: str) -> Optional[dict]:
    """
    Load a specific sample from Hugging Face dataset.
    
    Args:
        repo_id: Hugging Face repository ID
        sample_id: ID of the sample to load
        
    Returns:
        Sample data if found, None otherwise
    """
    try:
        dataset = load_dataset(repo_id, split="train")
        
        for sample in dataset:
            if sample.get('id') == sample_id:
                return sample
        
        print(f"Sample {sample_id} not found in dataset")
        return None
        
    except Exception as e:
        print(f"Error loading sample {sample_id} from {repo_id}: {e}")
        return None

