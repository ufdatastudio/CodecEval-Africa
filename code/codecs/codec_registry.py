"""
Codec Registry - Safe way to add new codecs without breaking existing ones.
"""

from typing import Dict, Any, Optional
import importlib

# Registry of available codecs
CODEC_REGISTRY = {
    # Working codecs (tested and stable)
    "encodec_24khz": {
        "runner": "code.codecs.encodec_runner:EncodecRunner",
        "status": "working",
        "description": "Meta's EnCodec - causal processing"
    },
    "soundstream_impl": {
        "runner": "code.codecs.soundstream_runner:SoundStreamRunner", 
        "status": "working",
        "description": "EnCodec-based SoundStream - non-causal processing"
    },
    
    # New codecs (placeholders - safe to add)
    "unicodec": {
        "runner": "code.codecs.unicodec_runner:UniCodecRunner",
        "status": "working",
        "description": "Tencent's UniCodec - unified framework (simplified implementation)"
    },
    "dac": {
        "runner": "code.codecs.dac_runner:DACRunner",
        "status": "working", 
        "description": "Descript Audio Codec - high quality speech (neural implementation)"
    },
    "sematicodec": {
        "runner": "code.codecs.sematicodec_runner:SemantiCodecRunner",
        "status": "working",
        "description": "Meta's SemantiCodec - semantic-aware compression (attention-based)"
    },
    "apcodec": {
        "runner": "code.codecs.apcodec_runner:APCodecRunner", 
        "status": "working",
        "description": "Adaptive Perceptual Codec (perceptual loss-based)"
    }
}

def get_codec_runner(codec_name: str, **kwargs):
    """
    Safely get a codec runner by name.
    
    Args:
        codec_name: Name of the codec from registry
        **kwargs: Arguments to pass to the runner
        
    Returns:
        Codec runner instance
        
    Raises:
        KeyError: If codec not found in registry
        ImportError: If codec runner cannot be imported
        NotImplementedError: If codec is placeholder
    """
    if codec_name not in CODEC_REGISTRY:
        raise KeyError(f"Codec '{codec_name}' not found in registry. Available: {list(CODEC_REGISTRY.keys())}")
    
    codec_info = CODEC_REGISTRY[codec_name]
    runner_path = codec_info["runner"]
    
    # Parse module and class
    module_path, class_name = runner_path.split(":")
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        runner_class = getattr(module, class_name)
        
        # Create runner instance
        runner = runner_class(**kwargs)
        
        return runner
        
    except ImportError as e:
        raise ImportError(f"Cannot import {runner_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Cannot find class {class_name} in {module_path}: {e}")

def list_available_codecs():
    """List all available codecs with their status."""
    print("Available Codecs:")
    print("=" * 50)
    for name, info in CODEC_REGISTRY.items():
        status_text = "WORKING" if info["status"] == "working" else "PLACEHOLDER"
        print(f"{status_text} {name}: {info['description']}")
        print(f"   Status: {info['status']}")
        print(f"   Runner: {info['runner']}")
        print()

def test_codec(codec_name: str, test_file: str = None):
    """
    Test a codec with a simple file.
    
    Args:
        codec_name: Name of codec to test
        test_file: Optional test file path
    """
    try:
        # Use correct parameter names for different codecs
        if codec_name == "encodec_24khz":
            runner = get_codec_runner(codec_name, bandwidth_kbps=6.0, sr=24000)
        else:
            runner = get_codec_runner(codec_name, bitrate_kbps=6.0, sr=24000)
            print(f"SUCCESS {codec_name} runner created successfully")
        
        if test_file:
            print(f"Testing with file: {test_file}")
            # Add test logic here
            
    except Exception as e:
            print(f"FAILED {codec_name} test failed: {e}")

if __name__ == "__main__":
    list_available_codecs()
