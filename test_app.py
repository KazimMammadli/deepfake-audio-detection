"""
test_app.py - Quick test script to verify the app components work

Run this before starting the Streamlit app to check for issues.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import streamlit
        import tensorflow
        import librosa
        import soundfile
        import numpy
        import matplotlib
        import lime
        import shap
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nRun: pip install -r requirements.txt")
        return False


def test_project_structure():
    """Test that required files and directories exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "src/inference/predictor.py",
        "src/inference/explainability.py",
    ]
    
    required_dirs = [
        "src",
        "src/inference",
        "models",
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} NOT FOUND")
            all_good = False
    
    for dir in required_dirs:
        if Path(dir).exists():
            print(f"✓ {dir}/")
        else:
            print(f"✗ {dir}/ NOT FOUND")
            all_good = False
    
    return all_good


def test_models():
    """Check if model files exist."""
    print("\nTesting model files...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("✗ models/ directory not found")
        return False
    
    model_files = list(models_dir.glob("*.keras"))
    
    if len(model_files) == 0:
        print("✗ No .keras model files found in models/")
        print("  Please train models first (see notebooks) or download pre-trained models")
        return False
    
    print(f"✓ Found {len(model_files)} model files:")
    for model in model_files:
        print(f"  - {model.name}")
    
    return True


def test_inference():
    """Test that inference components can be imported."""
    print("\nTesting inference components...")
    try:
        from src.inference.predictor import AudioPredictor, audio_to_mel
        from src.inference.explainability import (
            get_gradcam_heatmap,
            plot_gradcam,
            generate_lime_explanation,
            generate_shap_explanation
        )
        print("✓ Inference components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import inference components: {e}")
        return False


def test_tensorflow():
    """Test TensorFlow functionality."""
    print("\nTesting TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if GPU is available (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
        else:
            print("ℹ No GPU found (CPU will be used)")
        
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False


def test_audio_processing():
    """Test audio processing libraries."""
    print("\nTesting audio processing...")
    try:
        import librosa
        import soundfile
        import numpy as np
        
        # Create a simple sine wave
        sr = 16000
        duration = 2
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Test mel spectrogram extraction
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=512,
            hop_length=160,
            n_mels=80
        )
        
        print(f"✓ Audio processing works")
        print(f"  Created test audio: {duration}s at {sr}Hz")
        print(f"  Mel spectrogram shape: {mel.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing test failed: {e}")
        return False


def main():
    print("="*50)
    print("Deepfake Audio Detection - System Test")
    print("="*50)
    print()
    
    tests = [
        test_imports,
        test_project_structure,
        test_models,
        test_inference,
        test_tensorflow,
        test_audio_processing,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("="*50)
    print("Test Summary")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed! You can run the app with:")
        print("  streamlit run app.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        if not results[2]:  # models test
            print("\nNote: Model files are required to run the app.")
            print("Train them using the notebooks or download pre-trained models.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
