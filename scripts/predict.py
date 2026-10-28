#!/usr/bin/env python3
"""
predict.py — CLI tool to predict whether an audio clip is Real or Fake (Deepfake).

Usage:
  python predict.py path/to/sample.wav
  python predict.py path/to/sample.wav --model resnet50_v3 --explain
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_prediction(
    audio_path: str,
    model_name: str = "resnet50_v3",
    model_path: str = None,
    explain: bool = False,
    output_dir: str = "outputs",
):
    if model_path is None:
        model_path = os.path.join("models", f"{model_name}.keras")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at: {model_path}")
        sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at: {audio_path}")
        sys.exit(1)

    try:
        from src.inference.predictor import AudioPredictor, preprocess_mel_for_model
        from src.inference.explainability import get_gradcam_heatmap, plot_gradcam
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Error importing inference dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Loading model: {model_name} from {model_path}...")
    predictor = AudioPredictor(model_name, model_path)

    print(f"Analyzing audio: {audio_path}...")
    result = predictor.predict_file(audio_path)

    prob = result["probability"]
    label = result["label"]
    threshold = result["threshold"]

    print("\n" + "=" * 50)
    print("           DETECTION RESULT")
    print("=" * 50)
    print(f"  Prediction         : {label.upper()}")
    print(f"  Confidence         : {prob * 100:.2f}% (Fake Probability)")
    print(f"  Decision Threshold : {threshold:.4f}")
    print("=" * 50)

    if explain:
        os.makedirs(output_dir, exist_ok=True)
        print("\nGenerating Grad-CAM attention heatmap...")
        mel = result["mel"]
        x_input = preprocess_mel_for_model(mel, predictor.get_arch())
        try:
            heatmap = get_gradcam_heatmap(predictor.get_model(), x_input)
            fig = plot_gradcam(mel, heatmap, label, prob)
            save_dest = os.path.join(output_dir, f"gradcam_{Path(audio_path).stem}.png")
            fig.savefig(save_dest, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved Grad-CAM explanation to: {save_dest}")
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict deepfake audio from waveform")
    parser.add_argument("audio_path", type=str, help="Path to input audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50_v3",
        help="Model architecture name (default: resnet50_v3)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional direct path to .keras model file",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate and save Grad-CAM attention heatmap",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save explanation visuals (default: outputs)",
    )

    args = parser.parse_args()
    run_prediction(
        audio_path=args.audio_path,
        model_name=args.model,
        model_path=args.model_path,
        explain=args.explain,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
