#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained deepfake detection model on testing dataset.

Usage:
  python evaluate.py --model resnet50 --model-path models/resnet50_v3.keras --data-dir data/raw/for-2seconds
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_evaluation(
    model_path: str,
    model_name: str,
    data_dir: str,
    output_dir: str = "outputs",
    batch_size: int = 32,
    preset: str = "for-2sec",
):
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
        sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory does not exist: {data_dir}")
        sys.exit(1)

    try:
        import tensorflow as tf
        from src.features.mel_spectrogram import set_preset
        from src.data.dataset import build_tf_datasets
        from src.training.preprocessing import preprocess_for_model
        from src.evaluation.metrics import evaluate_model
        from src.inference.predictor import MODEL_ARCH, MODEL_THRESHOLDS
    except ImportError as e:
        print(f"Error importing evaluation dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    arch = MODEL_ARCH.get(model_name, "resnet50")
    threshold = MODEL_THRESHOLDS.get(model_name, 0.5)

    print(f"Loading model from {model_path} (architecture: {arch}, threshold: {threshold})...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Loading testing dataset from {data_dir}...")
    set_preset(preset)
    _, _, test_ds = build_tf_datasets(data_dir, preset=preset, batch_size=batch_size)

    test_ready = test_ds.map(
        lambda x, y: preprocess_for_model(x, y, arch),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    results = evaluate_model(
        model=model,
        test_ds=test_ready,
        model_name=model_name,
        output_dir=output_dir,
        threshold=threshold,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50_v3",
        help="Model identifier (resnet50_v3, mobilenetv2, baseline_cnn)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/for-2seconds",
        help="Path to dataset root",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    run_evaluation(
        model_path=args.model_path,
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
