#!/usr/bin/env python3
"""
train.py — Command-Line Interface for training Deepfake Audio Detection models.

Examples:
  python train.py --model baseline --epochs 20
  python train.py --model mobilenetv2 --epochs 10 --batch-size 32
  python train.py --model resnet50 --epochs 10 --data-dir data/raw/for-2seconds
"""

import argparse
import sys
from pathlib import Path

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Audio Detection Models")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["baseline", "mobilenetv2", "resnet50", "all"],
        help="Architecture to train",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/for-2seconds",
        help="Path to FoR dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation plots and logs",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save .keras checkpoint files",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for training / fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")

    args = parser.parse_args()

    try:
        from src.training.trainer import train_baseline_model, train_transfer_model
    except ImportError as e:
        print(f"Error loading training dependencies: {e}")
        print("Please ensure dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    if args.model == "baseline":
        train_baseline_model(
            dataset_dir=args.data_dir,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    elif args.model in ("mobilenetv2", "resnet50"):
        train_transfer_model(
            model_name=args.model,
            dataset_dir=args.data_dir,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            phase1_epochs=args.epochs,
            phase2_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    elif args.model == "all":
        train_baseline_model(dataset_dir=args.data_dir, output_dir=args.output_dir, models_dir=args.models_dir)
        train_transfer_model("mobilenetv2", dataset_dir=args.data_dir, output_dir=args.output_dir, models_dir=args.models_dir)
        train_transfer_model("resnet50", dataset_dir=args.data_dir, output_dir=args.output_dir, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
