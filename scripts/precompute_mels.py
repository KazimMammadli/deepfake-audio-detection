#!/usr/bin/env python3
"""
precompute_mels.py — Precompute and cache Mel spectrograms for rapid iteration.

Loads waveforms from raw data splits, computes Mel spectrograms, and saves
batched numpy arrays (.npy) into data/processed/.
"""

import os
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def precompute(dataset_dir: str, output_dir: str = "data/processed", preset: str = "for-2sec"):
    try:
        import numpy as np
        from src.data.dataset import FoRDataset
        from src.features.mel_spectrogram import set_preset
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    set_preset(preset)

    for split in ["training", "validation", "testing"]:
        print(f"\nProcessing {split} split from {dataset_dir}...")
        try:
            ds = FoRDataset(dataset_dir, split=split)
        except Exception as e:
            print(f"Skipping {split}: {e}")
            continue

        X, y = [], []
        total = len(ds)
        for i in range(total):
            mel, label = ds.get_item(i)
            X.append(mel)
            y.append(label)
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"  Processed {i + 1}/{total} samples")

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Shuffle
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        out_x = os.path.join(output_dir, f"{split}_X.npy")
        out_y = os.path.join(output_dir, f"{split}_y.npy")
        np.save(out_x, X)
        np.save(out_y, y)
        print(f"Saved {split}: X={X.shape}, y={y.shape} -> {out_x}, {out_y} ✅")


def main():
    parser = argparse.ArgumentParser(description="Precompute Mel spectrograms to disk")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/for-2seconds",
        help="Path to raw dataset folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Destination for .npy arrays",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="for-2sec",
        help="Mel spectrogram extraction preset",
    )
    args = parser.parse_args()
    precompute(args.data_dir, args.output_dir, args.preset)


if __name__ == "__main__":
    main()
