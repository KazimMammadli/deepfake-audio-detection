"""
trainer.py — Modular training pipeline for Deepfake Audio Detection models.

Provides standardized training routines for:
  - Baseline 3-block CNN
  - MobileNetV2 (two-phase transfer learning)
  - ResNet50 (two-phase transfer learning)

Handles preprocessing pipeline mapping, optimizer setup, learning rate schedules,
early stopping, model checkpoints, and evaluation callbacks.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from src.features.mel_spectrogram import set_preset, _cfg
from src.data.dataset import build_tf_datasets
from src.training.preprocessing import preprocess_for_model
from src.models import build_baseline_cnn, build_mobilenetv2, build_resnet50, unfreeze_mobilenetv2, unfreeze_resnet50
from src.evaluation.visualize import plot_history
from src.evaluation.metrics import evaluate_model


def get_preprocessed_datasets(
    dataset_dir: str,
    arch_type: str,
    batch_size: int = 32,
    preset: str = "for-2sec",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Build and adapt TensorFlow datasets for the target architecture.
    """
    set_preset(preset)
    train_ds, val_ds, test_ds = build_tf_datasets(
        dataset_dir, preset=preset, batch_size=batch_size
    )

    train_ready = train_ds.map(
        lambda x, y: preprocess_for_model(x, y, arch_type),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ready = val_ds.map(
        lambda x, y: preprocess_for_model(x, y, arch_type),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ready = test_ds.map(
        lambda x, y: preprocess_for_model(x, y, arch_type),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return train_ready, val_ready, test_ready


def train_baseline_model(
    dataset_dir: str,
    output_dir: str = "outputs",
    models_dir: str = "models",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 3,
) -> Dict[str, Any]:
    """
    Train custom 3-block Baseline CNN from scratch.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"=== [Baseline CNN] Loading data from {dataset_dir} ===")
    train_ds, val_ds, test_ds = get_preprocessed_datasets(
        dataset_dir, arch_type="custom", batch_size=batch_size
    )

    input_shape = (_cfg["n_mels"], _cfg["target_frames"], 1)
    model = build_baseline_cnn(input_shape=input_shape)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(models_dir, "baseline_cnn.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("=== [Baseline CNN] Starting Training ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    curve_path = os.path.join(output_dir, "baseline_cnn_curves.png")
    plot_history(history, title="Baseline CNN — Training History", save_path=curve_path)

    print("=== [Baseline CNN] Evaluating on Test Set ===")
    results = evaluate_model(
        model, test_ds, model_name="Baseline CNN", cmap="Blues", output_dir=output_dir
    )
    return {"model": model, "history": history, "results": results}


def train_transfer_model(
    model_name: str,
    dataset_dir: str,
    output_dir: str = "outputs",
    models_dir: str = "models",
    phase1_epochs: int = 10,
    phase2_epochs: int = 10,
    batch_size: int = 32,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-5,
    unfreeze_layers: int = 30,
    patience: int = 3,
) -> Dict[str, Any]:
    """
    Train transfer learning models (MobileNetV2 or ResNet50) in two phases:
      Phase 1: Freeze backbone, train classification head.
      Phase 2: Unfreeze top N layers of backbone, fine-tune at low learning rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_clean = model_name.lower().replace("-", "_")
    if "mobilenet" in model_clean:
        arch_type = "mobilenet_v2"
        display_name = "MobileNetV2"
        builder = build_mobilenetv2
        unfreezer = unfreeze_mobilenetv2
        cmap = "Greens"
        save_prefix = "mobilenetv2"
    elif "resnet" in model_clean:
        arch_type = "resnet50"
        display_name = "ResNet50"
        builder = build_resnet50
        unfreezer = unfreeze_resnet50
        cmap = "Oranges"
        save_prefix = "resnet50"
    else:
        raise ValueError(f"Unsupported transfer model: {model_name}")

    print(f"=== [{display_name}] Loading data from {dataset_dir} ===")
    train_ds, val_ds, test_ds = get_preprocessed_datasets(
        dataset_dir, arch_type=arch_type, batch_size=batch_size
    )

    # Phase 1: Build frozen model
    model, base_model = builder(frozen=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase1_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    phase1_ckpt = os.path.join(models_dir, f"{save_prefix}_phase1.keras")
    callbacks_p1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=phase1_ckpt, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print(f"=== [{display_name}] Phase 1: Training Classification Head ===")
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks_p1,
        verbose=1,
    )
    plot_history(
        history_p1,
        title=f"{display_name} — Phase 1 (Frozen Base)",
        save_path=os.path.join(output_dir, f"{save_prefix}_phase1_curves.png"),
    )

    # Phase 2: Unfreeze top layers and fine-tune
    print(f"=== [{display_name}] Phase 2: Fine-tuning last {unfreeze_layers} layers ===")
    unfreezer(base_model, unfreeze_layers=unfreeze_layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase2_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    final_ckpt = os.path.join(models_dir, f"{save_prefix}.keras")
    callbacks_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=final_ckpt, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=callbacks_p2,
        verbose=1,
    )
    plot_history(
        history_p2,
        title=f"{display_name} — Phase 2 (Fine-Tuning)",
        save_path=os.path.join(output_dir, f"{save_prefix}_phase2_curves.png"),
    )

    print(f"=== [{display_name}] Final Test Evaluation ===")
    results = evaluate_model(
        model, test_ds, model_name=display_name, cmap=cmap, output_dir=output_dir
    )

    return {
        "model": model,
        "history_phase1": history_p1,
        "history_phase2": history_p2,
        "results": results,
    }


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
