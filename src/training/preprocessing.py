"""
preprocessing.py — Pipeline adapter for model-specific input formats.

The FoR pipeline outputs channels-first tensors: (batch, 1, n_mels, target_frames).
Each model architecture expects a different shape. This module provides a single
adapter function used by all training notebooks.

Usage:
    from src.training.preprocessing import preprocess_for_model

    train_ds_ready = train_ds.map(
        lambda x, y: preprocess_for_model(x, y, "mobilenet_v2"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
"""

import tensorflow as tf


def preprocess_for_model(x, y, model_name: str):
    """
    Convert pipeline channels-first tensor to the shape each model expects.

    Transformation summary
    ----------------------
    'custom'      : (batch, 1, 80, T) → transpose → (batch, 80, T, 1)
    'mobilenet_v2': (batch, 1, 80, T) → transpose → repeat 3× → resize 224 →
                    mobilenet_v2.preprocess_input  →  (batch, 224, 224, 3)
    'resnet50'    : same steps as mobilenet_v2 but uses resnet50.preprocess_input

    Args
    ----
    x          : tf.Tensor  shape (batch, 1, n_mels, target_frames)  float32
    y          : tf.Tensor  shape (batch,)  int32
    model_name : str  one of  'custom' | 'mobilenet_v2' | 'resnet50'

    Returns
    -------
    (x_processed, y)
    """
    # Step 1 — channels-first → channels-last
    # (batch, 1, n_mels, target_frames) → (batch, n_mels, target_frames, 1)
    x = tf.transpose(x, perm=[0, 2, 3, 1])

    if model_name == "custom":
        # No further change needed — (batch, n_mels, target_frames, 1)
        return x, y

    if model_name in ("mobilenet_v2", "resnet50"):
        # Step 2 — grayscale → pseudo-RGB
        x = tf.repeat(x, repeats=3, axis=-1)        # (..., 3)

        # Step 3 — resize to ImageNet standard resolution
        x = tf.image.resize(x, [224, 224])           # (batch, 224, 224, 3)

        # Step 4 — model-specific pixel normalisation
        if model_name == "mobilenet_v2":
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        else:
            x = tf.keras.applications.resnet50.preprocess_input(x)

        return x, y

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        "Choose one of: 'custom' | 'mobilenet_v2' | 'resnet50'"
    )
