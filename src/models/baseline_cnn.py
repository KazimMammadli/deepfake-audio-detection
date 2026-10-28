"""
baseline_cnn.py — Custom 3-block Convolutional Neural Network architecture.

A lightweight convolutional architecture designed to classify Mel spectrograms
extracted from 2-second audio clips as either Real (0) or Fake (1).
"""

from typing import Tuple
from tensorflow import keras


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (80, 201, 1),
    dropout_rate: float = 0.3,
    name: str = "baseline_cnn",
) -> keras.Model:
    """
    Constructs a 3-block custom CNN with Global Average Pooling.

    Architecture:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> MaxPool2D
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> MaxPool2D
      - Block 3: Conv2D(128, 3x3) -> BatchNorm -> MaxPool2D
      - Head: GlobalAveragePooling2D -> Dropout -> Dense(1, sigmoid)

    Parameters:
      input_shape: Tuple specifying (height, width, channels) of input spectrograms.
      dropout_rate: Fraction of input units to drop before the final classification layer.
      name: Model name.

    Returns:
      A tf.keras.Model instance.
    """
    inputs = keras.Input(shape=input_shape, name="mel_input")

    # Block 1
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.MaxPooling2D(name="pool1")(x)

    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.MaxPooling2D(name="pool2")(x)

    # Block 3
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.MaxPooling2D(name="pool3")(x)

    # Classification Head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
