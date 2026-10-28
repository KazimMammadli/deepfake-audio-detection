"""
mobilenet.py — MobileNetV2 transfer learning model for deepfake audio detection.

Pretrained on ImageNet, adapted for 3-channel pseudo-RGB Mel spectrogram inputs
resized to (224, 224, 3).
"""

from typing import Tuple, Optional
from tensorflow import keras


def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: Optional[str] = "imagenet",
    dropout_rate: float = 0.3,
    frozen: bool = True,
    name: str = "mobilenetv2_deepfake",
) -> Tuple[keras.Model, keras.Model]:
    """
    Builds MobileNetV2 backbone with custom binary classification head.

    Parameters:
      input_shape: Shape of pseudo-RGB spectrogram tensor (224, 224, 3).
      weights: Pretrained weights to load (default: "imagenet").
      dropout_rate: Dropout probability before output dense layer.
      frozen: If True, freezes the base model backbone for Phase 1 head training.
      name: Model name.

    Returns:
      (full_model, base_model)
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = not frozen

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model, base_model


def unfreeze_mobilenetv2(base_model: keras.Model, unfreeze_layers: int = 30) -> None:
    """
    Unfreezes the top N layers of the MobileNetV2 base model for Phase 2 fine-tuning.

    Parameters:
      base_model: The underlying MobileNetV2 feature extractor.
      unfreeze_layers: Number of final layers to unfreeze (default: 30).
    """
    base_model.trainable = True
    if unfreeze_layers > 0:
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
