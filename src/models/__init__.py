"""
src.models — Model architectures for Deepfake Audio Detection.

Exposes:
  - build_baseline_cnn
  - build_mobilenetv2, unfreeze_mobilenetv2
  - build_resnet50, unfreeze_resnet50
  - get_model (factory function)
"""

from .baseline_cnn import build_baseline_cnn
from .mobilenet import build_mobilenetv2, unfreeze_mobilenetv2
from .resnet import build_resnet50, unfreeze_resnet50


def get_model(name: str, **kwargs):
    """
    Factory helper to instantiate a model architecture by string name.

    Supported names:
      - "baseline_cnn", "custom", "baseline"
      - "mobilenetv2", "mobilenet_v2"
      - "resnet50", "resnet"
    """
    name_clean = name.lower().replace("-", "_")
    if name_clean in ("baseline_cnn", "custom", "baseline"):
        return build_baseline_cnn(**kwargs)
    elif name_clean in ("mobilenetv2", "mobilenet_v2", "mobilenet"):
        return build_mobilenetv2(**kwargs)
    elif name_clean in ("resnet50", "resnet"):
        return build_resnet50(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{name}'. Supported: baseline_cnn, mobilenetv2, resnet50"
        )


__all__ = [
    "build_baseline_cnn",
    "build_mobilenetv2",
    "unfreeze_mobilenetv2",
    "build_resnet50",
    "unfreeze_resnet50",
    "get_model",
]
