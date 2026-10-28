"""src.data — Dataset loading and validation."""
from .dataset import FoRDataset, build_tf_datasets, build_arrays
from .preprocess import scan_dataset
__all__ = ["FoRDataset", "build_tf_datasets", "build_arrays", "scan_dataset"]
