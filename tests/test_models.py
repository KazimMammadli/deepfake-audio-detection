import unittest
import numpy as np

class TestModelArchitectures(unittest.TestCase):
    def test_baseline_cnn_build(self):
        try:
            import tensorflow as tf
            from src.models.baseline_cnn import build_baseline_cnn
            model = build_baseline_cnn(input_shape=(80, 201, 1))
            self.assertEqual(model.input_shape, (None, 80, 201, 1))
            self.assertEqual(model.output_shape, (None, 1))
        except ImportError:
            self.skipTest("TensorFlow not installed in test environment")

    def test_mobilenet_build(self):
        try:
            import tensorflow as tf
            from src.models.mobilenet import build_mobilenetv2, unfreeze_mobilenetv2
            model, base = build_mobilenetv2(weights=None, frozen=True)
            self.assertEqual(model.input_shape, (None, 224, 224, 3))
            self.assertFalse(base.trainable)
            unfreeze_mobilenetv2(base, unfreeze_layers=10)
            self.assertTrue(base.trainable)
        except ImportError:
            self.skipTest("TensorFlow not installed in test environment")

    def test_resnet_build(self):
        try:
            import tensorflow as tf
            from src.models.resnet import build_resnet50, unfreeze_resnet50
            model, base = build_resnet50(weights=None, frozen=True)
            self.assertEqual(model.input_shape, (None, 224, 224, 3))
            self.assertFalse(base.trainable)
            unfreeze_resnet50(base, unfreeze_layers=10)
            self.assertTrue(base.trainable)
        except ImportError:
            self.skipTest("TensorFlow not installed in test environment")

if __name__ == "__main__":
    unittest.main()
