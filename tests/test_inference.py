import unittest
import numpy as np

class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        try:
            import tensorflow
            import librosa
        except ImportError:
            self.skipTest("tensorflow or librosa not installed in environment")

        from src.inference.predictor import preprocess_mel_for_model, MODEL_ARCH, MODEL_THRESHOLDS
        self.preprocess_mel_for_model = preprocess_mel_for_model
        self.MODEL_ARCH = MODEL_ARCH
        self.MODEL_THRESHOLDS = MODEL_THRESHOLDS
        self.mel = np.random.uniform(0, 1, (80, 201)).astype(np.float32)

    def test_preprocess_custom(self):
        out = self.preprocess_mel_for_model(self.mel, "custom")
        self.assertEqual(out.shape, (1, 80, 201, 1))

    def test_model_arch_mappings(self):
        self.assertIn("baseline_cnn", self.MODEL_ARCH)
        self.assertIn("mobilenetv2", self.MODEL_ARCH)
        self.assertIn("resnet50_v3", self.MODEL_ARCH)

if __name__ == "__main__":
    unittest.main()
