import unittest
from pathlib import Path

class TestComponents(unittest.TestCase):
    def test_structure(self):
        req_files = [
            "app/app.py",
            "scripts/train.py",
            "scripts/predict.py",
            "scripts/evaluate.py",
            "scripts/precompute_mels.py",
            "requirements.txt",
            "pyproject.toml",
            "LICENSE",
            "src/inference/predictor.py",
            "src/inference/explainability.py",
            "src/models/baseline_cnn.py",
            "src/models/mobilenet.py",
            "src/models/resnet.py",
            "src/training/trainer.py",
        ]
        for f in req_files:
            self.assertTrue(Path(f).exists(), f"Missing file: {f}")

    def test_models_dir(self):
        self.assertTrue(Path("models").exists())

if __name__ == "__main__":
    unittest.main()
