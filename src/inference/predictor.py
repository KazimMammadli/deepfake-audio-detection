"""
predictor.py — Single-file inference engine for all three model architectures.

Handles:
  - Loading .keras models
  - Audio → mel spectrogram preprocessing (matches training pipeline exactly)
  - Model-specific input reshaping
  - Returning probability + binary prediction
"""

import json
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path

# ── Mel spectrogram config (must match training preset "for-2sec") ─────────────
MEL_CFG = dict(
    sample_rate   = 16000,
    n_fft         = 512,
    hop_length    = 160,
    win_length    = 400,
    n_mels        = 80,
    f_min         = 20.0,
    f_max         = 8000.0,
    target_frames = 201,
)

# Model name → architecture type mapping
MODEL_ARCH = {
    "baseline_cnn":        "custom",
    "mobilenetv2":         "mobilenet_v2",
    "mobilenetv2_phase1":  "mobilenet_v2",
    "resnet50":            "resnet50",
    "resnet50_phase1":     "resnet50",
    "resnet50_phase1_v2":  "resnet50",
    "resnet50_phase1_v3":  "resnet50",
    "resnet50_v2":         "resnet50",
    "resnet50_v3":         "resnet50",
}

# Per-model thresholds (override 0.5 where a calibrated threshold exists)
MODEL_THRESHOLDS = {
    "resnet50_v2": 0.2116357386112213,
}


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file, convert stereo→mono, return (waveform, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample to 16 kHz if needed."""
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    """
    Convert raw waveform to mel spectrogram matching the training pipeline.

    Returns shape: (80, 201)  float32  values in [0, 1]
    """
    cfg = MEL_CFG
    mel = librosa.feature.melspectrogram(
        y          = audio,
        sr         = cfg["sample_rate"],
        n_fft      = cfg["n_fft"],
        hop_length = cfg["hop_length"],
        win_length = cfg["win_length"],
        n_mels     = cfg["n_mels"],
        fmin       = cfg["f_min"],
        fmax       = cfg["f_max"],
        power      = 2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    # Pad or trim to fixed target_frames
    target = cfg["target_frames"]
    if mel_db.shape[1] < target:
        mel_db = np.pad(mel_db, ((0, 0), (0, target - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target]

    # Normalize to [0, 1]
    lo, hi = mel_db.min(), mel_db.max()
    if hi > lo:
        mel_db = (mel_db - lo) / (hi - lo)

    return mel_db.astype(np.float32)  # (80, 201)


def preprocess_mel_for_model(mel: np.ndarray, arch: str) -> np.ndarray:
    """
    Convert (80, 201) mel to model-specific input shape.

    custom       → (1, 80, 201, 1)    channels-last
    mobilenet_v2 → (1, 224, 224, 3)   pseudo-RGB, MobileNetV2 normalised
    resnet50     → (1, 224, 224, 3)   pseudo-RGB, ResNet50 normalised
    """
    # (80, 201) → (80, 201, 1)
    x = mel[:, :, np.newaxis]

    if arch == "custom":
        return x[np.newaxis, ...]  # (1, 80, 201, 1)

    # Grayscale → pseudo-RGB
    x = np.repeat(x, 3, axis=-1)  # (80, 201, 3)

    # Resize to 224×224
    x_tensor = tf.image.resize(x[np.newaxis, ...], [224, 224])  # (1, 224, 224, 3)

    if arch == "mobilenet_v2":
        x_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(x_tensor)
    elif arch == "resnet50":
        x_tensor = tf.keras.applications.resnet50.preprocess_input(x_tensor)

    return x_tensor.numpy()  # (1, 224, 224, 3)


class AudioPredictor:
    """
    Wraps a loaded Keras model and exposes a clean predict() interface.

    Usage:
        predictor = AudioPredictor("resnet50_v2", "models/resnet50_v2.keras")
        result = predictor.predict_file("audio.wav")
        # result: {"probability": 0.83, "label": "Fake", "threshold": 0.21}
    """

    def __init__(self, model_key: str, model_path: str):
        self.model_key  = model_key
        self.arch       = MODEL_ARCH.get(model_key, "resnet50")
        self.threshold  = MODEL_THRESHOLDS.get(model_key, 0.5)
        self.model      = tf.keras.models.load_model(model_path)

    def predict_from_audio(self, audio: np.ndarray, sr: int) -> dict:
        """
        Run prediction from a waveform array.

        Returns dict with keys:
          probability, label, threshold, mel (80×201 array)
        """
        audio = resample_audio(audio, sr)
        mel   = audio_to_mel(audio)
        x     = preprocess_mel_for_model(mel, self.arch)

        prob  = float(self.model.predict(x, verbose=0).flatten()[0])
        label = "Fake" if prob >= self.threshold else "Real"

        return {
            "probability": prob,
            "label":       label,
            "threshold":   self.threshold,
            "mel":         mel,
            "audio":       audio,
        }

    def predict_file(self, path: str) -> dict:
        audio, sr = load_audio(path)
        return self.predict_from_audio(audio, sr)

    def get_model(self):
        return self.model

    def get_arch(self):
        return self.arch
