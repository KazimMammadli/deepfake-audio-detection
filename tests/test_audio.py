import unittest
import numpy as np

class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        try:
            import librosa
            import soundfile
        except ImportError:
            self.skipTest("librosa or soundfile not installed in environment")

        from src.features.mel_spectrogram import extract, set_preset, PRESETS, _cfg
        self.extract = extract
        self.set_preset = set_preset
        self.set_preset("for-2sec")

        # Generate 2 seconds of synthetic audio (sine wave 440 Hz)
        self.sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        self.audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_mel_shape(self):
        mel = self.extract(self.audio)
        self.assertEqual(mel.shape, (80, 201))
        self.assertEqual(mel.dtype, np.float32)

    def test_mel_normalization(self):
        mel = self.extract(self.audio)
        self.assertGreaterEqual(mel.min(), 0.0)
        self.assertLessEqual(mel.max(), 1.0)

    def test_short_audio_padding(self):
        short_audio = self.audio[:8000]  # 0.5 sec
        mel = self.extract(short_audio)
        self.assertEqual(mel.shape, (80, 201))

    def test_long_audio_trimming(self):
        long_audio = np.tile(self.audio, 2)  # 4 sec
        mel = self.extract(long_audio)
        self.assertEqual(mel.shape, (80, 201))

if __name__ == "__main__":
    unittest.main()
