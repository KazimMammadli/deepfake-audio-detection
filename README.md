# 🎙️ Deepfake Audio Detection with Explainable AI (XAI)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Author](https://img.shields.io/badge/Author-Kazim%20Mammadli-purple.svg)](https://github.com/KazimMammadli)

An end-to-end deep learning system and interactive web application for detecting AI-synthesized (deepfake) speech vs. genuine human speech from raw audio waveforms using Mel spectrogram analysis and state-of-the-art Explainable AI (Grad-CAM, SHAP, and LIME).

---

## 🌟 Key Features

- 🎧 **Audio Feature Engineering**: Automated conversion of arbitrary audio formats (WAV, MP3, FLAC, OGG) to 16 kHz mono waveforms and normalized log-scaled Mel spectrograms (80 × 201).
- 🧠 **Multiple Model Architectures**:
  - **Baseline CNN**: Custom 3-block convolutional neural network optimized for fast inference and lightweight deployment.
  - **MobileNetV2**: Transfer learning with inverted residual blocks for resource-constrained environments.
  - **ResNet50**: Deep residual network fine-tuned in two phases (frozen backbone head training → fine-tuning top layers).
- 🔬 **Comprehensive Explainability (XAI)**:
  - **Grad-CAM**: Gradient-weighted class activation mapping highlighting attention regions across time-frequency bins.
  - **SHAP**: Shapley Additive exPlanations measuring positive (fake) and negative (real) feature attributions.
  - **LIME**: Superpixel region perturbations explaining local decision boundaries.
- 🚀 **Full-Stack Interfaces**:
  - **Interactive Streamlit Web Application**: Audio upload, waveform/spectrogram rendering, real-time prediction, and explainability maps.
  - **CLI Tools**: Modular scripts for inference (`scripts/predict.py`), training (`scripts/train.py`), and test evaluation (`scripts/evaluate.py`).
- 🐳 **Containerized & Production Ready**: Docker & Docker Compose setup, calibrated decision thresholds, unit test suite, and clean Python package structure.

---

## 🏗️ System Architecture

```
Raw Audio (.wav / .mp3)
       │
       ▼
Audio Preprocessing (16 kHz, Mono, 2-sec window)
       │
       ▼
Mel Spectrogram Extraction (80 Mels, 201 Frames, Normalized [0, 1])
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
Model Inference                               Explainable AI (XAI)
┌───────────────────────────┐                 ┌──────────────────────────┐
│  • Baseline CNN           │                 │  • Grad-CAM Heatmap      │
│  • MobileNetV2 (Transfer) │ ──────────────> │  • SHAP Value Analysis   │
│  • ResNet50 (Transfer)    │                 │  • LIME Perturbations    │
└─────────────┬─────────────┘                 └─────────────┬────────────┘
              │                                             │
              ▼                                             ▼
    Prediction Output                             Visual Interpretability
   (Real vs. Fake + %)                           (Spectrogram Overlays)
```

---

## 📁 Repository Structure

```
deepfake-audio-detection/
├── app/                       # Streamlit web application
│   └── app.py
│
├── scripts/                   # Command-line tools & utilities
│   ├── train.py               # Model training CLI
│   ├── predict.py             # Single-file prediction & Grad-CAM CLI
│   ├── evaluate.py            # Test dataset evaluation CLI
│   └── precompute_mels.py     # Spectrogram precomputing utility
│
├── src/                       # Core Python package
│   ├── data/                  # FoR dataset loaders, builders & validators
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── features/              # Mel spectrogram extraction & audio DSP
│   │   └── mel_spectrogram.py
│   ├── models/                # Deep learning architectures
│   │   ├── baseline_cnn.py
│   │   ├── mobilenet.py
│   │   └── resnet.py
│   ├── training/              # Training routines & preprocessing adapters
│   │   ├── preprocessing.py
│   │   └── trainer.py
│   ├── evaluation/            # Metrics reporting & visualization helpers
│   │   ├── metrics.py
│   │   └── visualize.py
│   └── inference/             # Model loading, predict engine & XAI methods
│       ├── predictor.py
│       └── explainability.py
│
├── models/                    # Trained weights (.keras) & threshold configs
│   ├── baseline_cnn.keras
│   ├── mobilenetv2.keras
│   ├── resnet50_v3.keras
│   └── threshold_resnetv2.json
│
├── notebooks/                 # Exploratory & Colab-ready experiments
│   ├── 01_pipeline_validation.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_mobilenetv2.ipynb
│   └── 04_resnet50.ipynb
│
├── tests/                     # Automated unit test suite
│   ├── test_components.py
│   ├── test_audio.py
│   ├── test_models.py
│   └── test_inference.py
│
├── docs/                      # Extended technical documentation
│   ├── ARCHITECTURE.md        # Detailed system design & component diagrams
│   └── DEPLOYMENT.md          # Cloud deployment guide (AWS, GCP, Heroku, etc.)
│
├── .streamlit/                # Streamlit configuration & styling
│   └── config.toml
│
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Container service orchestration
├── pyproject.toml             # Standard Python packaging configuration
├── requirements.txt           # Python package dependencies
├── run_app.sh                 # Linux / macOS launcher script
├── run_app.bat                # Windows launcher script
├── LICENSE                    # MIT License
└── README.md                  # Master documentation
```

---

## 🗂️ Dataset: FoR (Fake-or-Real)

The models are trained and validated on the **FoR (Fake-or-Real)** speech dataset (`for-2seconds` subset):
- **Sample Rate**: 16,000 Hz (mono)
- **Duration**: 2.0 seconds per clip (32,000 samples)
- **Classes**: `0` = Real Human Speech, `1` = Synthesized (Fake) Speech
- **Class Balance**: 50% Real / 50% Fake across all splits

| Split | Real Clips | Fake Clips | Total Clips |
|---|---|---|---|
| **Training** | 6,978 | 6,978 | **13,956** |
| **Validation** | 1,413 | 1,413 | **2,826** |
| **Testing** | 544 | 544 | **1,088** |

### Expected Directory Layout
```
data/raw/for-2seconds/
├── training/
│   ├── real/   # *.wav files
│   └── fake/   # *.wav files
├── validation/
│   ├── real/
│   └── fake/
└── testing/
    ├── real/
    └── fake/
```

---

## ⚡ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- System audio libraries (`libsndfile1`, `ffmpeg`):
  ```bash
  # Ubuntu / Debian
  sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg

  # macOS
  brew install libsndfile ffmpeg
  ```

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/KazimMammadli/deepfake-audio-detection.git
cd deepfake-audio-detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Or install in editable mode as a package
pip install -e .
```

---

## 🚀 Usage

### 1. Interactive Streamlit Web App
Launch the interactive web user interface:
```bash
# Linux / macOS quick launch
./run_app.sh

# Windows quick launch
run_app.bat

# Direct command
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser. Upload any audio file to inspect the waveform, Mel spectrogram, prediction score, and Grad-CAM/SHAP/LIME explanation overlays.

### 2. Command-Line Inference (`scripts/predict.py`)
Run inference on a single audio sample directly from the terminal:
```bash
# Basic prediction
python scripts/predict.py path/to/sample.wav

# Predict with specific model and generate Grad-CAM attention heatmap
python scripts/predict.py path/to/sample.wav --model resnet50_v3 --explain --output-dir outputs/
```

### 3. Model Training (`scripts/train.py`)
Train models from scratch or fine-tune transfer learning architectures:
```bash
# Train Baseline CNN
python scripts/train.py --model baseline --epochs 20 --data-dir data/raw/for-2seconds

# Train MobileNetV2 with two-phase transfer learning
python scripts/train.py --model mobilenetv2 --epochs 10 --batch-size 32

# Train ResNet50
python scripts/train.py --model resnet50 --epochs 10 --batch-size 32
```

### 4. Model Evaluation (`scripts/evaluate.py`)
Evaluate a model checkpoint on the testing dataset:
```bash
python scripts/evaluate.py --model resnet50_v3 --model-path models/resnet50_v3.keras --data-dir data/raw/for-2seconds
```

### 5. Precomputing Mel Spectrograms
Precompute spectrograms to accelerate training iterations:
```bash
python scripts/precompute_mels.py --data-dir data/raw/for-2seconds --output-dir data/processed
```

---

## 🧪 Running Unit Tests

Run the test suite using Python unittest:
```bash
python -m unittest discover tests
```
Or with pytest:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Run the complete application inside a container:

```bash
# Using Docker Compose
docker-compose up -d

# Or building manually
docker build -t deepfake-audio-detector .
docker run -p 8501:8501 -v $(pwd)/models:/app/models:ro deepfake-audio-detector
```
Access the application at `http://localhost:8501`.

---

## ⚖️ Ethics & Responsible AI

Deepfake audio detection technology carries significant societal impact. This software is developed for research, education, and forensic analysis. The author recognizes that automated audio classifiers are not infallible; model outputs should not be used as the sole deciding factor in legal, investigative, or editorial conclusions without human verification. Downstream applications should always account for generalization boundaries across novel vocoders and synthesis techniques.

---

## 👤 Author

**Kazim Mammadli**  
- GitHub: [@KazimMammadli](https://github.com/KazimMammadli)  
- Repository: [deepfake-audio-detection](https://github.com/KazimMammadli/deepfake-audio-detection)

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).

