# 🎙️ Deepfake Audio Detection

> A deep-learning pipeline that detects AI-generated (deepfake) speech from real human speech using Mel spectrogram analysis and CNN-based classifiers.

---

## 👥 Team — Holberton School Capstone

| Name | GitHub |
|------|--------|
| Kazim Mammadli | [@kazimmammadli](https://github.com/kazimmammadli) |
| Vidadi Javadov | [@VidadiJavadov](https://github.com/VidadiJavadov) |
| Zamin Sultanli | [@zaminsultanli](https://github.com/zaminsultanli) |

---

## 📁 Project Structure

```
deepfake-audio-detection/
│
├── notebooks/
│   ├── 01_pipeline_validation.ipynb   ← Data pipeline sanity checks
│   ├── 02_baseline_model.ipynb        ← 3-block custom CNN (baseline)
│   ├── 03_mobilenetv2.ipynb           ← MobileNetV2 transfer learning
│   └── 04_resnet50.ipynb              ← ResNet50 transfer learning
│
├── src/
│   ├── data/
│   │   ├── dataset.py                 ← FoRDataset + build_tf_datasets()
│   │   └── preprocess.py              ← Metadata CSV builder & validator
│   ├── features/
│   │   └── mel_spectrogram.py         ← Mel spectrogram extraction (presets)
│   ├── models/
│   │   └── baseline_cnn.py            ← (reserved for model classes)
│   ├── training/
│   │   ├── trainer.py                 ← (reserved for training scripts)
│   │   └── preprocessing.py           ← Model-specific input adapters
│   ├── evaluation/
│   │   ├── metrics.py                 ← Evaluation utilities
│   │   └── visualize.py               ← Plotting functions
│   └── inference/
│       ├── predictor.py               ← Single-file inference engine
│       ├── explainability.py          ← GradCAM, LIME, SHAP
│       └── __init__.py
│
├── models/
│   └── *.keras                        ← Trained model weights
│
├── app.py                             ← 🎯 Streamlit web application
├── test_app.py                        ← App component tests
├── run_app.bat                        ← Windows launcher
├── run_app.sh                         ← Linux/Mac launcher
├── Dockerfile                         ← Docker container config
├── docker-compose.yml                 ← Docker Compose config
├── .streamlit/config.toml             ← Streamlit configuration
├── DEPLOYMENT.md                      ← Deployment guide
├── README_APP.md                      ← App documentation
├── outputs/                           ← Saved plots (git-ignored)
├── data/                              ← Raw & processed datasets (git-ignored)
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**FoR (Fake-or-Real)** — `for-2seconds` subset  
- Audio: 16 kHz, mono, 2-second clips  
- Labels: `0` = Real · `1` = Fake  
- Class balance: perfectly 50 / 50 across all splits  

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Training | 6,978 | 6,978 | **13,956** |
| Validation | 1,413 | 1,413 | **2,826** |
| Testing | 544 | 544 | **1,088** |

Place the dataset at:
```
data/raw/for-2seconds/
    training/real/   *.wav
    training/fake/   *.wav
    validation/real/ *.wav
    validation/fake/ *.wav
    testing/real/    *.wav
    testing/fake/    *.wav
```

---

## ⚙️ Setup

```bash
git clone https://github.com/VidadiJavadov/deepfake-audio-detection.git
cd deepfake-audio-detection
pip install -r requirements.txt
```

**Tech stack:** Python 3.10+ · TensorFlow 2.13+ · Librosa · SoundFile · NumPy · scikit-learn · Matplotlib · SHAP · LIME · Streamlit  
**Target runtime:** Google Colab (GPU for training, CPU for inference)

---

## 🚀 Run Order

### Training Models (Notebooks)

Open notebooks in Colab in this order:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_pipeline_validation.ipynb` | Verify dataset, mel extraction, TF data loader |
| 2 | `02_baseline_model.ipynb` | Train 3-block custom CNN (sanity check) |
| 3 | `03_mobilenetv2.ipynb` | MobileNetV2 transfer learning (2-phase) |
| 4 | `04_resnet50.ipynb` | ResNet50 transfer learning (2-phase) |

**Running in Colab:**
```python
# Cell 1 — clone repo (run once per session)
!git clone https://github.com/VidadiJavadov/deepfake-audio-detection.git
%cd deepfake-audio-detection
!pip install -r requirements.txt
```

### 🎯 Streamlit Web App (New!)

**Quick Start:**
```bash
# Windows
run_app.bat

# Linux/Mac
chmod +x run_app.sh
./run_app.sh

# Or directly
streamlit run app.py
```

**Features:**
- 🎙️ Upload audio files (WAV, MP3, FLAC, OGG)
- 🔮 Real-time deepfake detection
- 📊 Interactive visualizations (waveform, mel spectrogram)
- 🔥 **GradCAM** heatmap (always shown)
- 🔍 **LIME** explanation (optional)
- 📈 **SHAP** values (optional)
- 🎯 Multiple model selection

**See [README_APP.md](README_APP.md) for full app documentation**  
**See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guide**

---

## 📊 Results

> Results will be updated after training runs complete.

| Model | Accuracy | F1-Score | AUC-ROC | Inference (ms/clip) |
|-------|----------|----------|---------|---------------------|
| Baseline CNN | TBD | TBD | TBD | TBD |
| MobileNetV2 | TBD | TBD | TBD | TBD |
| ResNet50 | TBD | TBD | TBD | TBD |

---

## ⚖️ Ethics Note

Deepfake audio detection technology carries significant societal responsibility. This system is trained exclusively on the publicly available FoR (Fake-or-Real) dataset and is intended purely for research and educational purposes as part of a Holberton School capstone project. The authors recognise that no detection system is infallible, and that over-reliance on automated tools could result in false accusations or the suppression of legitimate speech. This model should never be used as sole evidence in legal, journalistic, or forensic contexts. Deepfake-generation and detection technology evolves rapidly; models trained on today's data may not generalise to novel synthesis methods. We encourage transparent disclosure of model limitations and responsible deployment practices in any downstream application.


---

## 🎯 Streamlit Web App

After training models, use the interactive web app for inference:

### Quick Start

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Manual:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Features

- 🎛️ **Model Selection**: Choose from 9 trained models
- 📤 **Audio Upload**: Support for WAV, MP3, FLAC, OGG
- 🌊 **Waveform Visualization**: See the audio signal
- 🎵 **Mel Spectrogram**: Time-frequency representation
- 🔥 **GradCAM**: Highlights model attention regions (always shown)
- 🔍 **LIME**: Local interpretable explanations (optional)
- 📊 **SHAP**: Gradient-based feature importance (optional)

### Docker Deployment

```bash
# Using Docker Compose (recommended)
docker-compose up

# Or manual Docker
docker build -t deepfake-detector .
docker run -p 8501:8501 deepfake-detector
```

### Cloud Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for complete deployment guides:
- Streamlit Cloud
- Heroku
- AWS EC2
- Google Cloud Run
- Azure Container Instances
- DigitalOcean
- Kubernetes

---

## 🔬 Explainability Methods

### GradCAM (Gradient-weighted Class Activation Mapping)
- **Always shown** in the app
- Highlights spatial regions the model focused on
- Works by computing gradients of the prediction w.r.t. the last convolutional layer
- Visual heatmap overlaid on mel spectrogram

### LIME (Local Interpretable Model-agnostic Explanations)
- **Optional** (toggle in sidebar)
- Perturbs input regions and measures impact on prediction
- Shows which time-frequency regions are most important
- Configurable number of samples (100-2000)

### SHAP (SHapley Additive exPlanations)
- **Optional** (toggle in sidebar)
- Game theory-based feature attribution
- Uses gradient explainer for deep learning models
- Configurable background samples (10-100)

---

## 📁 Project Files

### New Files Created

```
deepfake-audio-detection/
├── app.py                             ← Streamlit web application (UPDATED)
├── DEPLOYMENT.md                      ← Complete deployment guide
├── Dockerfile                         ← Docker image configuration
├── docker-compose.yml                 ← Docker Compose setup
├── .dockerignore                      ← Docker build exclusions
├── .streamlit/
│   └── config.toml                    ← Streamlit theme & config
├── run_app.sh                         ← Quick start script (Linux/Mac)
├── run_app.bat                        ← Quick start script (Windows)
└── src/
    └── inference/
        ├── __init__.py                ← Package init
        ├── predictor.py               ← Audio prediction engine
        └── explainability.py          ← GradCAM, LIME, SHAP implementations
```

---
