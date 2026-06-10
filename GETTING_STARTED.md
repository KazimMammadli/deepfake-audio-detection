# 🚀 Getting Started with Deepfake Audio Detection

Complete guide to set up and run the Streamlit app in under 10 minutes.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.10 or higher** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **4GB+ RAM** (8GB recommended for ResNet50)
- **Internet connection** (for dependencies)

**Optional:**
- **GPU** with CUDA (for faster inference)
- **Docker** (for containerized deployment)

---

## ⚡ Quick Start (5 Minutes)

### 1. Clone Repository

```bash
git clone https://github.com/VidadiJavadov/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### 2. Install Dependencies

**Windows:**
```bash
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
pip3 install -r requirements.txt
```

### 3. Download Models

**Option A: Use Pre-trained Models**

If you have trained models, place them in the `models/` directory:
```
models/
├── baseline_cnn.keras
├── mobilenetv2.keras
├── resnet50_v2.keras
└── ... (other variants)
```

**Option B: Train Models Yourself**

Follow the notebooks in order:
1. `notebooks/01_pipeline_validation.ipynb`
2. `notebooks/02_baseline_model.ipynb`
3. `notebooks/03_mobilenetv2.ipynb`
4. `notebooks/04_resnet50.ipynb`

Models will be saved to `models/` automatically.

### 4. Run the App

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Or directly:**
```bash
streamlit run app.py
```

### 5. Open in Browser

The app automatically opens at: **http://localhost:8501**

---

## 🎯 First Time Usage

### Step 1: Select Model

In the left sidebar, choose a model:
- **Baseline CNN** - Fast, lightweight
- **MobileNetV2** - Balanced performance
- **ResNet50 v2** - Best accuracy (recommended)

### Step 2: Upload Audio

Click "Browse files" or drag & drop a WAV/MP3 file.

**Best results:** 16 kHz, mono, 2-second clips

### Step 3: View Results

The app shows:
- ✅ Prediction (Real or Fake)
- 📊 Confidence percentage
- 🌊 Audio waveform
- 🎵 Mel spectrogram
- 🔥 GradCAM heatmap

### Step 4: Enable Explainability (Optional)

Toggle in sidebar:
- **LIME** - Shows important regions (slow, 30-60s)
- **SHAP** - Feature importance (moderate, 10-20s)

---

## 📁 Project Structure Overview

```
deepfake-audio-detection/
│
├── app.py                    ← Main Streamlit app (START HERE)
├── test_app.py               ← Run tests before deployment
├── requirements.txt          ← Python dependencies
│
├── src/
│   ├── inference/
│   │   ├── predictor.py      ← Model loading & prediction
│   │   └── explainability.py ← GradCAM, LIME, SHAP
│   └── features/
│       └── mel_spectrogram.py ← Audio preprocessing
│
├── models/                   ← Place .keras files here
│   ├── baseline_cnn.keras
│   ├── mobilenetv2.keras
│   └── resnet50_v2.keras
│
└── notebooks/                ← Training notebooks (Colab)
    ├── 01_pipeline_validation.ipynb
    ├── 02_baseline_model.ipynb
    ├── 03_mobilenetv2.ipynb
    └── 04_resnet50.ipynb
```

---

## 🧪 Testing Your Setup

### Run Component Tests

```bash
python test_app.py
```

This checks:
- ✅ All imports work
- ✅ Model files exist
- ✅ Mel extraction works
- ✅ Predictor loads successfully
- ✅ GradCAM generates correctly

**Expected output:**
```
🧪 Deepfake Audio Detection App - Component Tests
====================================
Testing imports...
  ✅ All imports successful

Testing model files...
  ✅ baseline_cnn.keras
  ✅ mobilenetv2.keras
  ...

📊 Test Summary
  ✅ PASS  Imports
  ✅ PASS  Model Files
  ✅ PASS  Mel Extraction
  ✅ PASS  Predictor
  ✅ PASS  GradCAM

Passed: 5/5
🎉 All tests passed! Ready to run: streamlit run app.py
```

---

## 🐛 Troubleshooting

### Issue: Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Fix:**
```bash
pip install -r requirements.txt
```

### Issue: Model Not Found

**Error:**
```
Model file not found: models/resnet50_v2.keras
```

**Fix:**
1. Check `models/` directory exists
2. Ensure `.keras` files are present
3. Train models using notebooks or download pre-trained

### Issue: Out of Memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Fix:**
1. Use smaller model (Baseline CNN or MobileNetV2)
2. Disable LIME/SHAP
3. Close other applications
4. Increase system RAM

### Issue: Slow Inference

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Large model (ResNet50) | Use MobileNetV2 |
| LIME enabled | Reduce samples to 500 or disable |
| SHAP enabled | Reduce background samples to 25 |
| CPU-only | Enable GPU with CUDA |

### Issue: Audio Format Not Supported

**Error:**
```
Could not read audio file
```

**Fix:**
Convert to WAV using:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

## 🎓 Understanding the Output

### Prediction Label

- **Real (Human Voice)** ✅ - Natural human speech
- **Fake (AI-Generated)** ⚠️ - Synthetic/deepfake audio

### Confidence Score

- **0-30%** - Likely Real
- **30-70%** - Uncertain (review carefully)
- **70-100%** - Likely Fake

### Threshold

Model-specific decision boundary:
- Probability ≥ Threshold → Fake
- Probability < Threshold → Real

ResNet50 v2 uses optimized threshold: **0.21**

### Visualizations

**Waveform**
- Shows audio amplitude over time
- Helps identify clipping, silence, noise

**Mel Spectrogram**
- Frequency representation (80 mel bands)
- Shows speech patterns and formants
- Input to the neural network

**GradCAM Heatmap**
- Red/Yellow = Model focused here
- Blue = Model ignored
- Shows what influenced the decision

**LIME (Optional)**
- Green regions = Support prediction
- Red regions = Contradict prediction
- Helps understand local decision

**SHAP (Optional)**
- Brighter = More important
- Pixel-level feature attribution
- Based on game theory

---

## 📚 Next Steps

### For Users

1. **Test with various audio samples**
   - Real human recordings
   - Known deepfakes
   - Different speakers, languages

2. **Experiment with models**
   - Compare predictions across models
   - Check which model works best for your use case

3. **Explore explainability**
   - Enable LIME to see important regions
   - Use SHAP for detailed analysis
   - Compare explanations across samples

### For Developers

1. **Read the documentation**
   - [README_APP.md](README_APP.md) - App features
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Deploy to production
   - [CHECKLIST.md](CHECKLIST.md) - Pre-deployment checks

2. **Customize the app**
   - Add your own models
   - Modify UI/theme
   - Implement batch processing

3. **Deploy to production**
   - Use Docker for consistency
   - Deploy on Streamlit Cloud (free)
   - Or use AWS/GCP/Azure

---

## 💡 Tips & Best Practices

### Audio Preparation

✅ **Do:**
- Use 16 kHz sample rate
- Convert to mono
- Clip to 2-second segments
- Normalize audio levels

❌ **Avoid:**
- Extremely long files (>30s)
- Very low quality recordings
- Heavy background noise

### Model Selection

| Use Case | Recommended Model |
|----------|------------------|
| Quick testing | Baseline CNN |
| Mobile/Edge | MobileNetV2 |
| Best accuracy | ResNet50 v2 |
| Research | Compare all models |

### Performance Tips

- Start with GradCAM only (fastest)
- Enable LIME/SHAP only when needed
- Reduce sample counts for faster results
- Use smaller models for real-time apps

---

## 🔗 Useful Links

- **GitHub**: [deepfake-audio-detection](https://github.com/VidadiJavadov/deepfake-audio-detection)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **FoR Dataset**: [Fake-or-Real](https://bil.eecs.yorku.ca/datasets/)
- **TensorFlow**: [tensorflow.org](https://www.tensorflow.org)

---

## 📞 Getting Help

### Issues & Bugs
Open an issue on [GitHub Issues](https://github.com/VidadiJavadov/deepfake-audio-detection/issues)

### Questions
Check existing issues or create a new one

### Contributing
Pull requests welcome! See contributing guidelines

---

## ✅ Quick Reference Commands

```bash
# Run app
streamlit run app.py

# Run tests
python test_app.py

# Docker build
docker build -t deepfake-detector .

# Docker run
docker run -p 8501:8501 deepfake-detector

# Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop app
Ctrl + C (or docker-compose down)
```

---

**🎉 You're all set! Start detecting deepfakes now!**

Upload an audio file at **http://localhost:8501** and see the magic happen! ✨
