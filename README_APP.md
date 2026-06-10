# 🎙️ Deepfake Audio Detection - Streamlit App

Interactive web application for detecting AI-generated (deepfake) speech using deep learning models with explainability features.

---

## ✨ Features

### Core Functionality
- 🎯 **Multiple Models**: Choose from Baseline CNN, MobileNetV2, or ResNet50 variants
- 📤 **Audio Upload**: Support for WAV, MP3, FLAC, OGG formats
- 🎵 **Waveform Visualization**: Interactive audio waveform display
- 📊 **Mel Spectrogram**: Frequency-domain representation
- 🔮 **Real-time Prediction**: Instant detection with confidence scores

### Explainability (XAI)
- 🔥 **GradCAM** (Always shown): Highlights important regions the model focused on
- 🔍 **LIME** (Optional): Local Interpretable Model-agnostic Explanations
- 📈 **SHAP** (Optional): SHapley Additive exPlanations for feature importance

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/VidadiJavadov/deepfake-audio-detection.git
cd deepfake-audio-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📖 Usage Guide

### 1. Select a Model
In the left sidebar, choose from available models:
- **Baseline CNN**: Fast, lightweight custom architecture
- **MobileNetV2**: Balanced speed and accuracy
- **ResNet50 v2**: Best accuracy (recommended)

### 2. Upload Audio
- Click "Browse files" or drag & drop
- Supported formats: WAV, MP3, FLAC, OGG
- Best results: 16 kHz, mono, 2-second clips

### 3. View Results
The app displays:
- **Prediction**: Real (human) or Fake (AI-generated)
- **Confidence**: Probability score (0-100%)
- **Threshold**: Model-specific decision boundary

### 4. Analyze Visualizations

#### Always Shown:
- **Audio Waveform**: Time-domain representation
- **Mel Spectrogram**: Frequency features used by the model
- **GradCAM Heatmap**: Highlights regions model focused on (overlaid on mel spectrogram)

#### Optional (Toggle in Sidebar):
- **LIME Explanation**: Shows which time-frequency regions most influenced the prediction
- **SHAP Values**: Gradient-based feature importance heatmap

---

## 🧠 Understanding Explainability

### GradCAM (Gradient-weighted Class Activation Mapping)
- **What**: Visualizes which parts of the input the model "looks at"
- **How**: Uses gradients flowing into final conv layer
- **Interpretation**: Brighter (red/yellow) = more important for prediction
- **Use Case**: Quick visual check of model focus areas

### LIME (Local Interpretable Model-agnostic Explanations)
- **What**: Explains individual predictions by perturbing input regions
- **How**: Creates superpixels, toggles them on/off, measures impact
- **Interpretation**: Green = supports prediction, Red = contradicts
- **Use Case**: Understand which time-frequency segments matter most
- **Cost**: Computationally expensive (1000+ forward passes)

### SHAP (SHapley Additive exPlanations)
- **What**: Game-theory based feature attribution
- **How**: Uses gradients to compute each feature's contribution
- **Interpretation**: Brighter = higher importance for prediction
- **Use Case**: Precise pixel-level importance scores
- **Cost**: Moderate (requires background samples)

---

## 🎛️ Configuration

### Sidebar Options

| Setting | Description | Default |
|---------|-------------|---------|
| Model Selection | Choose architecture | ResNet50 v2 |
| Show LIME | Enable LIME explanation | Disabled |
| LIME Samples | Perturbation samples | 1000 |
| Show SHAP | Enable SHAP explanation | Disabled |
| SHAP Background | Background samples | 50 |

### Performance vs Quality Tradeoffs

| Feature | Speed | Memory | Accuracy |
|---------|-------|--------|----------|
| GradCAM | ⚡⚡⚡ Fast | Low | Visual only |
| LIME | 🐢 Slow | Medium | High |
| SHAP | 🐢🐢 Very Slow | High | Very High |

**Recommendation**: Start with GradCAM only, enable LIME/SHAP for deep analysis.

---

## 🏗️ Architecture

```
app.py (Streamlit UI)
├── src/inference/predictor.py       ← Model loading & prediction
├── src/inference/explainability.py  ← GradCAM, LIME, SHAP
├── src/features/mel_spectrogram.py  ← Audio → mel conversion
└── models/*.keras                    ← Trained model weights
```

### Data Flow

```
Audio File (.wav)
    ↓
Load & Resample to 16kHz
    ↓
Extract Mel Spectrogram (80×201)
    ↓
Model-specific Preprocessing
    ↓
CNN Inference
    ↓
Prediction (Real/Fake + Probability)
    ↓
Explainability (GradCAM/LIME/SHAP)
```

---

## 📊 Model Comparison

| Model | Parameters | Inference Time | Accuracy* | Best For |
|-------|-----------|----------------|-----------|----------|
| Baseline CNN | ~500K | ~50ms | ⭐⭐⭐ | Quick prototyping |
| MobileNetV2 | ~2M | ~100ms | ⭐⭐⭐⭐ | Mobile/edge devices |
| ResNet50 | ~23M | ~200ms | ⭐⭐⭐⭐⭐ | Production use |

*Accuracy on FoR-2sec test set

---

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
Error: Model file not found: models/resnet50_v2.keras
```
**Solution**: Ensure trained `.keras` models are in `models/` directory.

**2. Out of memory during LIME/SHAP**
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution**: Reduce sample counts or disable LIME/SHAP.

**3. Audio file not supported**
```
Error: Could not read audio file
```
**Solution**: Convert to WAV format using `ffmpeg` or `librosa`.

**4. Slow inference**
**Solution**: 
- Use MobileNetV2 instead of ResNet50
- Disable LIME/SHAP
- Use GPU if available (requires CUDA)

### Debug Mode

```bash
# Run with debug logging
streamlit run app.py --logger.level=debug
```

---

## 🔧 Customization

### Add Your Own Model

1. Train model following existing architecture pattern
2. Save as `.keras` file in `models/`
3. Edit `app.py`:
   ```python
   AVAILABLE_MODELS = {
       "Your Model": "your_model.keras",
       # ... existing models
   }
   ```
4. Add architecture mapping in `predictor.py` if using new architecture

### Change Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#YOUR_BG"
```

### Adjust Upload Limits

Edit `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200  # MB
```

---

## 📚 Technical Details

### Input Format
- **Sample Rate**: 16 kHz (auto-resampled)
- **Channels**: Mono (stereo auto-converted)
- **Duration**: Any (internally chunked to 2s for processing)

### Mel Spectrogram Config
```python
n_mels = 80
n_fft = 512
hop_length = 160
win_length = 400
f_min = 20 Hz
f_max = 8000 Hz
target_frames = 201
```

### Model Input Shapes
- **Baseline CNN**: (batch, 80, 201, 1)
- **MobileNetV2**: (batch, 224, 224, 3)
- **ResNet50**: (batch, 224, 224, 3)

---

## 🎓 Educational Use

This app is designed for:
- 📖 Understanding deepfake detection
- 🔬 Research and experimentation
- 👨‍🎓 Educational demonstrations
- 🛠️ Prototyping detection systems

**Not recommended for**:
- ⚖️ Legal evidence (no detection is 100% accurate)
- 🏛️ Forensic analysis without human expert review
- 📰 Journalism as sole verification method

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Real-time audio streaming
- [ ] Batch processing multiple files
- [ ] Export detailed reports (PDF/JSON)
- [ ] A/B model comparison
- [ ] Custom threshold calibration UI
- [ ] GPU acceleration toggle

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **FoR Dataset**: [Fake-or-Real dataset](https://bil.eecs.yorku.ca/datasets/)
- **Streamlit**: Web app framework
- **TensorFlow**: Deep learning framework
- **LIME**: Model interpretation library
- **SHAP**: Explainability library

---

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/VidadiJavadov/deepfake-audio-detection/issues)
- 📧 **Contact**: Open an issue for questions
- 📖 **Docs**: See `DEPLOYMENT.md` for deployment guide

---

**Built with ❤️ by the Holberton School Capstone Team**

[Kazim Mammadli](https://github.com/kazimmammadli) • [Vidadi Javadov](https://github.com/VidadiJavadov) • [Zamin Sultanli](https://github.com/zaminsultanli)
