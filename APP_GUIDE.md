# 📖 Streamlit App User Guide

Complete guide to using the Deepfake Audio Detection web application.

---

## 🚀 Starting the App

### Method 1: Quick Start Scripts

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
chmod +x run_app.sh
./run_app.sh
```

### Method 2: Manual Start

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 🎯 Using the App

### Step 1: Select a Model

In the **left sidebar**, choose from 9 available models:

| Model | Type | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **Baseline CNN** | Custom | ⚡⚡⚡ Fast | ⭐⭐ Good | Quick testing |
| **MobileNetV2 (Phase 1)** | Transfer | ⚡⚡ Fast | ⭐⭐⭐ Better | Balanced |
| **MobileNetV2 (Fine-tuned)** | Transfer | ⚡⚡ Fast | ⭐⭐⭐⭐ Great | Mobile/edge |
| **ResNet50 (Phase 1)** | Transfer | ⚡ Moderate | ⭐⭐⭐ Better | Deep features |
| **ResNet50 v2** | Transfer | ⚡ Moderate | ⭐⭐⭐⭐⭐ Best | **Recommended** |
| **ResNet50 v3** | Transfer | ⚡ Moderate | ⭐⭐⭐⭐ Great | Alternative |

**Recommended:** **ResNet50 v2** for best accuracy (optimized threshold)

### Step 2: Configure Explainability

Choose which explanation methods to show:

#### GradCAM (Always Shown)
- No configuration needed
- Automatically displays attention heatmap
- Shows where the model "looked" when making the decision

#### LIME (Optional)
- Toggle: Check **"Show LIME Explanation"**
- Samples: 100-2000 (default: 1000)
  - More samples = more accurate but slower
  - 1000 is a good balance (~10-15 seconds)

#### SHAP (Optional)
- Toggle: Check **"Show SHAP Explanation"**
- Background samples: 10-100 (default: 50)
  - More samples = more accurate but slower
  - 50 is recommended (~15-20 seconds)

**Performance Tip:** Start with only GradCAM, add LIME/SHAP if you need deeper analysis.

### Step 3: Upload Audio

1. Click **"Browse files"** or drag & drop
2. Supported formats: WAV, MP3, FLAC, OGG
3. Recommended: **2-second clips, 16 kHz, mono**

**Sample Audio Sources:**
- Record your own voice
- Use text-to-speech services (e.g., ElevenLabs, Google TTS)
- Download from the FoR dataset
- Test with any audio file

### Step 4: View Results

#### Prediction Section

Shows:
- **Model Name**: Which model made the prediction
- **Threshold**: Decision boundary (0.5 by default, 0.21 for ResNet50 v2)
- **Confidence**: Probability score (0-100%)
- **Prediction Label**: 
  - 🟢 **Real** (Human Voice) - Green
  - 🔴 **Fake** (AI-Generated) - Red
- **Progress Bar**: Visual confidence indicator

#### Visualizations

**1. Audio Waveform**
- Time-domain representation
- X-axis: Time (seconds)
- Y-axis: Amplitude
- Shows the raw audio signal

**2. Mel Spectrogram**
- Time-frequency representation
- X-axis: Time (seconds)
- Y-axis: Mel frequency bins (0-80)
- Color: Intensity (dB scale)
- Brighter = more energy at that frequency/time

**3. GradCAM Heatmap**
- Three views: Original mel | Heatmap | Overlay
- Red/yellow = high attention (model focused here)
- Blue/dark = low attention
- **Interpretation:**
  - If red regions align with typical speech patterns → Real
  - If red regions show artifacts or unnatural patterns → Fake

**4. LIME Explanation** (if enabled)
- Left: Feature importance (red=positive, blue=negative)
- Right: Overlay on mel spectrogram
- **Interpretation:**
  - Red regions pushed prediction toward "Fake"
  - Blue regions pushed prediction toward "Real"
  - Brighter = stronger influence

**5. SHAP Explanation** (if enabled)
- Left: SHAP importance heatmap
- Right: Overlay on mel spectrogram
- **Interpretation:**
  - Bright regions = high importance for the prediction
  - Shows global feature attribution

---

## 📊 Understanding Predictions

### Confidence Scores

| Probability | Meaning | Action |
|-------------|---------|--------|
| **0-20%** | Very confident REAL | Likely human voice |
| **20-45%** | Confident REAL | Probably human voice |
| **45-55%** | Uncertain | Borderline case, review carefully |
| **55-80%** | Confident FAKE | Probably AI-generated |
| **80-100%** | Very confident FAKE | Likely AI-generated |

**Note:** ResNet50 v2 uses a threshold of **0.21** (not 0.5), optimized for better precision/recall balance.

### False Positives/Negatives

**False Positive (Real labeled as Fake):**
- Poor audio quality (noise, distortion)
- Heavily processed audio (autotune, compression)
- Non-English speech (model trained on English)
- Very short clips (<1 second)

**False Negative (Fake labeled as Real):**
- High-quality deepfake audio (modern TTS)
- Novel synthesis methods not in training data
- Mixed real+fake audio

**Best Practice:** Always use explainability tools (GradCAM/LIME/SHAP) to understand WHY the model made its decision.

---

## 🔍 Interpreting Explainability

### GradCAM Patterns

#### Real Audio Indicators:
- Attention on **harmonic structures** (horizontal bands)
- Focus on **formant transitions** (speech-specific patterns)
- Distributed attention across time and frequency

#### Fake Audio Indicators:
- Attention on **artifacts** or unusual patterns
- Focus on **phase inconsistencies**
- Concentrated attention on specific anomalies

### LIME Analysis

- **Top 10-20 features** have the most impact
- Look for:
  - Consistent patterns across multiple regions
  - Unusual frequency bands being important
  - Time segments with high influence

### SHAP Analysis

- More computationally expensive but more theoretically grounded
- Shows **additive contributions** of each feature
- Useful for:
  - Understanding model behavior
  - Debugging edge cases
  - Research and analysis

---

## ⚙️ Advanced Tips

### Performance Optimization

1. **Use smaller models** for faster predictions (MobileNetV2)
2. **Disable LIME/SHAP** for real-time use
3. **Reduce sample counts** (LIME: 500, SHAP: 25)
4. **Pre-process audio** to 16 kHz mono before upload

### Best Audio Format

**Optimal:**
- Format: WAV (uncompressed)
- Sample rate: 16 kHz
- Channels: Mono
- Duration: 2 seconds
- Bit depth: 16-bit

**Acceptable:**
- MP3 at 128 kbps or higher
- Any sample rate (will be resampled)
- Stereo (will be converted to mono)
- Any duration (will be trimmed/padded)

### Batch Processing

Currently not supported in the web UI. For batch processing:

```python
from src.inference.predictor import AudioPredictor
from pathlib import Path

predictor = AudioPredictor("resnet50_v2", "models/resnet50_v2.keras")

for audio_file in Path("audio_folder").glob("*.wav"):
    result = predictor.predict_file(str(audio_file))
    print(f"{audio_file.name}: {result['label']} ({result['probability']:.2%})")
```

### Keyboard Shortcuts

- `Ctrl+K` or `Cmd+K`: Focus search
- `R`: Rerun app
- `C`: Clear cache
- `Ctrl+S`: Settings

### Saving Results

To save visualizations:
1. Right-click on any plot
2. Select "Save image as..."
3. Choose PNG or SVG format

Or use browser's print function to save full report.

---

## 🐛 Troubleshooting

### Issue: "Model file not found"

**Solution:**
```bash
# Check if models exist
ls -l models/*.keras

# If missing, train models first (see notebooks)
# Or download pre-trained models
```

### Issue: "Audio file upload fails"

**Solution:**
- Check file format (WAV, MP3, FLAC, OGG only)
- Ensure file size < 200 MB (default limit)
- Try converting with: `ffmpeg -i input.mp3 output.wav`

### Issue: "Out of memory error"

**Solution:**
1. Use smaller model (Baseline CNN or MobileNetV2)
2. Disable SHAP/LIME
3. Reduce sample counts
4. Restart the app

### Issue: "GradCAM shows blank heatmap"

**Solution:**
- This can happen with certain models/architectures
- Try a different model
- Check that model has convolutional layers

### Issue: "Prediction is always wrong"

**Possible causes:**
- Model not trained properly
- Audio is from a distribution very different from training data
- Model might be detecting something other than deepfake indicators

**Debugging:**
1. Test with known real/fake samples from the training set
2. Check GradCAM to see what the model is focusing on
3. Try different models
4. Review model training metrics

---

## 📚 Additional Resources

### Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [README.md](README.md) - Project overview

### Model Details
- Check notebooks for training details
- Review `src/evaluation/metrics.py` for evaluation code
- See `src/training/preprocessing.py` for data pipeline

### Research Papers
- **GradCAM:** [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- **LIME:** ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- **SHAP:** [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

---

## 🆘 Getting Help

1. **Check logs**: Look at terminal/console where app is running
2. **Error messages**: Read the full error message in the app
3. **GitHub Issues**: Report bugs or ask questions
4. **Documentation**: Review README.md and DEPLOYMENT.md

---

## ✨ Tips for Best Results

1. ✅ Use **ResNet50 v2** for most accurate predictions
2. ✅ Always check **GradCAM** to understand model attention
3. ✅ Use **LIME/SHAP** for borderline cases (45-55% confidence)
4. ✅ Test with **2-second clips** for optimal performance
5. ✅ **Compare multiple models** on the same audio for confidence
6. ✅ Look for **consistent patterns** across explainability methods
7. ✅ Remember: **No model is perfect** - use as a tool, not absolute truth

---

## 🎉 Happy Detecting!

Questions? Found a bug? Have suggestions?  
Open an issue on GitHub or contact the team.
