# 🏗️ Architecture & Design

Complete technical architecture of the Deepfake Audio Detection Streamlit application.

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Sidebar   │  │  Main Panel  │  │   Visualizations  │  │
│  │ - Model     │  │ - Upload     │  │ - Waveform        │  │
│  │ - LIME      │  │ - Prediction │  │ - Mel Spec        │  │
│  │ - SHAP      │  │ - Results    │  │ - GradCAM         │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              INFERENCE ENGINE (predictor.py)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  AudioPredictor                                       │   │
│  │  ├─ load_audio() → waveform                         │   │
│  │  ├─ resample_audio() → 16kHz                        │   │
│  │  ├─ audio_to_mel() → (80, 201)                      │   │
│  │  ├─ preprocess_mel_for_model() → model input        │   │
│  │  └─ predict() → probability, label                  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           EXPLAINABILITY (explainability.py)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   GradCAM    │  │     LIME     │  │     SHAP     │      │
│  │ - Gradients  │  │ - Perturb    │  │ - Shapley    │      │
│  │ - Heatmap    │  │ - Segment    │  │ - Values     │      │
│  │ - Overlay    │  │ - Explain    │  │ - Gradient   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DEEP LEARNING MODELS (TensorFlow)               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Baseline   │  │ MobileNetV2  │  │  ResNet50    │        │
│  │    CNN     │  │  (Transfer)  │  │  (Transfer)  │        │
│  │  3 layers  │  │  ImageNet    │  │  ImageNet    │        │
│  └────────────┘  └──────────────┘  └──────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            AUDIO PROCESSING (librosa, soundfile)             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  mel_spectrogram.py                                   │   │
│  │  ├─ Load audio (soundfile)                           │   │
│  │  ├─ Mel spectrogram extraction (librosa)             │   │
│  │  ├─ dB conversion                                     │   │
│  │  ├─ Normalization [0, 1]                             │   │
│  │  └─ Padding/Trimming to 201 frames                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
┌─────────────┐
│ User uploads│
│  audio file │
│ (.wav/.mp3) │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   load_audio()      │
│ - soundfile.read()  │
│ - stereo → mono     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  resample_audio()   │
│ - target: 16 kHz    │
│ - librosa.resample()│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   audio_to_mel()    │
│ - n_mels: 80        │
│ - frames: 201       │
│ - normalize [0,1]   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    preprocess_mel_for_model()       │
│                                     │
│  ┌────────┬──────────┬───────────┐ │
│  │Custom  │MobileNet │  ResNet   │ │
│  │(80,    │(224,     │  (224,    │ │
│  │ 201,1) │ 224,3)   │   224,3)  │ │
│  └────────┴──────────┴───────────┘ │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────┐
│   model.predict()        │
│ - Forward pass           │
│ - Output: probability    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Apply threshold         │
│ - prob >= thresh → Fake  │
│ - prob < thresh  → Real  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│    Return results        │
│ - label: Real/Fake       │
│ - probability: [0,1]     │
│ - mel: (80, 201)         │
│ - audio: waveform        │
└──────────────────────────┘
```

---

## 🧠 Model Architecture Details

### Baseline CNN

```
Input: (batch, 80, 201, 1)
    ↓
[Conv2D(32, 3×3)] → [BatchNorm] → [ReLU] → [MaxPool2D]
    ↓ (batch, 40, 100, 32)
[Conv2D(64, 3×3)] → [BatchNorm] → [ReLU] → [MaxPool2D]
    ↓ (batch, 20, 50, 64)
[Conv2D(128, 3×3)] → [BatchNorm] → [ReLU] → [MaxPool2D]
    ↓ (batch, 10, 25, 128)
[GlobalAveragePooling2D]
    ↓ (batch, 128)
[Dropout(0.3)]
    ↓
[Dense(1, sigmoid)]
    ↓
Output: (batch, 1)  probability ∈ [0, 1]
```

### MobileNetV2 Transfer

```
Input: (batch, 224, 224, 3)
    ↓
[MobileNetV2 Base] (ImageNet weights, frozen Phase 1)
    ↓ (batch, 7, 7, 1280)
[GlobalAveragePooling2D]
    ↓ (batch, 1280)
[Dropout(0.3)]
    ↓
[Dense(1, sigmoid)]
    ↓
Output: (batch, 1)  probability ∈ [0, 1]

Phase 2: Unfreeze last 30 layers, fine-tune
```

### ResNet50 Transfer

```
Input: (batch, 224, 224, 3)
    ↓
[ResNet50 Base] (ImageNet weights, frozen Phase 1)
    ↓ (batch, 7, 7, 2048)
[GlobalAveragePooling2D]
    ↓ (batch, 2048)
[Dropout(0.3)]
    ↓
[Dense(1, sigmoid)]
    ↓
Output: (batch, 1)  probability ∈ [0, 1]

Phase 2: Unfreeze last 30 layers, fine-tune
```

---

## 🔥 GradCAM Algorithm

```
Given: Model M, Input X, Predicted class c

1. Forward pass: X → M → probability p_c
   
2. Compute gradient of p_c w.r.t. last conv layer activations A:
   ∇_A p_c
   
3. Global average pooling of gradients:
   α_k = (1/Z) Σ Σ (∂p_c / ∂A^k_ij)  ← importance of feature map k
            i  j
   
4. Weighted combination of feature maps:
   L_GradCAM = ReLU(Σ α_k · A^k)
                    k
   
5. Upsample L_GradCAM to input size
   
6. Normalize to [0, 1]
   
7. Overlay on input image (mel spectrogram)
```

---

## 🔍 LIME Algorithm

```
Given: Model M, Input X (mel spectrogram), Number of samples N

1. Segment X into superpixels S = {s_1, s_2, ..., s_n}
   (using SLIC or QuickShift)
   
2. For i = 1 to N:
   a. Create perturbed sample X' by randomly hiding superpixels
   b. Predict: y'_i = M(X')
   c. Record: (binary_mask_i, y'_i)
   
3. Learn linear model g:
   g(mask) = w · mask + b
   
   Such that g approximates M locally around X:
   min Σ (M(X'_i) - g(mask_i))² · π(X'_i, X)
    w  i
    
   where π is a proximity kernel
   
4. Extract feature importance from weights w
   
5. Visualize top-k most important superpixels
```

---

## 📊 SHAP Algorithm

```
Given: Model M, Input X, Background dataset B

1. Select background samples: B_subset ⊂ B (e.g., 50 samples)
   
2. Use GradientExplainer:
   φ_j(X) = E[∇_X M(X)] · (X_j - E[X_j])
   
   where φ_j is the SHAP value for feature j
   
3. For each pixel (i, j) in input X:
   SHAP_ij = ∫ (∂M/∂X_ij) dX  (integrated over background)
   
4. Compute attribution:
   Importance_ij = |SHAP_ij|
   
5. Aggregate across color channels (if RGB):
   Heatmap_ij = mean(|SHAP_ij^c|)  for c in {R, G, B}
   
6. Normalize and visualize
```

---

## 🗄️ Data Structures

### AudioPredictor Class

```python
class AudioPredictor:
    Attributes:
        model_key: str           # e.g., "resnet50_v2"
        arch: str                # "custom" | "mobilenet_v2" | "resnet50"
        threshold: float         # Decision boundary (0.0 - 1.0)
        model: tf.keras.Model    # Loaded Keras model
    
    Methods:
        predict_from_audio(audio, sr) → dict
        predict_file(path) → dict
        get_model() → tf.keras.Model
        get_arch() → str
```

### Prediction Result Dictionary

```python
result = {
    "probability": float,    # [0, 1] - Model confidence
    "label": str,            # "Real" or "Fake"
    "threshold": float,      # Decision threshold used
    "mel": np.ndarray,       # (80, 201) mel spectrogram
    "audio": np.ndarray,     # (N,) waveform array
}
```

### Mel Configuration

```python
MEL_CFG = {
    "sample_rate": 16000,    # Hz
    "n_fft": 512,            # FFT window size
    "hop_length": 160,       # 10ms hop
    "win_length": 400,       # 25ms window
    "n_mels": 80,            # Mel bands
    "f_min": 20.0,           # Min frequency
    "f_max": 8000.0,         # Max frequency (Nyquist for 16kHz)
    "target_frames": 201,    # Fixed time dimension
}
```

---

## 🔀 State Management

### Streamlit Caching

```python
# Model loading (expensive, cache across runs)
@st.cache_resource
def load_predictor(model_name, model_filename):
    return AudioPredictor(...)

# Data processing (cheap, cache per input)
@st.cache_data
def process_audio(audio_bytes):
    return preprocess(audio_bytes)
```

### Session State

```python
# Streamlit maintains session state automatically
# No explicit state management needed for this app

# Temp file handling
temp_path = Path("temp_audio.wav")
# ... use file ...
temp_path.unlink(missing_ok=True)  # Clean up
```

---

## 🔐 Security Architecture

### Input Validation

```
User Upload
    ↓
┌───────────────────┐
│ File type check   │ ← Only WAV, MP3, FLAC, OGG
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Size limit check  │ ← Max 200 MB
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Safe file saving  │ ← Temp file, cleaned after
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Audio validation  │ ← Can soundfile read it?
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Process audio     │ ← Safe numpy operations
└───────────────────┘
```

### No Code Execution

- ✅ Only data processing (numpy, librosa)
- ✅ No `eval()`, `exec()`, or dynamic imports
- ✅ No shell command execution with user input
- ✅ Temp files use secure paths

---

## 📈 Performance Architecture

### Optimization Strategies

| Component | Strategy | Impact |
|-----------|----------|--------|
| Model Loading | `@st.cache_resource` | 1000x faster on reload |
| Mel Extraction | Numpy vectorization | 10x faster |
| GradCAM | Precompiled TF graph | 5x faster |
| LIME | Parallel perturbations | 2x faster |
| SHAP | Gradient (not KernelExplainer) | 10x faster |

### Memory Management

```
Base App:           ~500 MB
+ Baseline CNN:     +200 MB  → 700 MB total
+ MobileNetV2:      +500 MB  → 1.2 GB total
+ ResNet50:         +1.3 GB  → 2.5 GB total
+ LIME (active):    +500 MB  → 3.0 GB total (peak)
+ SHAP (active):    +200 MB  → 3.2 GB total (peak)
```

**Recommendation**: 4 GB RAM minimum, 8 GB comfortable

---

## 🌐 Deployment Architecture

### Docker Layers

```
┌────────────────────────────────────┐
│ python:3.10-slim (base)            │ ← 400 MB
├────────────────────────────────────┤
│ System dependencies                │ ← +100 MB
│ (libsndfile, ffmpeg, etc.)         │
├────────────────────────────────────┤
│ Python packages                    │ ← +1.5 GB
│ (tensorflow, streamlit, etc.)      │
├────────────────────────────────────┤
│ Application code                   │ ← +5 MB
│ (app.py, src/, etc.)               │
├────────────────────────────────────┤
│ Model files                        │ ← +500 MB - 2 GB
│ (*.keras models)                   │
└────────────────────────────────────┘
Total: ~2.5 - 4 GB (depending on models)
```

### Cloud Deployment Options

```
┌─────────────────────────────────────────────────┐
│          Streamlit Cloud (Recommended)          │
│  Pros: Free, Easy, Auto-scaling                 │
│  Cons: Resource limits, Public repo             │
└─────────────────────────────────────────────────┘
                     OR
┌─────────────────────────────────────────────────┐
│               Docker Container                   │
│    ┌─────────────────────────────────────┐     │
│    │  App Container (Port 8501)          │     │
│    │  + Nginx Reverse Proxy (Port 80)    │     │
│    └─────────────────────────────────────┘     │
│                                                  │
│  Deploy to:                                      │
│  - AWS EC2 / ECS / Fargate                      │
│  - Google Cloud Run / GKE                       │
│  - Azure Container Instances / AKS              │
│  - DigitalOcean Droplets / App Platform         │
└─────────────────────────────────────────────────┘
```

---

## 🧪 Testing Architecture

### Test Pyramid

```
        ┌─────────────────┐
       /  E2E Tests       /  ← Manual (UI testing)
      /  (Manual)        /
     ┌─────────────────┐
    /  Integration     /  ← test_app.py
   /  Tests           /      (Component tests)
  ┌─────────────────┐
 /  Unit Tests      /  ← (Implicit in components)
/  (Functions)     /
└─────────────────┘
```

### test_app.py Coverage

```
✅ Imports          → All dependencies present
✅ Model Files      → .keras files exist
✅ Mel Extraction   → Pipeline correctness
✅ Predictor        → Model loading & inference
✅ GradCAM          → Explainability generation
```

---

## 📊 Monitoring Architecture (Production)

```
┌──────────────────────────────────────────────┐
│            Streamlit App                     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│         Application Logs                     │
│  - Info: Predictions, model loads           │
│  - Warnings: Slow operations                 │
│  - Errors: Exceptions, failures              │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│      Log Aggregation Service                 │
│  (CloudWatch / Stackdriver / ELK)            │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│         Metrics & Alerts                     │
│  - Response time                             │
│  - Error rate                                │
│  - Memory usage                              │
│  - Model prediction distribution             │
└──────────────────────────────────────────────┘
```

---

## 🔄 CI/CD Architecture (Future)

```
┌─────────────┐
│  Git Push   │
└──────┬──────┘
       │
       ▼
┌────────────────────┐
│  GitHub Actions    │
│  - Run tests       │
│  - Build Docker    │
│  - Push to registry│
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│  Deploy to Staging │
│  - Health checks   │
│  - Smoke tests     │
└──────┬─────────────┘
       │
       ▼ (Manual approval)
┌────────────────────┐
│  Deploy to Prod    │
│  - Blue/green      │
│  - Rollback ready  │
└────────────────────┘
```

---

## 📝 Design Patterns Used

### Singleton Pattern
- Model loading (cached once per session)

### Factory Pattern
- Model architecture selection based on type

### Strategy Pattern
- Different preprocessing for different models

### Observer Pattern
- Streamlit reactive updates

### Builder Pattern
- GradCAM/LIME/SHAP visualization construction

---

## 🎯 Design Principles

### SOLID
- **S**ingle Responsibility: Each module has one job
- **O**pen/Closed: Easy to add new models
- **L**iskov Substitution: All models share interface
- **I**nterface Segregation: Minimal predictor API
- **D**ependency Inversion: Abstract model interface

### DRY (Don't Repeat Yourself)
- Reusable preprocessing functions
- Shared visualization utilities

### KISS (Keep It Simple, Stupid)
- Simple predictor API
- Clear data flow
- Minimal abstractions

---

**🏗️ Architecture designed for clarity, performance, and maintainability!**
