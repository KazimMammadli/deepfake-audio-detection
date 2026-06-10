# 🎯 START HERE - Deepfake Audio Detection App

**Welcome!** This is your entry point to the Deepfake Audio Detection Streamlit application.

---

## 🚀 Quick Launch (30 Seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

**That's it!** Open http://localhost:8501 in your browser.

---

## 📚 Documentation Navigation

Choose based on what you need:

### 👤 **I'm a User / First-Timer**
→ Read [GETTING_STARTED.md](GETTING_STARTED.md)
- Complete setup walkthrough
- Usage instructions
- Troubleshooting guide
- Understanding results

### 🎨 **I Want to Know App Features**
→ Read [README_APP.md](README_APP.md)
- All features explained
- Model comparison
- Explainability methods (GradCAM, LIME, SHAP)
- Customization guide

### 🚀 **I Want to Deploy to Production**
→ Read [DEPLOYMENT.md](DEPLOYMENT.md)
- Docker deployment
- Streamlit Cloud
- AWS/GCP/Azure guides
- Security & monitoring

### ✅ **I'm Ready to Deploy (Checklist)**
→ Read [CHECKLIST.md](CHECKLIST.md)
- Pre-deployment verification
- Testing checklist
- Performance validation
- Sign-off template

### 📊 **I Want the Full Technical Overview**
→ Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- Complete feature list
- Architecture details
- Performance metrics
- Files created summary

---

## 📁 Key Files

### Run the App
- **`app.py`** - Main Streamlit application
- **`run_app.bat`** - Windows launcher (double-click to run)
- **`run_app.sh`** - Linux/Mac launcher

### Test Before Running
- **`test_app.py`** - Verify all components work
  ```bash
  python test_app.py
  ```

### Configuration
- **`requirements.txt`** - Python dependencies
- **`.streamlit/config.toml`** - App settings (theme, upload limits)

### Deployment
- **`Dockerfile`** - Build container
- **`docker-compose.yml`** - Run with Docker Compose

---

## 🎯 What Does This App Do?

1. **Upload** an audio file (WAV, MP3, etc.)
2. **Select** a deep learning model (Baseline CNN, MobileNetV2, ResNet50)
3. **Detect** if the audio is Real (human) or Fake (AI-generated)
4. **View** visualizations:
   - Waveform
   - Mel spectrogram
   - **GradCAM heatmap** (always shown)
5. **Explain** predictions (optional):
   - **LIME**: Regional importance
   - **SHAP**: Feature attribution

---

## ⚡ Common Tasks

### Run Locally
```bash
streamlit run app.py
```

### Run Tests
```bash
python test_app.py
```

### Run with Docker
```bash
docker-compose up
```

### Stop Docker
```bash
docker-compose down
```

---

## 🐛 Something Not Working?

### Quick Fixes

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**Port 8501 already in use?**
```bash
streamlit run app.py --server.port=8502
```

**Model files not found?**
- Ensure `.keras` files are in `models/` directory
- Download or train models first (see notebooks)

**Out of memory?**
- Use smaller model (Baseline CNN or MobileNetV2)
- Disable LIME/SHAP in sidebar
- Close other applications

For more help, see [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)

---

## 📖 Full Documentation Index

| Document | Description | When to Read |
|----------|-------------|--------------|
| **START_HERE.md** | You are here! Quick navigation | First visit |
| **GETTING_STARTED.md** | Complete setup & usage guide | Setting up for first time |
| **README_APP.md** | App features & user guide | Learning features |
| **DEPLOYMENT.md** | Production deployment guide | Deploying to cloud/server |
| **CHECKLIST.md** | Pre-deployment checklist | Before going live |
| **PROJECT_SUMMARY.md** | Technical overview & stats | Understanding architecture |
| **README.md** | Project overview (original) | Understanding the project |

---

## 🎓 Learning Path

### Beginner
1. Read this file (START_HERE.md)
2. Read [GETTING_STARTED.md](GETTING_STARTED.md)
3. Run `test_app.py`
4. Launch `app.py`
5. Upload a sample audio file
6. Try different models

### Intermediate
1. Read [README_APP.md](README_APP.md)
2. Explore GradCAM visualizations
3. Enable LIME explanations
4. Compare models
5. Customize the UI

### Advanced
1. Read [DEPLOYMENT.md](DEPLOYMENT.md)
2. Build Docker image
3. Deploy to Streamlit Cloud
4. Set up production monitoring
5. Customize for your use case

---

## 🏗️ Project Structure

```
deepfake-audio-detection/
├── 📱 APP FILES
│   ├── app.py                    ← Main application
│   ├── test_app.py               ← Test script
│   ├── run_app.bat/sh            ← Launchers
│   └── requirements.txt          ← Dependencies
│
├── 🐳 DEPLOYMENT
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── ⚙️ CONFIGURATION
│   └── .streamlit/config.toml
│
├── 📚 DOCUMENTATION
│   ├── START_HERE.md            ← You are here
│   ├── GETTING_STARTED.md
│   ├── README_APP.md
│   ├── DEPLOYMENT.md
│   ├── CHECKLIST.md
│   └── PROJECT_SUMMARY.md
│
├── 🧠 SOURCE CODE
│   └── src/
│       ├── inference/
│       │   ├── predictor.py      ← Model loading
│       │   └── explainability.py ← GradCAM/LIME/SHAP
│       ├── features/
│       │   └── mel_spectrogram.py
│       ├── data/
│       ├── evaluation/
│       └── training/
│
├── 🎯 MODELS
│   └── models/*.keras            ← Trained models
│
└── 📓 NOTEBOOKS
    └── notebooks/*.ipynb         ← Training notebooks
```

---

## 🎯 Success Criteria Checklist

Before using the app, verify:

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] At least one `.keras` model in `models/` directory
- [ ] `test_app.py` passes all tests
- [ ] 4GB+ RAM available

---

## 🤝 Need Help?

### Documentation
All questions answered in the guides above

### Issues
- Check [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)
- Run `test_app.py` to diagnose
- Check GitHub Issues

### Contributing
Pull requests welcome! Areas for improvement:
- Batch processing
- Real-time streaming
- Additional models
- UI enhancements

---

## ✅ Quick Reference

| Action | Command |
|--------|---------|
| Install | `pip install -r requirements.txt` |
| Test | `python test_app.py` |
| Run | `streamlit run app.py` |
| Docker Build | `docker build -t deepfake-detector .` |
| Docker Run | `docker-compose up` |
| Stop | `Ctrl+C` or `docker-compose down` |

---

## 🎉 Ready to Start?

1. **First time?** → [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Already set up?** → `streamlit run app.py`
3. **Want to deploy?** → [DEPLOYMENT.md](DEPLOYMENT.md)

---

**🎙️ Let's detect some deepfakes!**

Open http://localhost:8501 after running the app and start uploading audio files! ✨
