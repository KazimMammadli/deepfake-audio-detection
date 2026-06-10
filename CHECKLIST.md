# ✅ Deployment Checklist

Use this checklist before deploying the Deepfake Audio Detection app to production.

---

## 📋 Pre-Deployment

### Code & Dependencies
- [ ] All required packages in `requirements.txt` are specified with version pins
- [ ] `test_app.py` passes all tests locally
- [ ] No hardcoded secrets or API keys in code
- [ ] `.gitignore` excludes sensitive files and large data files
- [ ] Code is properly documented

### Models
- [ ] All `.keras` model files are present in `models/` directory
- [ ] Models load successfully (test with `test_app.py`)
- [ ] Model file sizes are reasonable for deployment platform (<500MB recommended)
- [ ] Threshold values are calibrated (check `models/threshold_*.json` files)

### Testing
- [ ] App runs locally without errors: `streamlit run app.py`
- [ ] Audio upload works for all supported formats (WAV, MP3, FLAC, OGG)
- [ ] All visualizations render correctly (waveform, mel, GradCAM)
- [ ] LIME explanation works (may take 30-60s)
- [ ] SHAP explanation works (may take 10-20s)
- [ ] Tested with various audio lengths (1s, 2s, 5s, 10s)
- [ ] Tested with different sample rates (16kHz, 44.1kHz, 48kHz)

---

## 🐳 Docker Deployment

### Build & Test
- [ ] Docker image builds successfully: `docker build -t deepfake-detector .`
- [ ] Image size is reasonable (<5GB recommended)
- [ ] Container runs: `docker run -p 8501:8501 deepfake-detector`
- [ ] App accessible at `http://localhost:8501`
- [ ] Health check endpoint works: `curl http://localhost:8501/_stcore/health`
- [ ] Models are accessible inside container

### Docker Compose
- [ ] `docker-compose up` starts successfully
- [ ] Volume mounts work correctly (models directory)
- [ ] Container restarts on failure
- [ ] Logs are accessible: `docker-compose logs -f`

---

## ☁️ Cloud Deployment (Streamlit Cloud)

### Repository Setup
- [ ] Code pushed to GitHub repository
- [ ] Repository is public (or Streamlit Cloud has access if private)
- [ ] `.gitignore` prevents committing large model files

### Model Files
- [ ] Models are either:
  - [ ] Included in repo (if <500MB total), OR
  - [ ] Hosted externally (Google Drive, S3, etc.) with download script, OR
  - [ ] Using Git LFS for large files

### Streamlit Cloud Config
- [ ] App deployed on [share.streamlit.io](https://share.streamlit.io)
- [ ] Main file path set to `app.py`
- [ ] Python version set to 3.10+
- [ ] Environment variables configured (if needed)
- [ ] Secrets configured (if needed)
- [ ] App loads and runs successfully

---

## 🌐 Production Deployment (AWS/GCP/Azure)

### Infrastructure
- [ ] VM/Container instance provisioned with adequate resources:
  - [ ] CPU: 2+ cores
  - [ ] RAM: 4+ GB (8GB for ResNet50)
  - [ ] Storage: 10+ GB
- [ ] Security groups/firewall rules configured for port 8501
- [ ] SSL/TLS certificate configured (HTTPS)
- [ ] Domain name configured (optional)

### Reverse Proxy (Nginx/Caddy)
- [ ] Reverse proxy configured for Streamlit
- [ ] WebSocket support enabled
- [ ] HTTPS enforced
- [ ] Request size limits configured (200MB+ for audio uploads)

### Monitoring & Logs
- [ ] Application logging configured
- [ ] Error tracking setup (Sentry, CloudWatch, etc.)
- [ ] Performance monitoring enabled
- [ ] Disk space monitoring
- [ ] Memory usage alerts

### Security
- [ ] HTTPS/TLS enabled
- [ ] CORS configured properly
- [ ] Rate limiting implemented (optional)
- [ ] Authentication added (if required)
- [ ] File upload size limits enforced
- [ ] Input validation on uploaded files

---

## 🔧 Performance Optimization

### Model Optimization
- [ ] Consider model quantization for faster inference
- [ ] Enable GPU if available (requires CUDA Docker image)
- [ ] Batch processing for multiple files (if needed)

### Caching
- [ ] `@st.cache_resource` used for model loading (already implemented)
- [ ] `@st.cache_data` used for data processing (where applicable)

### Resource Limits
- [ ] Memory limits configured in Docker/Kubernetes
- [ ] CPU limits configured
- [ ] Timeout configured for long-running operations

---

## 📊 User Experience

### UI/UX
- [ ] App loads quickly (<5 seconds)
- [ ] Upload instructions are clear
- [ ] Progress indicators show during processing
- [ ] Error messages are user-friendly
- [ ] Results are easy to interpret

### Documentation
- [ ] README.md is complete and accurate
- [ ] README_APP.md covers all features
- [ ] DEPLOYMENT.md has deployment instructions
- [ ] Example audio files provided (optional)

---

## 🧪 Post-Deployment Testing

### Functional Tests
- [ ] Upload and predict with real audio samples
- [ ] Test all available models (Baseline, MobileNetV2, ResNet50)
- [ ] Verify GradCAM generates correctly
- [ ] Test LIME with reduced sample count (500) for speed
- [ ] Test SHAP with reduced background samples (25)
- [ ] Test with edge cases:
  - [ ] Very short audio (<1s)
  - [ ] Very long audio (>10s)
  - [ ] Silence/noise
  - [ ] Various formats (WAV, MP3, etc.)

### Performance Tests
- [ ] Single user inference time acceptable (<5s for prediction)
- [ ] Concurrent users supported (test with 5-10 simultaneous uploads)
- [ ] Memory usage stable (no memory leaks)
- [ ] App doesn't crash under load

### Monitoring
- [ ] Application metrics dashboard setup
- [ ] Alerts configured for downtime
- [ ] Log aggregation working
- [ ] Error tracking capturing issues

---

## 🚨 Rollback Plan

- [ ] Previous version tagged in Git
- [ ] Rollback procedure documented
- [ ] Backup of working model files available
- [ ] Database backup (if applicable)

---

## 📞 Support & Maintenance

### Documentation
- [ ] Deployment runbook created
- [ ] Troubleshooting guide available
- [ ] Contact information for support

### Maintenance Plan
- [ ] Dependency update schedule defined
- [ ] Model retraining plan (if applicable)
- [ ] Backup and recovery procedures documented

---

## ✅ Final Sign-Off

**Deployment Date:** _________________

**Deployed By:** _________________

**Environment:** 
- [ ] Development
- [ ] Staging  
- [ ] Production

**Sign-Off:**
- [ ] Technical Lead Approval
- [ ] QA Testing Complete
- [ ] Documentation Complete
- [ ] Monitoring Configured

---

## 📝 Notes

(Add any deployment-specific notes, issues encountered, or special configurations)

---

**🎉 Congratulations! Your Deepfake Audio Detection app is ready for deployment!**
