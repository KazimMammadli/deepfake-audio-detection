# 🚀 Deployment Guide

This guide covers multiple deployment options for the Deepfake Audio Detection Streamlit app.

---

## 📋 Prerequisites

- Python 3.10+
- All trained model files in `models/` directory
- Dependencies from `requirements.txt`

---

## 🏠 Local Deployment

### Option 1: Direct Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Option 2: Custom Port

```bash
streamlit run app.py --server.port=8080
```

---

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t deepfake-audio-detector .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  deepfake-audio-detector
```

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

Access at `http://localhost:8501`

---

## ☁️ Cloud Deployment Options

### 1. Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Configure:
   - **Main file:** `app.py`
   - **Python version:** 3.10
5. Deploy!

**Note:** Streamlit Cloud has limited resources. Large models (ResNet50) may be slow.

**Add secrets** in Streamlit Cloud dashboard if needed.

### 2. Heroku

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-app-name

# Add buildpacks
heroku buildpacks:add --index 1 https://github.com/heroku/heroku-buildpack-apt
heroku buildpacks:add --index 2 heroku/python

# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Create Aptfile for system dependencies
cat > Aptfile << EOF
libsndfile1
ffmpeg
EOF

# Deploy
git push heroku main

# Open app
heroku open
```

### 3. AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10 and dependencies
sudo apt install python3.10 python3-pip libsndfile1 ffmpeg -y

# Clone your repo
git clone https://github.com/your-username/deepfake-audio-detection.git
cd deepfake-audio-detection

# Install Python dependencies
pip3 install -r requirements.txt

# Run with nohup (keeps running after SSH disconnect)
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

# Or use systemd service (recommended)
sudo nano /etc/systemd/system/deepfake-detector.service
```

**systemd service file:**
```ini
[Unit]
Description=Deepfake Audio Detector Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/deepfake-audio-detection
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable deepfake-detector
sudo systemctl start deepfake-detector

# Check status
sudo systemctl status deepfake-detector
```

**Configure security group** to allow inbound traffic on port 8501.

### 4. Google Cloud Run

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/deepfake-detector

# Deploy to Cloud Run
gcloud run deploy deepfake-detector \
  --image gcr.io/YOUR_PROJECT_ID/deepfake-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 8501

# Get the service URL
gcloud run services describe deepfake-detector --region us-central1 --format 'value(status.url)'
```

### 5. Azure Container Instances

```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name deepfake-rg --location eastus

# Create container registry
az acr create --resource-group deepfake-rg --name deepfakeacr --sku Basic

# Login to ACR
az acr login --name deepfakeacr

# Build and push
az acr build --registry deepfakeacr --image deepfake-detector:latest .

# Deploy to ACI
az container create \
  --resource-group deepfake-rg \
  --name deepfake-detector \
  --image deepfakeacr.azurecr.io/deepfake-detector:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server deepfakeacr.azurecr.io \
  --registry-username $(az acr credential show --name deepfakeacr --query username -o tsv) \
  --registry-password $(az acr credential show --name deepfakeacr --query passwords[0].value -o tsv) \
  --dns-name-label deepfake-detector-app \
  --ports 8501

# Get the FQDN
az container show --resource-group deepfake-rg --name deepfake-detector --query ipAddress.fqdn
```

### 6. DigitalOcean App Platform

1. Connect your GitHub repository
2. Choose **Web Service**
3. Configure:
   - **Dockerfile:** Auto-detected
   - **HTTP Port:** 8501
   - **Instance Size:** Basic (2GB RAM recommended)
4. Click **Deploy**

---

## 🔧 Production Considerations

### Resource Requirements

| Model | Minimum RAM | Recommended RAM | CPU Cores |
|-------|-------------|-----------------|-----------|
| Baseline CNN | 1 GB | 2 GB | 1 |
| MobileNetV2 | 2 GB | 4 GB | 2 |
| ResNet50 | 4 GB | 8 GB | 2-4 |

### Performance Optimization

1. **Model Caching**: Streamlit's `@st.cache_resource` is already implemented
2. **Reduce Model Size**: Use model quantization or pruning
3. **Use CPU-optimized TensorFlow**: `pip install tensorflow-cpu` (smaller, faster on CPU)
4. **Add GPU Support**: For Docker, use `tensorflow-gpu` and NVIDIA Docker runtime

### Security

1. **HTTPS**: Use reverse proxy (nginx/Caddy) with SSL certificate
2. **Rate Limiting**: Implement to prevent abuse
3. **File Upload Restrictions**: Already limited to audio files in the app
4. **Authentication**: Add if needed (Streamlit supports auth via secrets)

### Monitoring

```python
# Add to app.py for basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
```

Consider integrating:
- **Sentry** for error tracking
- **Prometheus** for metrics
- **Grafana** for dashboards

---

## 🧪 Testing Deployment

```bash
# Health check
curl http://your-domain:8501/_stcore/health

# Expected response: {"status": "ok"}
```

---

## 📊 Scaling

### Horizontal Scaling

Use Kubernetes for multi-instance deployment:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepfake-detector
  template:
    metadata:
      labels:
        app: deepfake-detector
    spec:
      containers:
      - name: app
        image: your-registry/deepfake-detector:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: deepfake-detector-service
spec:
  selector:
    app: deepfake-detector
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

```bash
kubectl apply -f deployment.yaml
```

---

## 🆘 Troubleshooting

### Common Issues

**1. Model files not found**
```bash
# Ensure models are in the correct location
ls -lh models/*.keras
```

**2. Out of memory errors**
- Increase container/instance memory
- Use smaller models (MobileNetV2 instead of ResNet50)
- Reduce LIME/SHAP sample counts

**3. Slow predictions**
- Use CPU-optimized TensorFlow
- Consider model quantization
- Add GPU support

**4. Port already in use**
```bash
# Find process using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>
```

---

## 📝 Environment Variables

Create `.env` file for configuration:

```env
# App settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Model settings
DEFAULT_MODEL=resnet50_v2
MODELS_DIR=./models

# Feature flags
ENABLE_LIME=true
ENABLE_SHAP=true
ENABLE_GRADCAM=true
```

Load in app with `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 🎉 Success!

Your Deepfake Audio Detection app should now be deployed and accessible!

For issues or questions, refer to:
- [Streamlit Docs](https://docs.streamlit.io/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- Project Issues on GitHub
