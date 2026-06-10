"""
Streamlit App: Deepfake Audio Detector

Upload an audio clip to detect whether it is Real or AI-generated (Fake).
Explanations via Grad-CAM, SHAP, and LIME are included.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import librosa.display
from pathlib import Path
import sys
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.predictor import AudioPredictor, audio_to_mel, preprocess_mel_for_model
from src.inference.explainability import (
    get_gradcam_heatmap,
    plot_gradcam,
    generate_lime_explanation,
    plot_lime_explanation,
    generate_shap_explanation,
    plot_shap_explanation,
)

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6C63FF;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .fake-label {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 2rem;
    }
    .real-label {
        color: #00C897;
        font-weight: bold;
        font-size: 2rem;
    }
    .section-header {
        color: #6C63FF;
        font-weight: 600;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 10px;
        font-size: 0.85rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
THRESHOLD = 0.5
SEGMENT_DURATION = 2.0  # seconds
TARGET_SR = 16000

# Available models
AVAILABLE_MODELS = {
    "mobilenetv2": "mobilenetv2.keras",
    "baseline_cnn": "baseline_cnn.keras",
    "resnet50_v3": "resnet50_v3.keras",
}

# ── Helper Functions ────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name: str):
    """Load and cache the model."""
    model_filename = AVAILABLE_MODELS[model_name]
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    
    model_key = model_filename.replace(".keras", "")
    return AudioPredictor(model_key, str(model_path))


def segment_audio(audio, sr, segment_duration=2.0):
    """Segment audio into fixed-duration chunks."""
    segment_samples = int(segment_duration * sr)
    segments = []
    
    for start in range(0, len(audio), segment_samples):
        end = start + segment_samples
        segment = audio[start:end]
        
        # Pad if too short
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        
        segments.append(segment)
    
    return segments


def predict_segments(predictor, segments, sr):
    """Run prediction on all segments."""
    results = []
    
    for i, segment in enumerate(segments):
        result = predictor.predict_from_audio(segment, sr)
        result['segment_id'] = i
        results.append(result)
    
    return results


def select_segment(results, mode):
    """Select segment based on mode."""
    if mode == "Most confident (Fake)":
        return max(results, key=lambda x: x['probability'])
    elif mode == "Least confident (Fake)":
        return min(results, key=lambda x: x['probability'])
    else:  # First segment
        return results[0]


def plot_probability_bars(fake_prob, real_prob, threshold=0.5):
    """Plot horizontal probability bars with threshold line."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    labels = ['Fake', 'Real']
    probabilities = [fake_prob, real_prob]
    colors = ['#FF4B4B' if fake_prob >= threshold else '#00C897',
              '#00C897' if real_prob >= threshold else '#FF4B4B']
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probabilities, color=colors, alpha=0.7)
    
    # Threshold line
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold ({threshold})')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig


def plot_waveform(audio, sr):
    """Plot audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, linewidth=0.5, color='#6C63FF', alpha=0.8)
    ax.fill_between(time, audio, alpha=0.3, color='#6C63FF')
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig


def plot_mel_spectrogram(mel, title="Mel Spectrogram"):
    """Plot mel spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    img = librosa.display.specshow(
        mel,
        sr=TARGET_SR,
        hop_length=160,
        x_axis='frames',
        y_axis='mel',
        ax=ax,
        cmap='viridis'
    )
    
    ax.set_xlabel('Time Frames (0-200)', fontsize=11)
    ax.set_ylabel('Mel Bins (0-80)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig


def plot_gradcam_comparison(mel, heatmap, prediction):
    """Plot mel spectrogram and Grad-CAM side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Left: Mel Spectrogram
    img1 = librosa.display.specshow(
        mel,
        sr=TARGET_SR,
        hop_length=160,
        x_axis='frames',
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frames (0-200)', fontsize=10)
    axes[0].set_ylabel('Mel Bins (0-80)', fontsize=10)
    
    # Right: Grad-CAM Heatmap
    # Resize heatmap to mel shape if needed
    if heatmap.shape != mel.shape:
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            [mel.shape[0], mel.shape[1]]
        ).numpy().squeeze()
    else:
        heatmap_resized = heatmap
    
    img2 = axes[1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title(f'Grad-CAM Heatmap (prediction: {prediction})', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames (0-200)', fontsize=10)
    axes[1].set_ylabel('Mel Bins (0-80)', fontsize=10)
    fig.colorbar(img2, ax=axes[1])
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_shap_comparison(mel, shap_values, prediction):
    """Plot mel spectrogram and SHAP values side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Left: Mel Spectrogram
    img1 = librosa.display.specshow(
        mel,
        sr=TARGET_SR,
        hop_length=160,
        x_axis='frames',
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frames (0-200)', fontsize=10)
    axes[0].set_ylabel('Mel Bins (0-80)', fontsize=10)
    
    # Right: SHAP Values
    # Process SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap_aggregated = shap_values[0]  # First batch item
    if len(shap_aggregated.shape) > 2:
        shap_aggregated = np.sum(shap_aggregated, axis=-1)
    
    # Resize to mel shape if needed
    if shap_aggregated.shape != mel.shape:
        shap_aggregated = tf.image.resize(
            shap_aggregated[..., np.newaxis],
            [mel.shape[0], mel.shape[1]]
        ).numpy().squeeze()
    
    img2 = axes[1].imshow(shap_aggregated, aspect='auto', origin='lower', 
                         cmap='RdBu_r', vmin=-np.abs(shap_aggregated).max(), 
                         vmax=np.abs(shap_aggregated).max())
    axes[1].set_title(f'SHAP Values (red=increases Fake prob, blue=decreases)', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames (0-200)', fontsize=10)
    axes[1].set_ylabel('Mel Bins (0-80)', fontsize=10)
    fig.colorbar(img2, ax=axes[1])
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_lime_comparison(mel, explanation, prediction, show_supporting_only=False):
    """Plot mel spectrogram and LIME explanation side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Left: Mel Spectrogram (grayscale)
    axes[0].imshow(mel, aspect='auto', origin='lower', cmap='gray')
    axes[0].set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frames (0-200)', fontsize=10)
    axes[0].set_ylabel('Mel Bins (0-80)', fontsize=10)
    
    # Right: LIME Explanation
    axes[1].imshow(mel, aspect='auto', origin='lower', cmap='gray', alpha=0.5)
    
    # Extract feature importance
    lime_values = np.zeros(mel.shape[0] * mel.shape[1])
    class_idx = 1 if prediction == "Fake" else 0
    
    for feature_idx, weight in explanation.as_list(label=class_idx):
        idx = int(feature_idx.split('_')[1])
        lime_values[idx] = weight
    
    lime_heatmap = lime_values.reshape(mel.shape)
    
    # Create overlay: green for supporting, red for opposing
    overlay = np.zeros((*mel.shape, 3))
    
    # Supporting regions (positive weights) → green
    supporting_mask = lime_heatmap > 0
    overlay[supporting_mask, 1] = np.abs(lime_heatmap[supporting_mask]) / np.abs(lime_heatmap).max()
    
    if not show_supporting_only:
        # Opposing regions (negative weights) → red
        opposing_mask = lime_heatmap < 0
        overlay[opposing_mask, 0] = np.abs(lime_heatmap[opposing_mask]) / np.abs(lime_heatmap).max()
    
    axes[1].imshow(overlay, aspect='auto', origin='lower', alpha=0.6)
    axes[1].set_title(f'LIME Explanation (predicted: {prediction})', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames (0-200)', fontsize=10)
    axes[1].set_ylabel('Mel Bins (0-80)', fontsize=10)
    
    # Legend
    green_patch = mpatches.Patch(color='green', label=f'Supports "{prediction}"')
    red_patch = mpatches.Patch(color='red', label=f'Opposes "{prediction}"')
    axes[1].legend(handles=[green_patch, red_patch] if not show_supporting_only else [green_patch],
                  loc='upper right', fontsize=9)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Model selection
    model_name = st.selectbox(
        "Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,  # Default: resnet50_v3
    )
    
    st.markdown("---")
    st.markdown("### 🧑‍💻 XAI Methods")
    
    # XAI method checkboxes
    enable_gradcam = st.checkbox("Grad-CAM", value=True)
    enable_shap = st.checkbox(
        "SHAP", 
        value=True,
        help="SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction using game theory."
    )
    enable_lime = st.checkbox(
        "LIME", 
        value=True,
        help="LIME (Local Interpretable Model-agnostic Explanations) perturbs regions and measures prediction changes."
    )
    
    st.markdown("---")
    st.markdown("### XAI on which segment?")
    
    segment_mode = st.radio(
        "",
        options=[
            "🔴 Most confident (Fake)",
            "Least confident (Fake)",
            "First segment"
        ],
        index=0,
    )
    
    # Clean up mode string
    segment_mode = segment_mode.replace("🔴 ", "")
    
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">FoR Dataset · 16kHz · 2-second clips</div>',
        unsafe_allow_html=True
    )


# ── Main Panel ──────────────────────────────────────────────────────────────

# Header
st.markdown('<div class="main-header">🎙️ Deepfake Audio Detector</div>', 
           unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an audio clip to detect whether it is Real or AI-generated (Fake). '
    'Explanations via Grad-CAM, SHAP, and LIME are included.</div>',
    unsafe_allow_html=True
)

# File Upload
uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"],
    help="Upload a WAV audio file (preferably 16kHz, 2-second clips work best)"
)

if uploaded_file is not None:
    # Show file info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.write(f"**File:** {uploaded_file.name} | **Size:** {file_size_mb:.2f} MB")
    
    # Load model
    with st.spinner(f"Loading model: {model_name}..."):
        predictor = load_model(model_name)
    
    if predictor is None:
        st.stop()
    
    # Load and process audio
    with st.spinner("Processing audio..."):
        # Save uploaded file temporarily
        temp_path = Path("temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load audio
        audio, sr = librosa.load(str(temp_path), sr=TARGET_SR, mono=True)
        
        # Segment audio
        segments = segment_audio(audio, sr, SEGMENT_DURATION)
        num_segments = len(segments)
        
        # Run predictions on all segments
        segment_results = predict_segments(predictor, segments, sr)
        
        # Count segments above threshold
        segments_above_threshold = sum(1 for r in segment_results if r['probability'] >= THRESHOLD)
        
        # Select segment based on mode
        selected_result = select_segment(segment_results, segment_mode)
        
    # Display metadata
    st.write(f"**Decision threshold:** {THRESHOLD} | **Segments:** {segments_above_threshold} above threshold (total: {num_segments})")
    
    st.markdown("---")
    
    # ── Prediction Result Section ──────────────────────────────────────────
    st.markdown("## 📊 Prediction Result")
    
    prediction_label = selected_result['label']
    fake_prob = selected_result['probability']
    real_prob = 1.0 - fake_prob
    
    # Big label
    if prediction_label == "Fake":
        st.markdown(
            f'<div class="fake-label">🔴 Fake</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="real-label">🟢 Real</div>',
            unsafe_allow_html=True
        )
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{max(fake_prob, real_prob):.1%}")
    with col2:
        st.metric("Fake Probability", f"{fake_prob:.1%}")
    with col3:
        st.metric("Real Probability", f"{real_prob:.1%}")
    
    # Probability bars
    fig_bars = plot_probability_bars(fake_prob, real_prob, THRESHOLD)
    st.pyplot(fig_bars)
    plt.close(fig_bars)
    
    st.markdown("---")
    
    # ── Audio Waveform Section ─────────────────────────────────────────────
    st.markdown("## 🔊 Audio Waveform")
    
    # Audio player
    st.audio(str(temp_path))
    
    # Waveform plot
    selected_audio = segments[selected_result['segment_id']]
    fig_wave = plot_waveform(selected_audio, sr)
    st.pyplot(fig_wave)
    plt.close(fig_wave)
    
    st.markdown("---")
    
    # ── Mel Spectrogram Section ────────────────────────────────────────────
    st.markdown("## 📈 Mel Spectrogram")
    
    mel = selected_result['mel']
    fig_mel = plot_mel_spectrogram(mel)
    st.pyplot(fig_mel)
    plt.close(fig_mel)
    
    st.markdown("---")
    
    # ── XAI Explanation Sections ───────────────────────────────────────────
    
    # Prepare input for XAI methods
    x_input = preprocess_mel_for_model(mel, predictor.get_arch())
    
    # Grad-CAM
    if enable_gradcam:
        st.markdown("## 🔥 Grad-CAM Explanation")
        st.markdown(
            "*Grad-CAM highlights which time-frequency regions of the mel spectrogram drove the model's decision. "
            "The heatmap uses a **jet colormap** (blue → cyan → yellow → red). "
            "**Red/yellow = high importance** (the model focused here most). "
            "The third panel overlays the heatmap on the original spectrogram for direct comparison.*"
        )
        
        with st.spinner("Generating Grad-CAM..."):
            try:
                heatmap = get_gradcam_heatmap(predictor.get_model(), x_input)
                fig_gradcam = plot_gradcam(mel, heatmap, prediction_label, fake_prob)
                st.pyplot(fig_gradcam)
                plt.close(fig_gradcam)
            except Exception as e:
                st.error(f"Grad-CAM generation failed: {str(e)}")
                st.exception(e)
        
        st.markdown("---")
    
    # SHAP
    if enable_shap:
        st.markdown("## 🔵 SHAP Explanation")
        st.markdown(
            "*SHAP (SHapley Additive exPlanations) measures how each time-frequency cell pushes the prediction. "
            "**Red** regions increase the Fake probability. "
            "**Blue** regions decrease the Fake probability (push toward Real). "
            "The middle panel shows absolute magnitude — brighter = more influential.*"
        )
        
        with st.spinner("Generating SHAP explanation (this may take a minute)..."):
            try:
                shap_values, _ = generate_shap_explanation(
                    predictor.get_model(),
                    mel,
                    predictor.get_arch(),
                    background_samples=30
                )
                fig_shap = plot_shap_explanation(shap_values, mel, prediction_label, fake_prob)
                st.pyplot(fig_shap)
                plt.close(fig_shap)
            except Exception as e:
                st.error(f"SHAP generation failed: {str(e)}")
                st.exception(e)
        
        st.markdown("---")
    
    # LIME
    if enable_lime:
        st.markdown("## 🟡 LIME Explanation")
        st.markdown(
            "*LIME divides the spectrogram into rectangular superpixel segments, then perturbs them "
            "(replaces each segment with the mean value) to measure how predictions change. "
            "Segments that strongly affect the prediction are highlighted: "
            "**Green** = supports the predicted label, **Red** = opposes the predicted label.*"
        )
        
        with st.spinner("Generating LIME explanation (this may take ~30 seconds)..."):
            try:
                explanation = generate_lime_explanation(
                    predictor.get_model(),
                    mel,
                    predictor.get_arch(),
                    num_samples=500,
                    num_segments=40
                )
                fig_lime = plot_lime_explanation(explanation, mel, prediction_label, fake_prob)
                st.pyplot(fig_lime)
                plt.close(fig_lime)
            except Exception as e:
                st.error(f"LIME generation failed: {str(e)}")
                st.exception(e)
        
        st.markdown("---")
    
    # Clean up temp file
    temp_path.unlink(missing_ok=True)

else:
    # Landing page
    st.info("👆 Upload a WAV audio file to get started")
    
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. **Upload** a WAV audio file (2-second clips recommended)
    2. **Select** a model and XAI methods from the sidebar
    3. **View** the prediction (Real vs Fake) with confidence scores
    4. **Analyze** explanations:
       - **Grad-CAM**: Visual attention heatmap
       - **SHAP**: Feature importance (red=Fake, blue=Real)
       - **LIME**: Region-based perturbation analysis
    """)
