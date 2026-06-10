"""
explainability.py — Fixed GradCAM, LIME, and SHAP for deepfake audio models.

Key fixes:
  - GradCAM: proper nested-model layer search + ReLU + smooth overlay
  - SHAP: sign-aware (red=Fake, blue=Real) with proper aggregation
  - LIME: superpixel/segment-based image explainer (not tabular)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


# ── Utility ────────────────────────────────────────────────────────────────────

def _resize_to_mel(arr_2d, mel_shape):
    """Resize a 2-D array to mel_shape using bilinear interpolation."""
    if arr_2d.shape == mel_shape:
        return arr_2d
    resized = tf.image.resize(
        arr_2d[..., np.newaxis],            # (H, W, 1)
        [mel_shape[0], mel_shape[1]],
        method="bilinear",
    ).numpy().squeeze(-1)                   # (mel_H, mel_W)
    return resized


def _find_last_conv(model):
    """
    Return the last Conv2D (or DepthwiseConv2D) layer, searching recursively
    into nested sub-models (ResNet50 / MobileNetV2 use a nested Model layer).
    Returns the layer object or None.
    """
    last = None
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            last = layer
        elif isinstance(layer, tf.keras.Model):
            # Recurse one level (handles ResNet50Base, MobileNetV2Base, etc.)
            for sublayer in layer.layers:
                if isinstance(sublayer, (tf.keras.layers.Conv2D,
                                         tf.keras.layers.DepthwiseConv2D)):
                    last = sublayer
    return last


# ── GradCAM ────────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(model, x_input, last_conv_layer_name=None):
    """
    Generate a GradCAM-style heatmap showing which time-frequency regions of
    the mel spectrogram most influenced the model's decision.

    Uses a robust two-strategy approach:
      1. Try proper GradCAM via a nested-model two-stage tape (ResNet/MobileNet).
      2. Fall back to input-gradient saliency (always works, any architecture).

    Returns:
        heatmap : 2-D float32 array, values in [0, 1]
    """
    x_input = tf.cast(x_input, tf.float32)

    # ── Strategy 1: Two-stage GradCAM for nested models ──────────────────────
    # ResNet50 / MobileNetV2 embed their conv layers inside a sub-Model.
    # We can't directly link sub-layer outputs to the outer model's inputs,
    # so we build two separate models and chain them with a watched variable.
    try:
        heatmap = _gradcam_nested(model, x_input)
        if heatmap is not None:
            return heatmap
    except Exception:
        pass

    # ── Strategy 2: Fallback — input-gradient saliency ───────────────────────
    return _gradient_saliency(model, x_input)


def _find_nested_base(model):
    """Return (base_model, last_conv_layer) for the first nested sub-Model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            last_conv = None
            for sub in layer.layers:
                if isinstance(sub, (tf.keras.layers.Conv2D,
                                    tf.keras.layers.DepthwiseConv2D)):
                    last_conv = sub
            if last_conv is not None:
                return layer, last_conv
    return None, None


def _gradcam_nested(model, x_input):
    """
    Two-stage GradCAM for models with a nested base (ResNet50, MobileNetV2).

    Stage A: x_input  → base_model → [conv_out, base_features]
    Stage B: base_features → head layers → score

    We watch `base_features` (the BOTTLENECK between base and head) inside a
    GradientTape, then back-propagate through the head to get
    d(score)/d(base_features).  We then separately get d(base_features)/d(conv_out)
    via a second tape pass to chain the gradients (chain rule).
    """
    base_model, last_conv = _find_nested_base(model)
    if base_model is None:
        return None

    # ── Sub-model A: base_model.inputs → [conv_out, base_out] ────────────────
    try:
        inner_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[last_conv.output, base_model.output],
        )
    except Exception:
        return None

    # ── Sub-model B: base_out shape → final prediction ────────────────────────
    # Replay the head layers that come AFTER the nested base model.
    base_out_shape = base_model.output_shape[1:]  # e.g. (7,7,2048)
    head_input = tf.keras.Input(shape=base_out_shape)
    x_head = head_input
    found_base = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            found_base = True
            continue
        if found_base and not isinstance(layer, tf.keras.layers.InputLayer):
            try:
                x_head = layer(x_head)
            except Exception:
                return None
    try:
        head_model = tf.keras.Model(inputs=head_input, outputs=x_head)
    except Exception:
        return None

    # ── Two-tape GradCAM ──────────────────────────────────────────────────────
    # Tape 1: get d(score)/d(base_features)
    with tf.GradientTape() as tape1:
        conv_out, base_feats = inner_model(x_input, training=False)
        tape1.watch(base_feats)
        preds = head_model(base_feats, training=False)
        score = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]

    grad_score_base = tape1.gradient(score, base_feats)    # d(score)/d(base_feats)
    if grad_score_base is None:
        return None

    # Tape 2: get d(base_features)/d(conv_out) projected onto grad_score_base
    # We use the trick: d(score)/d(conv_out) ≈ pooled channel weights × conv_out
    with tf.GradientTape() as tape2:
        conv_out2, base_feats2 = inner_model(x_input, training=False)
        tape2.watch(conv_out2)
        # Dot with upstream gradient (stop_gradient to avoid higher-order)
        loss = tf.reduce_sum(base_feats2 * tf.stop_gradient(grad_score_base))

    grad_conv = tape2.gradient(loss, conv_out2)            # (1, h, w, C)
    if grad_conv is None:
        return None

    # ── GradCAM pooling ───────────────────────────────────────────────────────
    pooled = tf.reduce_mean(grad_conv, axis=(0, 1, 2))     # (C,)
    cam    = conv_out[0] @ pooled[..., tf.newaxis]          # (h, w, 1)
    cam    = tf.squeeze(cam)                                # (h, w)
    cam    = tf.nn.relu(cam)
    cam    = cam / (tf.reduce_max(cam) + 1e-10)
    return cam.numpy().astype(np.float32)


def _gradient_saliency(model, x_input):
    """
    Robust fallback: compute d(score)/d(input) and aggregate spatially.
    Works with ANY model architecture (no layer name lookups needed).
    Produces a spatial importance map in input resolution.
    """
    x_var = tf.Variable(tf.cast(x_input, tf.float32))

    with tf.GradientTape() as tape:
        preds = model(x_var, training=False)
        score = preds[0, 0] if preds.shape[-1] == 1 else preds[0, 1]

    grads = tape.gradient(score, x_var)    # (1, H, W, C)
    if grads is None:
        raise ValueError("Could not compute any gradients for this model.")

    grads = grads[0]                       # (H, W, C)

    # ReLU + mean over channels → (H, W)
    heatmap = tf.reduce_mean(tf.nn.relu(grads), axis=-1)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy().astype(np.float32)


# ── SHAP Implementation ────────────────────────────────────────────────────────

def generate_shap_explanation(model, mel, arch, background_samples=30):
    """
    Generate SHAP explanation using GradientExplainer (works with any Keras model).

    Args:
        model             : Keras model
        mel               : Mel spectrogram (80, 201) float32
        arch              : Architecture string
        background_samples: Number of background reference samples

    Returns:
        shap_values   : ndarray shape matching x_processed (with batch dim)
        expected_value: scalar float
    """
    from src.inference.predictor import preprocess_mel_for_model

    x_processed = preprocess_mel_for_model(mel, arch)      # (1, H, W, C)
    input_shape  = x_processed.shape[1:]                    # (H, W, C)

    # Background: mix of zeros and Gaussian noise (mean≈mel range)
    np.random.seed(42)
    bg_list = []
    half = background_samples // 2

    for _ in range(half):
        blank = np.zeros((80, 201), dtype=np.float32)
        bg_list.append(preprocess_mel_for_model(blank, arch))

    for _ in range(background_samples - half):
        noise = np.clip(np.random.randn(80, 201).astype(np.float32) * 0.15 + 0.5, 0, 1)
        bg_list.append(preprocess_mel_for_model(noise, arch))

    background = np.vstack(bg_list)   # (background_samples, H, W, C)

    # Try GradientExplainer first (robust), fall back to GradientXInput
    try:
        import shap
        explainer   = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(x_processed)   # list or ndarray
    except Exception:
        shap_values = _gradient_x_input(model, x_processed)

    expected_value = float(np.mean(model.predict(background, verbose=0)))
    return shap_values, expected_value


def _gradient_x_input(model, x_input):
    """Gradient × Input attribution (fast fallback)."""
    x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        preds = model(x_tensor, training=False)
        score = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]
    grads = tape.gradient(score, x_tensor)
    return (grads * x_tensor).numpy()


# ── LIME Implementation ────────────────────────────────────────────────────────

def generate_lime_explanation(model, mel, arch, num_samples=500, num_segments=40):
    """
    Generate a LIME explanation using image-style superpixel segmentation.

    Each "superpixel" is a rectangular patch of the mel spectrogram.
    We perturb patches (set them to mean value) and measure how the
    prediction changes — this reveals which regions drive the decision.

    Args:
        model        : Keras model
        mel          : Mel spectrogram (80, 201) float32
        arch         : Architecture string
        num_samples  : Number of perturbed samples
        num_segments : Number of rectangular segments to create

    Returns:
        dict with keys:
            'weights'    : 2-D array (mel.shape), importance per pixel
            'segments'   : 2-D int array, segment IDs per pixel
            'labels'     : list of (segment_id, weight) sorted by |weight|
            'prediction' : scalar prediction probability
    """
    from src.inference.predictor import preprocess_mel_for_model

    H, W = mel.shape

    # ── Create rectangular superpixel grid ──────────────────────────────────
    n_rows = max(4, int(np.sqrt(num_segments * H / W)))
    n_cols = max(4, int(np.sqrt(num_segments * W / H)))
    segments = np.zeros((H, W), dtype=np.int32)
    seg_id = 0
    row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=int)
    for ri in range(n_rows):
        for ci in range(n_cols):
            r0, r1 = row_edges[ri], row_edges[ri + 1]
            c0, c1 = col_edges[ci], col_edges[ci + 1]
            segments[r0:r1, c0:c1] = seg_id
            seg_id += 1

    n_seg      = seg_id
    mel_mean   = float(mel.mean())

    # ── Prediction function ──────────────────────────────────────────────────
    def predict_fn(masks):
        """
        masks: (n_samples, n_seg) binary array.
        Returns (n_samples, 2) — [prob_real, prob_fake].
        """
        batch_inputs = []
        for mask_row in masks:
            perturbed = mel.copy()
            for s in range(n_seg):
                if mask_row[s] == 0:               # segment is OFF → replace
                    perturbed[segments == s] = mel_mean
            x = preprocess_mel_for_model(perturbed, arch)
            batch_inputs.append(x[0])              # (H, W, C)

        batch_arr = np.stack(batch_inputs, axis=0)  # (B, H, W, C)
        probs     = model.predict(batch_arr, verbose=0).flatten()   # (B,)
        return np.column_stack([1 - probs, probs])  # (B, 2)

    # ── LIME sampling ────────────────────────────────────────────────────────
    np.random.seed(0)
    masks   = np.random.randint(0, 2, size=(num_samples, n_seg))
    preds   = predict_fn(masks)                     # (num_samples, 2)
    fake_probs = preds[:, 1]                        # predict "Fake"

    # ── Fit a weighted linear model per segment ──────────────────────────────
    # Similarity kernel: cosine distance from all-ones (= original image)
    original_mask = np.ones(n_seg)
    distances     = np.sqrt(np.sum((masks - original_mask) ** 2, axis=1))
    kernel_width  = 0.25 * np.sqrt(n_seg)
    weights       = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

    from numpy.linalg import lstsq
    W_diag = np.diag(weights)
    A      = masks.T @ W_diag @ masks      # (n_seg, n_seg)
    b      = masks.T @ W_diag @ fake_probs # (n_seg,)
    try:
        coefs, _, _, _ = lstsq(A, b, rcond=None)
    except Exception:
        coefs = np.zeros(n_seg)

    # ── Build pixel-level weight map ─────────────────────────────────────────
    weight_map = np.zeros((H, W), dtype=np.float32)
    for s in range(n_seg):
        weight_map[segments == s] = coefs[s]

    # Baseline prediction (all segments ON = original)
    x_orig    = preprocess_mel_for_model(mel, arch)
    base_prob = float(model.predict(x_orig, verbose=0).flatten()[0])

    labels = sorted(enumerate(coefs), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "weights":    weight_map,
        "segments":   segments,
        "labels":     labels,
        "prediction": base_prob,
        "coefs":      coefs,
    }


# ── Visualization Functions ────────────────────────────────────────────────────

def plot_gradcam(mel, heatmap, prediction_label, probability, save_path=None):
    """
    Three-panel GradCAM figure:
      1. Original mel spectrogram
      2. Standalone GradCAM heatmap (jet colormap)
      3. GradCAM heatmap overlaid on the mel spectrogram
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")

    heat = _resize_to_mel(heatmap, mel.shape)

    # ── Panel 1: Original mel ────────────────────────────────────────────────
    im0 = axes[0].imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Original Mel Spectrogram", color="white",
                       fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time Frames", color="white")
    axes[0].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[0])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # ── Panel 2: GradCAM heatmap only ───────────────────────────────────────
    im1 = axes[1].imshow(heat, aspect="auto", origin="lower",
                          cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GradCAM Importance Heatmap\n"
                       "(Red = high importance for decision)",
                       color="white", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Time Frames", color="white")
    axes[1].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[1])
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("Importance", color="white")
    cb1.ax.yaxis.set_tick_params(color="white")

    # ── Panel 3: Overlay ─────────────────────────────────────────────────────
    # Normalize mel for display
    mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-10)
    axes[2].imshow(mel_norm, aspect="auto", origin="lower",
                   cmap="gray", alpha=0.55)
    im2 = axes[2].imshow(heat, aspect="auto", origin="lower",
                          cmap="jet", alpha=0.6, vmin=0, vmax=1)
    label_color = "#FF4B4B" if prediction_label == "Fake" else "#00C897"
    axes[2].set_title(
        f"GradCAM Overlay\nPrediction: {prediction_label}  ({probability:.1%})",
        color=label_color, fontsize=12, fontweight="bold",
    )
    axes[2].set_xlabel("Time Frames", color="white")
    axes[2].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[2])
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label("GradCAM Score", color="white")
    cb2.ax.yaxis.set_tick_params(color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_explanation(shap_values, mel, prediction_label, probability, save_path=None):
    """
    Three-panel SHAP figure:
      1. SHAP values (RdBu — red pushes toward Fake, blue toward Real)
      2. Absolute importance heatmap
      3. Overlay on mel spectrogram
    """
    # ── Aggregate SHAP values to 2-D ────────────────────────────────────────
    sv = shap_values

    # If list (one array per output), take the first (Fake class)
    if isinstance(sv, list):
        sv = sv[0] if len(sv) > 0 else sv

    sv = np.array(sv)

    # Remove batch dimension if it's 1
    if sv.shape[0] == 1:
        sv = sv[0]
        
    # Remove class dimension if it's an explicit 1 or 2 at the end (e.g., shape ends in 1 or 2)
    # But only if it has enough dimensions to still represent an image
    if sv.ndim >= 3 and sv.shape[-1] <= 2:
        sv = sv[..., -1]
        
    # Sum over channel dimension if it exists → signed (H, W)
    if sv.ndim >= 3:
        sv_signed = sv.sum(axis=-1)
    else:
        sv_signed = sv

    sv_abs = np.abs(sv_signed)

    # Resize to mel shape
    sv_signed = _resize_to_mel(sv_signed, mel.shape)
    sv_abs    = _resize_to_mel(sv_abs,    mel.shape)

    # Normalize absolute for display
    sv_abs_norm = sv_abs / (sv_abs.max() + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")

    # ── Panel 1: Signed SHAP (RdBu) ─────────────────────────────────────────
    vmax = np.abs(sv_signed).max() + 1e-10
    im0  = axes[0].imshow(sv_signed, aspect="auto", origin="lower",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(
        "SHAP Values\n(Red → pushes Fake↑  |  Blue → pushes Real↑)",
        color="white", fontsize=12, fontweight="bold",
    )
    axes[0].set_xlabel("Time Frames", color="white")
    axes[0].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[0])
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("SHAP value", color="white")

    # ── Panel 2: Absolute importance ─────────────────────────────────────────
    im1 = axes[1].imshow(sv_abs_norm, aspect="auto", origin="lower", cmap="hot")
    axes[1].set_title(
        "SHAP |Absolute| Importance\n(Brighter = more influential region)",
        color="white", fontsize=12, fontweight="bold",
    )
    axes[1].set_xlabel("Time Frames", color="white")
    axes[1].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[1])
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("|SHAP|", color="white")

    # ── Panel 3: Overlay on mel ───────────────────────────────────────────────
    mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-10)
    axes[2].imshow(mel_norm, aspect="auto", origin="lower", cmap="gray", alpha=0.55)
    im2 = axes[2].imshow(sv_abs_norm, aspect="auto", origin="lower",
                          cmap="hot", alpha=0.6)
    label_color = "#FF4B4B" if prediction_label == "Fake" else "#00C897"
    axes[2].set_title(
        f"SHAP Overlay\nPrediction: {prediction_label}  ({probability:.1%})",
        color=label_color, fontsize=12, fontweight="bold",
    )
    axes[2].set_xlabel("Time Frames", color="white")
    axes[2].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[2])
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label("|SHAP|", color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lime_explanation(explanation, mel, prediction_label, probability, save_path=None):
    """
    Three-panel LIME figure:
      1. Signed weight map  (green = supports prediction, red = opposes)
      2. Segmentation overlay showing top contributing regions
      3. Overlay on mel spectrogram
    """
    # explanation is now a dict from generate_lime_explanation()
    weight_map = explanation["weights"]    # (H, W) signed floats
    segments   = explanation["segments"]   # (H, W) int segment IDs
    coefs      = explanation["coefs"]      # (n_seg,) per-segment weights

    # ── Panel 1: Signed importance ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")

    vmax = np.abs(weight_map).max() + 1e-10
    im0  = axes[0].imshow(weight_map, aspect="auto", origin="lower",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(
        "LIME Feature Weights\n(Green/Pos → supports prediction  |  Red/Neg → opposes)",
        color="white", fontsize=11, fontweight="bold",
    )
    axes[0].set_xlabel("Time Frames", color="white")
    axes[0].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[0])
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("Weight", color="white")

    # ── Panel 2: Top regions highlighted on mel ──────────────────────────────
    # Find top-K supporting and opposing segments
    n_top = min(5, len(coefs) // 4)
    sorted_idx = np.argsort(coefs)
    top_support = sorted_idx[-n_top:]   # most positive
    top_oppose  = sorted_idx[:n_top]    # most negative

    mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-10)
    axes[1].imshow(mel_norm, aspect="auto", origin="lower", cmap="gray", alpha=0.7)

    # Green overlay for supporting, red for opposing
    overlay = np.zeros((*mel.shape, 4))  # RGBA
    for s in top_support:
        mask = segments == s
        overlay[mask] = [0.0, 0.9, 0.3, 0.6]   # green
    for s in top_oppose:
        mask = segments == s
        overlay[mask] = [0.9, 0.1, 0.1, 0.6]   # red

    axes[1].imshow(overlay, aspect="auto", origin="lower")
    green_patch = mpatches.Patch(color="#00E64D", label=f'Supports "{prediction_label}"')
    red_patch   = mpatches.Patch(color="#E61A1A", label=f'Opposes "{prediction_label}"')
    axes[1].legend(handles=[green_patch, red_patch], loc="upper right",
                   fontsize=9, facecolor="#1e222a", labelcolor="white")
    axes[1].set_title(
        f"LIME Top-{n_top} Key Regions\n(Green = supporting, Red = opposing)",
        color="white", fontsize=12, fontweight="bold",
    )
    axes[1].set_xlabel("Time Frames", color="white")
    axes[1].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[1])

    # ── Panel 3: Absolute importance overlay ─────────────────────────────────
    wt_abs  = np.abs(weight_map)
    wt_norm = wt_abs / (wt_abs.max() + 1e-10)

    axes[2].imshow(mel_norm, aspect="auto", origin="lower", cmap="gray", alpha=0.55)
    im2 = axes[2].imshow(wt_norm, aspect="auto", origin="lower",
                          cmap="hot", alpha=0.6)
    label_color = "#FF4B4B" if prediction_label == "Fake" else "#00C897"
    axes[2].set_title(
        f"LIME Importance Overlay\nPrediction: {prediction_label}  ({probability:.1%})",
        color=label_color, fontsize=12, fontweight="bold",
    )
    axes[2].set_xlabel("Time Frames", color="white")
    axes[2].set_ylabel("Mel Frequency Bins", color="white")
    _style_ax(axes[2])
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label("Importance", color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Style helper ───────────────────────────────────────────────────────────────

def _style_ax(ax):
    """Apply dark-theme styling to a matplotlib axis."""
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
