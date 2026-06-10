"""
metrics.py — Shared evaluation utilities for all training notebooks.

Usage:
    from src.evaluation.metrics import evaluate_model

    results = evaluate_model(
        model, test_ds,
        model_name="MobileNetV2",
        cmap="Greens",
        output_dir="../outputs",
    )
    # returns dict: accuracy, f1, auc, confusion_matrix
"""

import os
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from src.evaluation.visualize import plot_confusion_matrix


def evaluate_model(
    model,
    test_ds,
    model_name: str,
    cmap: str = "Blues",
    output_dir: str = "../outputs",
    threshold: float = 0.5,
) -> dict:
    """
    Collect predictions on a test dataset, print a metrics report, and save
    a confusion-matrix plot.

    Args
    ----
    model      : compiled Keras model (expects preprocessed batches)
    test_ds    : tf.data.Dataset yielding (x, y) batches — already preprocessed
    model_name : display name used in print headers and saved filenames
    cmap       : matplotlib colormap for the confusion matrix (default 'Blues')
    output_dir : directory where the PNG will be saved
    threshold  : probability threshold for binary decision (default 0.5)

    Returns
    -------
    dict with keys: 'accuracy', 'f1', 'auc', 'confusion_matrix'
    """
    print(f"Collecting predictions on test set  [{model_name}]...")

    y_true, y_prob = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0).flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = (y_pred == y_true).mean()
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred)

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 45
    print(f"\n{sep}")
    print(f"  {model_name} — Test Results")
    print(sep)
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f} %)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(sep)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(output_dir, f"{safe_name}_confusion.png")
    plot_confusion_matrix(
        cm,
        title=f"Confusion Matrix — {model_name}",
        save_path=save_path,
        cmap=cmap,
    )

    return {"accuracy": acc, "f1": f1, "auc": auc, "confusion_matrix": cm}
