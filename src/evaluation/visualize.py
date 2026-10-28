"""
visualize.py — Shared plotting utilities for training notebooks.

Usage:
    from src.evaluation.visualize import plot_history, plot_confusion_matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, title: str, save_path: str) -> None:
    """
    Plot accuracy and loss curves from a Keras training history object.

    Args
    ----
    history   : keras History object returned by model.fit()
    title     : suptitle for the figure
    save_path : absolute or relative path where the PNG will be saved
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs_ran, history.history["accuracy"],
                 color="#2196F3", linewidth=2, label="Train")
    axes[0].plot(epochs_ran, history.history["val_accuracy"],
                 color="#4CAF50", linewidth=2, linestyle="--", label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs_ran, history.history["loss"],
                 color="#2196F3", linewidth=2, label="Train")
    axes[1].plot(epochs_ran, history.history["val_loss"],
                 color="#F44336", linewidth=2, linestyle="--", label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def plot_confusion_matrix(cm: np.ndarray, title: str,
                          save_path: str, cmap: str = "Blues") -> None:
    """
    Plot and save a 2×2 confusion matrix for binary classification.

    Args
    ----
    cm        : 2D numpy array from sklearn.metrics.confusion_matrix
    title     : figure title
    save_path : path where the PNG will be saved
    cmap      : matplotlib colormap name (default 'Blues')
    """
    classes = ["Real (0)", "Fake (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")
