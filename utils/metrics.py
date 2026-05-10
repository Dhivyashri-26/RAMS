"""
utils/metrics.py — Evaluation Metrics & Visualization
RAMS Framework — Objectives 3 & 4
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list, save_path: str,
                           title: str = "Confusion Matrix"):
    """Normalised confusion matrix with class labels."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n), max(6, n // 2 + 2)))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw Counts", "Normalised"]
    ):
        mask = np.zeros_like(data, dtype=bool)
        sns.heatmap(data, ax=ax, cmap="Blues", annot=True, fmt=fmt,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.3, linecolor="gray",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"{title} — {subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    plt.suptitle("RAMS Hybrid Detection Engine — Confusion Matrix\n"
                 "Objective 3: Multi-class Threat Classification",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Confusion matrix saved: {save_path}")


def plot_training_history(history: dict, save_path: str):
    """Plot Bi-LSTM training loss and F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_title("Bi-LSTM Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_f1"], "g-o", markersize=3, label="Val F1 (weighted)")
    ax2.set_title("Bi-LSTM Validation F1 Score", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("RAMS — Bi-LSTM Training Progress (Objective 3)",
                  fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Training history saved: {save_path}")


def plot_model_comparison(results: dict, save_path: str):
    """
    Bar chart comparing XGBoost vs Bi-LSTM vs Hybrid on key metrics.
    """
    models = ["XGBoost", "Bi-LSTM", "Hybrid Ensemble"]
    metrics = {
        "F1 (Weighted)": [
            results["xgboost"]["f1_weighted"],
            results["bilstm"]["f1_weighted"],
            results["ensemble"]["f1_weighted"],
        ],
        "F1 (Macro)": [
            results["xgboost"].get("f1_macro", 0),
            0,   # LSTM macro not computed separately
            results["ensemble"]["f1_macro"],
        ],
        "Accuracy": [
            results["xgboost"]["accuracy"],
            0,
            results["ensemble"]["accuracy"],
        ],
    }

    x = np.arange(len(models))
    width = 0.25
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=colours[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RAMS — Model Comparison: XGBoost vs Bi-LSTM vs Hybrid\n"
                  "Objective 3: Hybrid ML Detection Engine",
                  fontsize=12, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Model comparison chart saved: {save_path}")


def plot_class_distribution(y: np.ndarray, class_names: list,
                              title: str, save_path: str):
    """Visualise class imbalance in the dataset."""
    classes, counts = np.unique(y, return_counts=True)
    labels = [class_names[c] if c < len(class_names) else f"Class {c}"
              for c in classes]

    # Sort by count descending
    order = np.argsort(counts)[::-1]
    counts = counts[order]
    labels = [labels[i] for i in order]

    colours = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), counts, color=colours)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Sample Count", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yscale("log")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.1,
                f"{count:,}", ha="center", va="bottom",
                fontsize=7, rotation=0)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Class distribution saved: {save_path}")


def print_metrics_table(results: dict):
    """Print a formatted metrics comparison table to console."""
    print("\n" + "═"*65)
    print("  RAMS FRAMEWORK — OBJECTIVE 3 RESULTS SUMMARY")
    print("═"*65)
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}")
    print(f"  {'─'*57}")

    xgb = results.get("xgboost", {})
    lstm = results.get("bilstm", {})
    ens = results.get("ensemble", {})

    print(f"  {'XGBoost':<22} "
          f"{xgb.get('accuracy', 0):>10.4f} "
          f"{xgb.get('f1_weighted', 0):>12.4f} "
          f"{xgb.get('f1_macro', 0):>10.4f}")
    print(f"  {'Bi-LSTM':<22} "
          f"{'—':>10} "
          f"{lstm.get('f1_weighted', 0):>12.4f} "
          f"{'—':>10}")
    print(f"  {'Hybrid Ensemble':<22} "
          f"{ens.get('accuracy', 0):>10.4f} "
          f"{ens.get('f1_weighted', 0):>12.4f} "
          f"{ens.get('f1_macro', 0):>10.4f}")
    print("═"*65 + "\n")
