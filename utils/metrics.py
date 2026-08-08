"""utils/metrics.py — Evaluation Metrics & Plots (Objectives 3 & 4)"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    unique = np.unique(np.concatenate([y_true, y_pred]))
    filtered = [class_names[i] for i in unique] if class_names else [str(i) for i in unique]
    cm = confusion_matrix(y_true, y_pred, labels=unique)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(filtered)), max(6, len(filtered)//2+2)))
    for ax, data, fmt, sub in zip(axes, [cm, cm_norm], ["d", ".2f"], ["Raw", "Normalised"]):
        sns.heatmap(data, ax=ax, cmap="Blues", annot=True, fmt=fmt,
                    xticklabels=filtered, yticklabels=filtered,
                    linewidths=0.3, linecolor="gray")
        ax.set_title(f"{title} — {sub}", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Confusion matrix: {save_path}")


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-o", ms=3, label="Train")
    ax1.plot(epochs, history["val_loss"], "r-o", ms=3, label="Val")
    ax1.set_title("Bi-LSTM Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_f1"], "g-o", ms=3)
    ax2.set_title("Bi-LSTM Val F1", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1"); ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)
    plt.suptitle("RAMS — Bi-LSTM Training (Objective 3)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Training history: {save_path}")


def plot_model_comparison(results, save_path):
    models = ["XGBoost", "Bi-LSTM", "Hybrid"]
    keys = ["xgboost", "bilstm", "ensemble"]
    metrics = {"F1-W": [results[k].get("f1_weighted",0) for k in keys],
               "F1-M": [results[k].get("f1_macro",0) for k in keys],
               "Accuracy": [results[k].get("accuracy",0) for k in keys]}
    x = np.arange(len(models)); width = 0.25
    colours = ["#1f77b4","#ff7f0e","#2ca02c"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i*width, vals, width, label=metric,
                      color=colours[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.005, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x + width); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score"); ax.legend(fontsize=10)
    ax.set_title("RAMS — Model Comparison (Objective 3)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Model comparison: {save_path}")


def plot_class_distribution(y, class_names, title, save_path):
    classes, counts = np.unique(y, return_counts=True)
    labels = [class_names[c] if c < len(class_names) else f"C{c}" for c in classes]
    order = np.argsort(counts)[::-1]
    counts, labels = counts[order], [labels[i] for i in order]
    colours = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), counts, color=colours)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count"); ax.set_title(title, fontweight="bold")
    ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                f"{count:,}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Class distribution: {save_path}")


def print_metrics_table(results):
    print("\n" + "═"*65)
    print("  RAMS FRAMEWORK — RESULTS SUMMARY")
    print("═"*65)
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1-W':>10} {'F1-M':>10}")
    print(f"  {'─'*55}")
    for name, key in [("XGBoost","xgboost"),("Bi-LSTM","bilstm"),
                       ("Hybrid Ensemble","ensemble")]:
        r = results.get(key, {})
        print(f"  {name:<22} {r.get('accuracy','—'):>10} "
              f"{r.get('f1_weighted',0):>10.4f} "
              f"{r.get('f1_macro','—'):>10}")
    print("═"*65 + "\n")
