"""
demo_synthetic.py — Quick Demo: RAMS Objectives 3 & 4
======================================================
Runs the FULL pipeline on synthetic data.
No dataset downloads required. Ideal for testing/development.

Run:
    python demo_synthetic.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import BILSTM_CONFIG, XGBOOST_CONFIG, HYBRID_CONFIG, SHAP_CONFIG, RESULTS_DIR, MODELS_DIR
from data.data_loader import generate_synthetic_data, RAMSDataPreprocessor
from models.bilstm_model import BiLSTMTrainer
from models.xgboost_model import XGBoostDetector
from models.hybrid_engine import HybridDetectionEngine
from explainability.shap_explainer import RAMSExplainer
from utils.metrics import (
    plot_confusion_matrix, plot_training_history,
    plot_model_comparison, plot_class_distribution,
    print_metrics_table
)

print("\n" + "█"*65)
print("  RAMS DEMO — Objectives 3 & 4 (Synthetic Data)")
print("  Hybrid ML Detection Engine + Explainable AI (SHAP)")
print("█"*65 + "\n")

# ─── 1. Generate data ─────────────────────────────────────────────────────────
print("▶ Step 1: Generating synthetic network flow data...")
df = generate_synthetic_data(n_samples=8000, n_features=60)

# ─── 2. Preprocess ────────────────────────────────────────────────────────────
print("\n▶ Step 2: Preprocessing...")
N_FEATURES = 40   # smaller for fast demo
SEQ_LEN = 5       # shorter sequence window for demo

preprocessor = RAMSDataPreprocessor(
    n_features=N_FEATURES,
    sequence_len=SEQ_LEN,
    use_smote=True,
)
data = preprocessor.fit_transform(df)
n_classes = data["n_classes"]
class_names = list(data["classes"])
n_features = data["n_features"]
selected_features = preprocessor.selected_features or [f"f{i}" for i in range(n_features)]

print(f"\n  Classes ({n_classes}): {class_names}")

plot_class_distribution(
    data["y_train"], class_names,
    "RAMS Demo — Class Distribution",
    os.path.join(RESULTS_DIR, "demo_class_distribution.png")
)

# ─── 3. Train XGBoost ─────────────────────────────────────────────────────────
print("\n▶ Step 3: Training XGBoost...")
xgb_config = XGBOOST_CONFIG.copy()
xgb_config["n_estimators"] = 100    # reduced for demo speed

xgb_detector = XGBoostDetector(
    config=xgb_config,
    n_classes=n_classes,
    save_path=os.path.join(MODELS_DIR, "demo_xgboost.joblib")
)
xgb_detector.train(data["X_train"], data["y_train"],
                    data["X_val"], data["y_val"])

# ─── 4. Train Bi-LSTM ─────────────────────────────────────────────────────────
print("\n▶ Step 4: Training Bi-LSTM...")
bilstm_config = BILSTM_CONFIG.copy()
bilstm_config.update({
    "input_size": n_features,
    "num_classes": n_classes,
    "hidden_size": 64,       # smaller for demo
    "epochs": 10,            # fewer epochs for demo
    "sequence_len": SEQ_LEN,
    "early_stopping_patience": 3,
})

bilstm_trainer = BiLSTMTrainer(
    config=bilstm_config,
    n_classes=n_classes,
    save_path=os.path.join(MODELS_DIR, "demo_bilstm.pt")
)
history = bilstm_trainer.train(
    data["X_train_seq"], data["y_train_seq"],
    data["X_val_seq"], data["y_val_seq"]
)
plot_training_history(history,
    os.path.join(RESULTS_DIR, "demo_bilstm_training.png"))

# ─── 5. Hybrid Evaluation ─────────────────────────────────────────────────────
print("\n▶ Step 5: Hybrid Ensemble Evaluation (Objective 3)...")
engine = HybridDetectionEngine(
    bilstm_trainer=bilstm_trainer,
    xgboost_detector=xgb_detector,
    config=HYBRID_CONFIG,
    class_names=class_names,
)
results = engine.evaluate(data)
print_metrics_table(results)

preds = results["ensemble"]["predictions"]
confs = results["ensemble"]["confidence"]
labels = results["ensemble"]["labels"]

plot_confusion_matrix(labels, preds, class_names,
    os.path.join(RESULTS_DIR, "demo_confusion_matrix.png"),
    title="RAMS Demo — Hybrid Detection Engine")

plot_model_comparison(results,
    os.path.join(RESULTS_DIR, "demo_model_comparison.png"))

# ─── 6. XAI/SHAP (Objective 4) ───────────────────────────────────────────────
print("\n▶ Step 6: Explainable AI — SHAP Analysis (Objective 4)...")
explainer = RAMSExplainer(
    xgboost_model=xgb_detector,
    bilstm_trainer=bilstm_trainer,
    feature_names=selected_features,
    class_names=class_names,
    config=SHAP_CONFIG,
    output_dir=RESULTS_DIR,
)

# Background: first 50 test sequences (ideally BENIGN)
bg_seqs = data["X_test_seq"][:50]

explainer.run_full_xai_pipeline(
    X_test=data["X_test"],
    y_test=labels,
    predictions=preds,
    confidences=confs,
    X_seq_background=bg_seqs,
)

# ─── Sample Alert Demo ────────────────────────────────────────────────────────
print("\n▶ Demo Alert Generation...")
alert = engine.generate_alert({
    "label": "DDoS",
    "severity": "critical",
    "confidence": 0.94,
})
print("\n  RAMS Security Alert:")
for k, v in alert.items():
    print(f"    {k}: {v}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'█'*65}")
print(f"  DEMO COMPLETE ✓")
print(f"  All results saved to: {RESULTS_DIR}")
print(f"{'█'*65}\n")
print("  Generated files:")
for f in sorted(os.listdir(RESULTS_DIR)):
    fpath = os.path.join(RESULTS_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / 1024
        print(f"    {f:<50} {size:>8.1f} KB")
