"""
main.py — RAMS Framework: Objectives 3 & 4 Full Pipeline
=========================================================
Hybrid ML Detection Engine + Explainable AI (XAI)

Usage:
    # Synthetic data demo (no downloads):
    python main.py --dataset synthetic

    # CIC-IDS2017 (download first):
    python main.py --dataset cicids2017 --data_path ./data/cicids2017/

    # TON_IoT:
    python main.py --dataset toniot --data_path ./data/toniot/

    # Skip SHAP (faster run):
    python main.py --dataset synthetic --skip_xai

    # Small quick test:
    python main.py --dataset synthetic --n_samples 2000 --epochs 5
"""

import os
import sys
import argparse
import time
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BILSTM_CONFIG, XGBOOST_CONFIG, HYBRID_CONFIG,
    SHAP_CONFIG, RESULTS_DIR, MODELS_DIR
)
from data.data_loader import load_dataset, RAMSDataPreprocessor
from models.bilstm_model import BiLSTMTrainer
from models.xgboost_model import XGBoostDetector
from models.hybrid_engine import HybridDetectionEngine
from explainability.shap_explainer import RAMSExplainer
from utils.metrics import (
    plot_confusion_matrix, plot_training_history,
    plot_model_comparison, plot_class_distribution,
    print_metrics_table
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAMS Framework — Objectives 3 & 4"
    )
    parser.add_argument("--dataset", default="synthetic",
                        choices=["cicids2017", "toniot", "synthetic"],
                        help="Dataset to use")
    parser.add_argument("--data_path", default=None,
                        help="Path to dataset CSV files")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Samples for synthetic dataset")
    parser.add_argument("--n_features", type=int, default=60,
                        help="Feature dimensionality after selection")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override Bi-LSTM training epochs")
    parser.add_argument("--skip_bilstm", action="store_true",
                        help="Skip Bi-LSTM training (XGBoost + SHAP only)")
    parser.add_argument("--skip_xai", action="store_true",
                        help="Skip XAI/SHAP computation")
    parser.add_argument("--results_dir", default=RESULTS_DIR,
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("\n" + "█"*65)
    print("  RAMS FRAMEWORK — RESILIENT AUTONOMOUS MULTI-TIER SECURITY")
    print("  Objectives 3 & 4: Hybrid ML Engine + Explainable AI (XAI)")
    print("█"*65)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Results:    {args.results_dir}")
    print(f"  Skip Bi-LSTM: {args.skip_bilstm}")
    print(f"  Skip XAI:   {args.skip_xai}")
    print("█"*65 + "\n")

    # ─── STEP 1: Load Dataset ─────────────────────────────────────────────────
    print("▶ STEP 1: Loading Dataset")
    df = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        n_samples=args.n_samples
    )

    # ─── STEP 2: Preprocess ───────────────────────────────────────────────────
    print("\n▶ STEP 2: Preprocessing (Feature Selection + Scaling + Sequencing)")
    preprocessor = RAMSDataPreprocessor(
        n_features=args.n_features,
        sequence_len=BILSTM_CONFIG["sequence_len"],
        use_smote=True,
    )
    data = preprocessor.fit_transform(df)

    n_classes = data["n_classes"]
    class_names = list(data["classes"])
    n_features = data["n_features"]
    selected_features = preprocessor.selected_features or \
        [f"f{i}" for i in range(n_features)]

    print(f"\n  Classes ({n_classes}): {class_names}")
    print(f"  Features selected: {n_features}")

    # Visualise class distribution
    plot_class_distribution(
        data["y_train"], class_names,
        "RAMS — Training Set Class Distribution (CIC-IDS2017 style)",
        os.path.join(args.results_dir, "class_distribution.png")
    )

    # ─── STEP 3: Train XGBoost ────────────────────────────────────────────────
    print("\n▶ STEP 3: Training XGBoost Detector")
    xgb_detector = XGBoostDetector(
        config=XGBOOST_CONFIG,
        n_classes=n_classes,
        save_path=os.path.join(MODELS_DIR, "xgboost_model.joblib")
    )
    xgb_detector.train(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"]
    )

    # ─── STEP 4: Train Bi-LSTM ────────────────────────────────────────────────
    bilstm_config = BILSTM_CONFIG.copy()
    bilstm_config["input_size"] = n_features
    bilstm_config["num_classes"] = n_classes
    if args.epochs:
        bilstm_config["epochs"] = args.epochs

    bilstm_trainer = None
    bilstm_history = None

    if not args.skip_bilstm:
        print("\n▶ STEP 4: Training Bi-LSTM Detector")
        bilstm_trainer = BiLSTMTrainer(
            config=bilstm_config,
            n_classes=n_classes,
            save_path=os.path.join(MODELS_DIR, "bilstm_best.pt")
        )
        bilstm_history = bilstm_trainer.train(
            data["X_train_seq"], data["y_train_seq"],
            data["X_val_seq"], data["y_val_seq"]
        )
        plot_training_history(
            bilstm_history,
            os.path.join(args.results_dir, "bilstm_training_history.png")
        )
    else:
        print("\n▶ STEP 4: Bi-LSTM skipped (--skip_bilstm flag)")

    # ─── STEP 5: Hybrid Ensemble Evaluation ───────────────────────────────────
    print("\n▶ STEP 5: Hybrid Ensemble Evaluation (Objective 3)")

    hybrid_engine = HybridDetectionEngine(
        bilstm_trainer=bilstm_trainer,
        xgboost_detector=xgb_detector,
        config=HYBRID_CONFIG,
        class_names=class_names,
    )

    if bilstm_trainer is not None:
        results = hybrid_engine.evaluate(data)
        ensemble_preds = results["ensemble"]["predictions"]
        ensemble_conf = results["ensemble"]["confidence"]
        y_test_aligned = results["ensemble"]["labels"]
    else:
        # XGBoost-only evaluation if Bi-LSTM skipped
        xgb_results = xgb_detector.evaluate(
            data["X_test"], data["y_test"], class_names
        )
        results = {
            "xgboost": xgb_results,
            "bilstm": {"f1_weighted": 0.0},
            "ensemble": {
                **xgb_results,
                "f1_macro": xgb_results.get("f1_macro", 0),
                "confidence": xgb_results["probabilities"].max(axis=1)
            },
        }
        ensemble_preds = xgb_results["predictions"]
        ensemble_conf = xgb_results["probabilities"].max(axis=1)
        y_test_aligned = data["y_test"]

    print_metrics_table(results)

    # Confusion matrix
    plot_confusion_matrix(
        y_test_aligned, ensemble_preds, class_names,
        os.path.join(args.results_dir, "confusion_matrix.png"),
        title="RAMS Hybrid Engine"
    )

    # Model comparison chart
    plot_model_comparison(
        results,
        os.path.join(args.results_dir, "model_comparison.png")
    )

    # ─── STEP 6: Explainable AI (Objective 4) ─────────────────────────────────
    if not args.skip_xai:
        print("\n▶ STEP 6: Explainable AI — SHAP Analysis (Objective 4)")

        explainer = RAMSExplainer(
            xgboost_model=xgb_detector,
            bilstm_trainer=bilstm_trainer,
            feature_names=selected_features,
            class_names=class_names,
            config=SHAP_CONFIG,
            output_dir=args.results_dir,
        )

        # Use BENIGN test samples as background for DeepExplainer
        if bilstm_trainer is not None and len(data["X_test_seq"]) > 0:
            benign_class_idx = (list(class_names).index("BENIGN")
                                if "BENIGN" in class_names else 0)
            benign_mask = y_test_aligned == benign_class_idx
            if benign_mask.sum() > 0:
                bg_seqs = data["X_test_seq"][:min(100, len(data["X_test_seq"]))]
            else:
                bg_seqs = data["X_test_seq"][:min(50, len(data["X_test_seq"]))]
        else:
            bg_seqs = None

        explainer.run_full_xai_pipeline(
            X_test=data["X_test"],
            y_test=y_test_aligned,
            predictions=ensemble_preds,
            confidences=ensemble_conf,
            X_seq_background=bg_seqs,
        )
    else:
        print("\n▶ STEP 6: XAI skipped (--skip_xai flag)")

    # ─── Demo Alert ───────────────────────────────────────────────────────────
    print("\n▶ DEMO: Generating sample security alert + MTD trigger")
    sample_alert = hybrid_engine.generate_alert({
        "label": class_names[ensemble_preds[0]] if len(ensemble_preds) > 0 else "DDoS",
        "severity": "critical",
        "confidence": float(ensemble_conf[0]) if len(ensemble_conf) > 0 else 0.92,
    })
    print(f"\n  Sample Alert:")
    for k, v in sample_alert.items():
        print(f"    {k}: {v}")

    # ─── Final Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'█'*65}")
    print(f"  RAMS Objectives 3 & 4 — COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {args.results_dir}")
    print(f"{'█'*65}\n")

    print("  Output files:")
    for fname in sorted(os.listdir(args.results_dir)):
        fpath = os.path.join(args.results_dir, fname)
        size = os.path.getsize(fpath) / 1024
        print(f"    {fname:<45} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
