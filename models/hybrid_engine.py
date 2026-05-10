"""
models/hybrid_engine.py — Hybrid ML Detection Engine
RAMS Framework — Objective 3

Combines Bi-LSTM (temporal/sequential) + XGBoost (tabular/feature-based)
into a weighted probability ensemble.

Design rationale (from FUSE-Net + BoT-EnsIDS papers):
  - XGBoost: excellent at catching volumetric attacks (DDoS, PortScan)
    from single-flow features like packet count, bytes/s
  - Bi-LSTM: excels at sequential/temporal attacks (slow DoS, botnet
    C2 beaconing, staged infiltration) that need window context
  - Weighted fusion: combines strengths, raises confidence threshold
    before flagging as critical (reduces false positives)

Output feeds directly into:
  - SHAP Explainer (Objective 4)
  - MTD trigger (Objective 5) via THREAT_SEVERITY mapping
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime
from typing import Optional

from sklearn.metrics import classification_report

try:
    from config import HYBRID_CONFIG, THREAT_SEVERITY, RESULTS_DIR
    from models.bilstm_model import BiLSTMTrainer, compute_class_weights
    from models.xgboost_model import XGBoostDetector
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import HYBRID_CONFIG, THREAT_SEVERITY, RESULTS_DIR
    from models.bilstm_model import BiLSTMTrainer, compute_class_weights
    from models.xgboost_model import XGBoostDetector


class HybridDetectionEngine:
    """
    RAMS Cloud Tier — Hybrid ML Detection Engine (Objective 3).

    Architecture:
      ┌─────────────────────────────────────────────────────────┐
      │               Anomalous flows from Edge                  │
      │              (Tier 2 forwarded traffic)                  │
      └──────────────┬──────────────────┬───────────────────────┘
                     │                  │
             ┌───────▼──────┐   ┌───────▼──────┐
             │   XGBoost    │   │   Bi-LSTM     │
             │  (tabular)   │   │ (sequential)  │
             │  P(class|x)  │   │  P(class|seq) │
             └───────┬──────┘   └───────┬───────┘
                     │                  │
             ┌───────▼──────────────────▼───────┐
             │     Weighted Probability Fusion    │
             │  w_xgb=0.55, w_lstm=0.45          │
             └───────────────┬───────────────────┘
                             │
             ┌───────────────▼───────────────────┐
             │    Threat Classification + Alert   │
             │    → SHAP Explanation (Obj 4)      │
             │    → MTD Trigger (Obj 5)           │
             └───────────────────────────────────┘
    """

    def __init__(self, bilstm_trainer: BiLSTMTrainer,
                 xgboost_detector: XGBoostDetector,
                 config: dict = None,
                 class_names: list = None):
        self.bilstm = bilstm_trainer
        self.xgboost = xgboost_detector
        self.config = config or HYBRID_CONFIG
        self.class_names = class_names
        self.n_classes = None

        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

    def train(self, data: dict, bilstm_config: dict,
              xgboost_config: dict) -> dict:
        """
        Train both models using preprocessed data splits.

        Args:
            data: output from RAMSDataPreprocessor.fit_transform()
            bilstm_config: Bi-LSTM hyperparameters
            xgboost_config: XGBoost hyperparameters

        Returns:
            Training history for both models
        """
        self.n_classes = data["n_classes"]
        self.class_names = list(data["classes"])

        print("\n" + "="*60)
        print(" RAMS — Objective 3: Hybrid Detection Engine Training")
        print("="*60)

        # ── Train XGBoost ─────────────────────────────────────────
        print("\n[Hybrid] Phase 1: XGBoost Training")
        self.xgboost.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"]
        )

        # ── Train Bi-LSTM ─────────────────────────────────────────
        print("\n[Hybrid] Phase 2: Bi-LSTM Training")
        class_weights = compute_class_weights(data["y_train_seq"])
        bilstm_history = self.bilstm.train(
            data["X_train_seq"], data["y_train_seq"],
            data["X_val_seq"], data["y_val_seq"],
        )

        print("\n[Hybrid] Both models trained successfully.")
        return {"bilstm_history": bilstm_history}

    def _align_probs(self, xgb_probs: np.ndarray,
                     lstm_probs: np.ndarray) -> tuple:
        """
        Align probability arrays when XGBoost (flat) and Bi-LSTM (sequential)
        have different sample counts due to windowing.

        XGBoost processes all n flows; Bi-LSTM processes (n - seq_len) windows.
        We align by trimming XGBoost to match LSTM output.
        """
        n_lstm = len(lstm_probs)
        n_xgb = len(xgb_probs)

        if n_xgb > n_lstm:
            # Trim the beginning of XGBoost predictions (windowing offset)
            xgb_probs = xgb_probs[n_xgb - n_lstm:]

        elif n_lstm > n_xgb:
            lstm_probs = lstm_probs[n_lstm - n_xgb:]

        return xgb_probs, lstm_probs

    def predict_ensemble(self, X_flat: np.ndarray,
                          X_seq: np.ndarray) -> dict:
        """
        Run hybrid ensemble inference.

        Args:
            X_flat: (n, n_features) — for XGBoost
            X_seq:  (n, seq_len, n_features) — for Bi-LSTM

        Returns:
            dict with predictions, probabilities, confidence, severity
        """
        # Get probabilities from both models
        xgb_probs = self.xgboost.predict_proba(X_flat)   # (n, n_classes)
        lstm_probs = self.bilstm.predict_proba(X_seq)     # (m, n_classes)

        # Handle n_classes mismatch (edge case with unseen classes)
        if xgb_probs.shape[1] != lstm_probs.shape[1]:
            min_classes = min(xgb_probs.shape[1], lstm_probs.shape[1])
            xgb_probs = xgb_probs[:, :min_classes]
            lstm_probs = lstm_probs[:, :min_classes]

        # Align sample counts
        xgb_probs, lstm_probs = self._align_probs(xgb_probs, lstm_probs)

        # Weighted fusion
        w_xgb = self.config["xgboost_weight"]
        w_lstm = self.config["bilstm_weight"]
        fused_probs = w_xgb * xgb_probs + w_lstm * lstm_probs

        # Final predictions
        predictions = np.argmax(fused_probs, axis=1)
        confidence = np.max(fused_probs, axis=1)

        # Flag uncertain predictions
        conf_threshold = self.config["confidence_threshold"]
        uncertain_mask = confidence < conf_threshold
        high_risk_mask = confidence >= self.config["high_risk_threshold"]

        # Map to threat severity (for MTD trigger — Obj 5)
        severities = [THREAT_SEVERITY.get(int(p), "unknown") for p in predictions]

        # Class names
        if self.class_names:
            pred_labels = [self.class_names[p] for p in predictions]
        else:
            pred_labels = [str(p) for p in predictions]

        return {
            "predictions": predictions,
            "pred_labels": pred_labels,
            "probabilities": fused_probs,
            "xgb_probabilities": xgb_probs,
            "lstm_probabilities": lstm_probs,
            "confidence": confidence,
            "uncertain_mask": uncertain_mask,
            "high_risk_mask": high_risk_mask,
            "severities": severities,
            "n_uncertain": uncertain_mask.sum(),
            "n_high_risk": high_risk_mask.sum(),
        }

    def evaluate(self, data: dict) -> dict:
        """Full evaluation on test set with per-model and ensemble metrics."""
        from sklearn.metrics import f1_score, accuracy_score, classification_report

        print("\n" + "="*60)
        print(" RAMS — Objective 3: Hybrid Engine Evaluation")
        print("="*60)

        # ── XGBoost eval ──────────────────────────────────────────
        print("\n[Hybrid] XGBoost Evaluation:")
        xgb_results = self.xgboost.evaluate(
            data["X_test"], data["y_test"], self.class_names
        )

        # ── Bi-LSTM eval ──────────────────────────────────────────
        print("\n[Hybrid] Bi-LSTM Evaluation:")
        lstm_results = self.bilstm.evaluate(
            data["X_test_seq"], data["y_test_seq"], self.class_names
        )

        # ── Hybrid ensemble eval ──────────────────────────────────
        print("\n[Hybrid] Ensemble Evaluation:")
        ensemble_out = self.predict_ensemble(data["X_test"], data["X_test_seq"])
        preds = ensemble_out["predictions"]

        # Align labels (same windowing offset as LSTM)
        n_pred = len(preds)
        y_test_aligned = data["y_test_seq"][:n_pred] if \
            len(data["y_test_seq"]) >= n_pred else data["y_test"][:n_pred]

        acc = accuracy_score(y_test_aligned, preds)
        f1_w = f1_score(y_test_aligned, preds, average="weighted", zero_division=0)
        f1_m = f1_score(y_test_aligned, preds, average="macro", zero_division=0)

        # FIXED
        unique_labels = np.unique(np.concatenate([y_test_aligned, preds]))
        filtered_names = ([self.class_names[i] for i in unique_labels]
            if self.class_names else None)

        report = classification_report(
            y_test_aligned, preds,
            labels=unique_labels,
            target_names=filtered_names,
            zero_division=0,
            output_dict=True
            )

        print(f"\n  {'Model':<20} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}")
        print(f"  {'-'*55}")
        print(f"  {'XGBoost':<20} "
              f"{xgb_results['accuracy']:>10.4f} "
              f"{xgb_results['f1_weighted']:>12.4f} "
              f"{xgb_results['f1_macro']:>10.4f}")
        print(f"  {'Bi-LSTM':<20} "
              f"{'N/A':>10} "
              f"{lstm_results['f1_weighted']:>12.4f} "
              f"{'N/A':>10}")
        print(f"  {'Hybrid Ensemble':<20} "
              f"{acc:>10.4f} "
              f"{f1_w:>12.4f} "
              f"{f1_m:>10.4f}")
        print(f"\n  High-risk alerts:    {ensemble_out['n_high_risk']}")
        print(f"  Uncertain flags:     {ensemble_out['n_uncertain']}")

        results = {
            "xgboost": xgb_results,
            "bilstm": lstm_results,
            "ensemble": {
                "accuracy": acc,
                "f1_weighted": f1_w,
                "f1_macro": f1_m,
                "report": report,
                "predictions": preds,
                "labels": y_test_aligned,
                **ensemble_out,
            },
        }

        # Save results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "xgboost_accuracy": xgb_results["accuracy"],
            "xgboost_f1_weighted": xgb_results["f1_weighted"],
            "bilstm_f1_weighted": lstm_results["f1_weighted"],
            "ensemble_accuracy": acc,
            "ensemble_f1_weighted": f1_w,
            "ensemble_f1_macro": f1_m,
        }
        out_path = os.path.join(self.results_dir, "detection_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Hybrid] Results saved to {out_path}")

        return results

    def generate_alert(self, flow_data: dict) -> dict:
        """
        Generate a structured security alert for a single detected threat.
        This feeds into the MTD response system (Objective 5).

        Args:
            flow_data: single flow prediction result

        Returns:
            Alert dict with threat details for MTD trigger
        """
        severity = flow_data.get("severity", "unknown")
        label = flow_data.get("label", "Unknown")
        confidence = flow_data.get("confidence", 0.0)

        alert = {
            "alert_id": f"RAMS-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "threat_type": label,
            "severity": severity,
            "confidence": float(confidence),
            "source": "RAMS-HybridEngine-Obj3",
            "mtd_action_required": severity in ("critical", "high"),
            "recommended_mtd_action": self._get_mtd_action(label, severity),
            "explanation_requested": True,   # Triggers Obj 4 SHAP explanation
        }
        return alert

    def _get_mtd_action(self, threat_type: str, severity: str) -> str:
        """Map threat type to MTD action (for Objective 5 integration)."""
        actions = {
            "DDoS": "ip_shuffling",
            "Bot": "container_mutation",
            "Infiltration": "pod_recreation",
            "Heartbleed": "port_hopping",
            "DoS Hulk": "ip_shuffling",
            "PortScan": "port_hopping",
        }
        for key, action in actions.items():
            if key in threat_type:
                return action
        if severity == "critical":
            return "pod_recreation"
        elif severity == "high":
            return "container_mutation"
        return "none"
