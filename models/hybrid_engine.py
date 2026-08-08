"""
models/hybrid_engine.py — Hybrid ML Detection Engine (Objective 3)
All classification_report fixes applied.
"""

import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HYBRID_CONFIG, THREAT_SEVERITY, RESULTS_DIR


class HybridDetectionEngine:
    def __init__(self, bilstm_trainer, xgboost_detector,
                 config=None, class_names=None):
        self.bilstm = bilstm_trainer
        self.xgboost = xgboost_detector
        self.config = config or HYBRID_CONFIG
        self.class_names = class_names
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def _align(self, xgb, lstm):
        n = min(len(xgb), len(lstm))
        return xgb[len(xgb)-n:], lstm[len(lstm)-n:]

    def predict_ensemble(self, X_flat, X_seq):
        xgb_p = self.xgboost.predict_proba(X_flat)
        lstm_p = self.bilstm.predict_proba(X_seq)
        if xgb_p.shape[1] != lstm_p.shape[1]:
            n = min(xgb_p.shape[1], lstm_p.shape[1])
            xgb_p, lstm_p = xgb_p[:, :n], lstm_p[:, :n]
        xgb_p, lstm_p = self._align(xgb_p, lstm_p)
        w = self.config
        fused = w["xgboost_weight"] * xgb_p + w["bilstm_weight"] * lstm_p
        preds = np.argmax(fused, axis=1)
        conf = np.max(fused, axis=1)
        severities = [THREAT_SEVERITY.get(int(p), "unknown") for p in preds]
        pred_labels = ([self.class_names[p] for p in preds]
                       if self.class_names else [str(p) for p in preds])
        return {
            "predictions": preds, "pred_labels": pred_labels,
            "probabilities": fused, "xgb_probabilities": xgb_p,
            "lstm_probabilities": lstm_p, "confidence": conf,
            "uncertain_mask": conf < w["confidence_threshold"],
            "high_risk_mask": conf >= w["high_risk_threshold"],
            "severities": severities,
            "n_uncertain": (conf < w["confidence_threshold"]).sum(),
            "n_high_risk": (conf >= w["high_risk_threshold"]).sum(),
        }

    def evaluate(self, data):
        print("\n" + "="*60)
        print(" RAMS — Objective 3: Hybrid Engine Evaluation")
        print("="*60)

        xgb_results = self.xgboost.evaluate(
            data["X_test"], data["y_test"], self.class_names)

        lstm_results = self.bilstm.evaluate(
            data["X_test_seq"], data["y_test_seq"], self.class_names)

        ens = self.predict_ensemble(data["X_test"], data["X_test_seq"])
        preds = ens["predictions"]
        n = len(preds)
        y_aligned = (data["y_test_seq"][:n] if len(data["y_test_seq"]) >= n
                     else data["y_test"][:n])

        acc = accuracy_score(y_aligned, preds)
        f1_w = f1_score(y_aligned, preds, average="weighted", zero_division=0)
        f1_m = f1_score(y_aligned, preds, average="macro", zero_division=0)

        unique_labels = np.unique(np.concatenate([y_aligned, preds]))
        filtered_names = ([self.class_names[i] for i in unique_labels]
                          if self.class_names else None)
        report = classification_report(
            y_aligned, preds, labels=unique_labels,
            target_names=filtered_names, zero_division=0, output_dict=True
        )

        print(f"\n  {'Model':<20} {'Accuracy':>10} {'F1-W':>10} {'F1-M':>10}")
        print(f"  {'-'*52}")
        print(f"  {'XGBoost':<20} {xgb_results['accuracy']:>10.4f} "
              f"{xgb_results['f1_weighted']:>10.4f} "
              f"{xgb_results['f1_macro']:>10.4f}")
        print(f"  {'Bi-LSTM':<20} {'—':>10} "
              f"{lstm_results['f1_weighted']:>10.4f} {'—':>10}")
        print(f"  {'Hybrid Ensemble':<20} {acc:>10.4f} "
              f"{f1_w:>10.4f} {f1_m:>10.4f}")
        print(f"\n  High-risk alerts: {ens['n_high_risk']} | "
              f"Uncertain: {ens['n_uncertain']}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "xgboost_accuracy": xgb_results["accuracy"],
            "xgboost_f1_weighted": xgb_results["f1_weighted"],
            "bilstm_f1_weighted": lstm_results["f1_weighted"],
            "ensemble_accuracy": acc,
            "ensemble_f1_weighted": f1_w,
            "ensemble_f1_macro": f1_m,
        }
        with open(os.path.join(RESULTS_DIR, "detection_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "xgboost": xgb_results,
            "bilstm": lstm_results,
            "ensemble": {
                "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
                "report": report, "predictions": preds, "labels": y_aligned,
                **ens,
            },
        }

    def generate_alert(self, flow_data):
        severity = flow_data.get("severity", "unknown")
        label = flow_data.get("label", "Unknown")
        confidence = flow_data.get("confidence", 0.0)
        return {
            "alert_id": f"RAMS-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "threat_type": label,
            "severity": severity,
            "confidence": float(confidence),
            "source": "RAMS-HybridEngine-Obj3",
            "mtd_action_required": severity in ("critical", "high"),
            "recommended_mtd_action": self._get_mtd_action(label, severity),
            "explanation_requested": True,
        }

    def _get_mtd_action(self, threat_type, severity):
        actions = {
            "DDoS": "ip_shuffling", "Bot": "container_mutation",
            "Infiltration": "pod_recreation", "Heartbleed": "port_hopping",
            "DoS Hulk": "ip_shuffling", "PortScan": "port_hopping",
        }
        for key, action in actions.items():
            if key in threat_type:
                return action
        return "pod_recreation" if severity == "critical" else \
               "container_mutation" if severity == "high" else "none"
