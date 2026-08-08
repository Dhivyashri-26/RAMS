"""
explainability/shap_explainer.py — SHAP XAI Module (Objective 4)
All version-compatibility fixes applied.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import shap

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


def _safe_feat_name(feat_list, i):
    """Safely get feature name by index."""
    try:
        if feat_list is not None and len(feat_list) > 0:
            if 0 <= int(i) < len(feat_list):
                return str(feat_list[int(i)])
    except Exception:
        pass
    return f"f{i}"


def _mean_abs_shap(shap_values):
    """
    Compute mean |SHAP| per feature, handling both:
      - list of arrays (multi-class XGBoost): list of (n_samples, n_features)
      - single 2D array (binary): (n_samples, n_features)
      - single 3D array: (n_samples, n_features, n_classes)
    Returns: 1D array of shape (n_features,)
    """
    if isinstance(shap_values, list):
        # Multi-class: list of (n_samples, n_features)
        return np.mean([np.abs(np.array(sv)).mean(axis=0)
                        for sv in shap_values], axis=0)
    sv = np.array(shap_values)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes)
        return np.abs(sv).mean(axis=(0, 2))
    elif sv.ndim == 2:
        return np.abs(sv).mean(axis=0)
    else:
        return np.abs(sv)


def _get_class_shap(shap_values, class_idx):
    """Extract SHAP values for a specific class."""
    if isinstance(shap_values, list):
        return np.array(shap_values[class_idx])
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[:, :, class_idx]
    return sv


class RAMSExplainer:
    def __init__(self, xgboost_model, bilstm_trainer=None,
                 feature_names=None, class_names=None,
                 config=None, output_dir=None):
        self.xgboost_model = xgboost_model
        self.bilstm_trainer = bilstm_trainer
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.class_names = list(class_names) if class_names is not None else []
        self.config = config or {}
        self.output_dir = output_dir or RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.xgb_explainer = None
        self.shap_values_xgb = None
        self.X_sample_xgb = None
        print("[XAI] RAMS Explainer initialised (Objective 4)")

    def setup_tree_explainer(self):
        print("[XAI] Setting up TreeExplainer for XGBoost...")
        self.xgb_explainer = shap.TreeExplainer(
            self.xgboost_model.model,
            feature_perturbation="tree_path_dependent",
            model_output="raw"
        )

    def compute_xgb_shap(self, X, max_samples=5000):
        if self.xgb_explainer is None:
            self.setup_tree_explainer()
        X = np.array(X)
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
            print(f"[XAI] Subsampled {max_samples} for SHAP")
        print(f"[XAI] Computing SHAP values for {len(X)} samples...")
        self.shap_values_xgb = self.xgb_explainer.shap_values(X)
        self.X_sample_xgb = X
        print(f"[XAI] SHAP done. Type: {type(self.shap_values_xgb)}")
        return self.shap_values_xgb

    def plot_global_summary(self, shap_values=None, X=None,
                             save_name="shap_global_summary.png"):
        try:
            if shap_values is None:
                shap_values = self.shap_values_xgb
            if X is None:
                X = self.X_sample_xgb
            X = np.array(X)
            n_feat = X.shape[1]
            feat = (self.feature_names[:n_feat]
                    if self.feature_names else
                    [f"f{i}" for i in range(n_feat)])

            # For multi-class use mean abs; for binary use directly
            if isinstance(shap_values, list):
                sv = np.mean([np.abs(np.array(s)) for s in shap_values], axis=0)
            else:
                sv = np.array(shap_values)
                if sv.ndim == 3:
                    sv = np.abs(sv).mean(axis=2)

            fig, ax = plt.subplots(figsize=(12, 8))
            plt.sca(ax)
            shap.summary_plot(sv, X, feature_names=feat,
                              max_display=self.config.get("max_display", 20),
                              show=False)
            plt.title("RAMS — Global Feature Importance (SHAP)\nObjective 4: XAI",
                      fontweight="bold")
            plt.tight_layout()
            path = os.path.join(self.output_dir, save_name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[XAI] Global summary: {path}")
            return path
        except Exception as e:
            print(f"[XAI] Global summary plot skipped: {e}")
            plt.close("all")
            return None

    def plot_feature_importance_bar(self, shap_values=None,
                                     save_name="shap_feature_importance.png"):
        try:
            if shap_values is None:
                shap_values = self.shap_values_xgb

            mean_abs = _mean_abs_shap(shap_values)
            mean_abs = np.array(mean_abs, dtype=float).flatten()
            n_feat = len(mean_abs)

            feat = (self.feature_names[:n_feat]
                    if self.feature_names else
                    [f"f{i}" for i in range(n_feat)])

            top_n = min(self.config.get("max_display", 20), n_feat)
            top_idx = np.argsort(mean_abs)[-top_n:][::-1]
            top_feat = [_safe_feat_name(feat, i) for i in top_idx]
            top_vals = mean_abs[top_idx]

            max_val = top_vals[0] if top_vals[0] > 0 else 1.0
            colours = ["#d62728" if v > max_val * 0.7 else
                       "#ff7f0e" if v > max_val * 0.4 else
                       "#1f77b4" for v in top_vals]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(range(top_n), top_vals[::-1], color=colours[::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_feat[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("RAMS — Feature Importance (SHAP)\nObjective 4: XAI",
                         fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(self.output_dir, save_name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[XAI] Feature importance bar: {path}")
            return path
        except Exception as e:
            print(f"[XAI] Feature importance bar skipped: {e}")
            plt.close("all")
            return None

    def plot_per_class_importance(self, shap_values=None,
                                   save_name="shap_per_class.png"):
        try:
            if shap_values is None:
                shap_values = self.shap_values_xgb

            # Build per-class importance matrix
            if isinstance(shap_values, list):
                n_classes = len(shap_values)
                class_imp = np.array(
                    [np.abs(np.array(sv)).mean(axis=0) for sv in shap_values]
                )
            else:
                sv = np.array(shap_values)
                if sv.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    n_classes = sv.shape[2]
                    class_imp = np.abs(sv).mean(axis=0).T  # (n_classes, n_features)
                else:
                    print("[XAI] Per-class plot requires multi-class SHAP.")
                    return None

            n_top = min(15, class_imp.shape[1])
            n_feat = class_imp.shape[1]
            feat = (self.feature_names[:n_feat]
                    if self.feature_names else
                    [f"f{i}" for i in range(n_feat)])

            overall = class_imp.mean(axis=0)
            top_idx = np.argsort(overall)[-n_top:][::-1]
            top_feat = [_safe_feat_name(feat, i) for i in top_idx]

            row_labels = (self.class_names[:n_classes]
                          if self.class_names else
                          [f"Class {i}" for i in range(n_classes)])

            hm = pd.DataFrame(
                class_imp[:, top_idx],
                index=row_labels,
                columns=top_feat
            )

            try:
                import seaborn as sns
            except ImportError:
                print("[XAI] seaborn not available, skipping per-class heatmap")
                return None

            fig, ax = plt.subplots(
                figsize=(14, max(6, n_classes * 0.5 + 2))
            )
            sns.heatmap(hm, ax=ax, cmap="YlOrRd", linewidths=0.3,
                        cbar_kws={"label": "Mean |SHAP|"})
            ax.set_title("RAMS — SHAP Importance per Threat Class\nObjective 4",
                         fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            path = os.path.join(self.output_dir, save_name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[XAI] Per-class heatmap: {path}")
            return path
        except Exception as e:
            print(f"[XAI] Per-class heatmap skipped: {e}")
            plt.close("all")
            return None

    def explain_single_alert(self, x, prediction, confidence,
                              alert_id="ALERT-001"):
        try:
            if self.xgb_explainer is None:
                self.setup_tree_explainer()

            x = np.array(x).flatten()
            shap_vals = self.xgb_explainer.shap_values(x.reshape(1, -1))

            # Extract for predicted class
            if isinstance(shap_vals, list):
                pred_idx = min(int(prediction), len(shap_vals) - 1)
                pred_shap = np.array(shap_vals[pred_idx]).flatten()
            else:
                sv = np.array(shap_vals)
                if sv.ndim == 3:
                    pred_idx = min(int(prediction), sv.shape[2] - 1)
                    pred_shap = sv[0, :, pred_idx]
                else:
                    pred_shap = sv.flatten()

            n = len(pred_shap)
            feat = (self.feature_names[:n]
                    if self.feature_names else
                    [f"f{i}" for i in range(n)])

            top_pos_idx = np.argsort(pred_shap)[::-1][:10]
            top_neg_idx = np.argsort(pred_shap)[:10]

            top_pos = [{"feature": _safe_feat_name(feat, i),
                        "shap_value": float(pred_shap[i]),
                        "feature_value": float(x[i]) if i < len(x) else 0.0}
                       for i in top_pos_idx if pred_shap[i] > 0]
            top_neg = [{"feature": _safe_feat_name(feat, i),
                        "shap_value": float(pred_shap[i]),
                        "feature_value": float(x[i]) if i < len(x) else 0.0}
                       for i in top_neg_idx if pred_shap[i] < 0]

            threat = (_safe_feat_name(self.class_names, prediction)
                      if self.class_names else f"Class {prediction}")

            plot_path = self._plot_waterfall(
                pred_shap, x, feat, threat, confidence, alert_id
            )

            result = {
                "alert_id": alert_id,
                "threat_class": threat,
                "confidence": float(confidence),
                "top_positive_features": top_pos,
                "top_negative_features": top_neg,
                "plot_path": plot_path,
            }
            json_path = os.path.join(
                self.output_dir, f"explanation_{alert_id}.json"
            )
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as e:
            print(f"[XAI] Alert explanation skipped: {e}")
            return {"alert_id": alert_id, "error": str(e),
                    "top_positive_features": [], "top_negative_features": []}

    def _plot_waterfall(self, shap_vals, x, feat, threat, conf, alert_id):
        try:
            shap_vals = np.array(shap_vals).flatten()
            x = np.array(x).flatten()
            top_n = min(15, len(shap_vals))
            idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
            sv = shap_vals[idx]
            fn = [_safe_feat_name(feat, i) for i in idx]
            fv = [float(x[i]) if i < len(x) else 0.0 for i in idx]
            colours = ["#d62728" if v > 0 else "#1f77b4" for v in sv]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(range(top_n), sv[::-1], color=colours[::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(
                [f"{f}\n(val={v:.2f})" for f, v in zip(fn[::-1], fv[::-1])],
                fontsize=8
            )
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP Value")
            ax.set_title(f"Alert [{alert_id}] — {threat} | Conf: {conf:.1%}",
                         fontweight="bold")
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"waterfall_{alert_id}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            return path
        except Exception as e:
            print(f"[XAI] Waterfall plot skipped: {e}")
            plt.close("all")
            return None

    def run_full_xai_pipeline(self, X_test, y_test, predictions,
                               confidences, X_seq_background=None):
        print("\n" + "="*60)
        print(" RAMS — Objective 4: Explainable AI Pipeline")
        print("="*60)

        # Step 1: Setup + compute SHAP
        self.setup_tree_explainer()
        shap_values = self.compute_xgb_shap(X_test)

        # Step 2: Global plots (each wrapped — won't crash the pipeline)
        print("\n[XAI] Generating global plots...")
        self.plot_global_summary(shap_values, self.X_sample_xgb)
        self.plot_feature_importance_bar(shap_values)
        self.plot_per_class_importance(shap_values)

        # Step 3: Local alert explanations
        print("\n[XAI] Generating local alert explanations...")
        explanations = []
        confidences = np.array(confidences)
        predictions = np.array(predictions)

        # Only explain non-BENIGN high-confidence detections
        non_benign = np.where(predictions != 0)[0]
        if len(non_benign) > 0:
            sorted_by_conf = non_benign[
                np.argsort(confidences[non_benign])[::-1]
            ]
            for rank, idx in enumerate(sorted_by_conf[:5]):
                if idx >= len(self.X_sample_xgb):
                    continue
                pred = int(predictions[idx])
                conf = float(confidences[idx])
                exp = self.explain_single_alert(
                    self.X_sample_xgb[idx], pred, conf,
                    alert_id=f"ALERT-{rank+1:03d}"
                )
                explanations.append(exp)
                threat = exp.get("threat_class", f"Class {pred}")
                top_f = (exp["top_positive_features"][0]["feature"]
                         if exp.get("top_positive_features") else "N/A")
                print(f"  Alert {rank+1}: {threat} "
                      f"(conf={conf:.1%}) | Top: {top_f}")
        else:
            print("[XAI] No non-BENIGN predictions found for local explanation.")

        # Step 4: Save text report
        report_path = os.path.join(self.output_dir, "xai_report.txt")
        with open(report_path, "w") as f:
            f.write("RAMS XAI THREAT EXPLANATION REPORT\nObjective 4\n\n")
            for exp in explanations:
                f.write(f"Alert: {exp.get('alert_id')} — "
                        f"{exp.get('threat_class','?')}\n")
                f.write(f"Confidence: {exp.get('confidence',0):.1%}\n")
                top = exp.get("top_positive_features", [])
                if top:
                    f.write(f"Top indicator: {top[0]['feature']}\n")
                f.write("\n")
        print(f"[XAI] Report: {report_path}")
        print(f"[XAI] Objective 4 complete. Outputs: {self.output_dir}")
        return explanations
