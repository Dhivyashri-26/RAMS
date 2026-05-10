"""
explainability/shap_explainer.py — SHAP-Based Explainable AI Module
RAMS Framework — Objective 4

Provides transparent, interpretable explanations for every threat detection
decision using SHAP (SHapley Additive exPlanations).

Reference:
  "Enhancing Healthcare Data Privacy in Cloud IoT Networks Using Anomaly
   Detection and Optimization with Explainable AI (ExAI)"
  → Adapted for cybersecurity threat detection in smart mobility networks

SHAP integration points:
  1. TreeExplainer → XGBoost (fast, exact Shapley values)
  2. DeepExplainer → Bi-LSTM (approximate, sampled background)
  3. Global feature importance (dataset-level)
  4. Local explanation (per-alert, per-flow)
  5. Threat report generation (human-readable output for SOC analysts)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

import shap


class RAMSExplainer:
    """
    SHAP-based Explainable AI for RAMS Hybrid Detection Engine.

    Fulfils Objective 4: "Integrate Explainable AI (XAI) for
    transparent decision-making."

    Key outputs:
      - Global SHAP summary plots (what features matter most overall)
      - Per-class SHAP beeswarm plots (how each attack type is detected)
      - Local waterfall/force plots (why THIS specific alert was raised)
      - Human-readable threat explanation reports
      - JSON explanation export (for API/dashboard integration)
    """

    def __init__(self, xgboost_model, bilstm_trainer=None,
                 feature_names: list = None, class_names: list = None,
                 config: dict = None, output_dir: str = "results"):
        """
        Args:
            xgboost_model: fitted XGBoostDetector
            bilstm_trainer: fitted BiLSTMTrainer (optional, for deep SHAP)
            feature_names: list of feature column names
            class_names: list of threat class names
            config: SHAP_CONFIG from config.py
            output_dir: directory to save SHAP plots and reports
        """
        self.xgboost_model = xgboost_model
        self.bilstm_trainer = bilstm_trainer
        self.feature_names = feature_names or []
        self.class_names = class_names or []
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.xgb_explainer = None
        self.deep_explainer = None
        self.shap_values_xgb = None

        print("[XAI] RAMS Explainer initialised (Objective 4)")

    # ══════════════════════════════════════════════════════════════
    # SETUP EXPLAINERS
    # ══════════════════════════════════════════════════════════════

    def setup_tree_explainer(self, X_background: np.ndarray = None):
        """
        Initialise SHAP TreeExplainer for XGBoost.

        TreeExplainer computes exact Shapley values for tree models.
        No background data needed (uses tree structure directly).
        This is the primary explainer for RAMS (fast, exact).
        """
        print("[XAI] Setting up TreeExplainer for XGBoost...")
        self.xgb_explainer = shap.TreeExplainer(
            self.xgboost_model.model,
            feature_perturbation="tree_path_dependent",
            model_output="raw"
        )
        print("[XAI] TreeExplainer ready.")

    def setup_deep_explainer(self, X_background: np.ndarray):
        """
        Initialise SHAP DeepExplainer for Bi-LSTM.

        Uses a small background sample to estimate baseline attribution.
        DeepExplainer uses the DeepLIFT algorithm adapted for PyTorch.

        Args:
            X_background: (n_background, seq_len, n_features)
                         Representative sample of BENIGN traffic
        """
        if self.bilstm_trainer is None:
            print("[XAI] No Bi-LSTM trainer provided, skipping DeepExplainer.")
            return

        import torch
        print(f"[XAI] Setting up DeepExplainer for Bi-LSTM "
              f"(background={len(X_background)} samples)...")
        model = self.bilstm_trainer.model
        model.eval()
        device = self.bilstm_trainer.device
        bg_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)

        self.deep_explainer = shap.DeepExplainer(model, bg_tensor)
        print("[XAI] DeepExplainer ready.")

    # ══════════════════════════════════════════════════════════════
    # COMPUTE SHAP VALUES
    # ══════════════════════════════════════════════════════════════

    def compute_xgb_shap(self, X: np.ndarray,
                          max_samples: int = 5000) -> np.ndarray:
        """
        Compute SHAP values for XGBoost predictions.

        For multi-class: returns array of shape (n_samples, n_features, n_classes)
        For binary:      returns array of shape (n_samples, n_features)

        Subsample for large datasets to keep computation tractable.
        """
        if self.xgb_explainer is None:
            self.setup_tree_explainer()

        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
            print(f"[XAI] Subsampled {max_samples} rows for SHAP computation")
        else:
            X_sample = X

        print(f"[XAI] Computing XGBoost SHAP values for {len(X_sample)} samples...")
        shap_values = self.xgb_explainer.shap_values(X_sample)
        self.shap_values_xgb = shap_values
        self.X_sample_xgb = X_sample
        print(f"[XAI] SHAP values computed. "
              f"Shape: {np.array(shap_values).shape}")
        return shap_values

    def compute_lstm_shap(self, X_seq: np.ndarray,
                           max_samples: int = 500) -> np.ndarray:
        """
        Compute SHAP values for Bi-LSTM using DeepExplainer.

        Returns per-time-step, per-feature attributions.
        Shape: (n_samples, seq_len, n_features)
        """
        if self.deep_explainer is None:
            print("[XAI] DeepExplainer not set up — run setup_deep_explainer() first")
            return None

        import torch
        if len(X_seq) > max_samples:
            idx = np.random.choice(len(X_seq), max_samples, replace=False)
            X_seq = X_seq[idx]

        print(f"[XAI] Computing Bi-LSTM SHAP values for {len(X_seq)} sequences...")
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(
            self.bilstm_trainer.device
        )
        shap_values = self.deep_explainer.shap_values(X_tensor)
        print("[XAI] LSTM SHAP values computed.")
        return shap_values

    # ══════════════════════════════════════════════════════════════
    # GLOBAL VISUALISATIONS
    # ══════════════════════════════════════════════════════════════

    def plot_global_summary(self, shap_values=None, X=None,
                             class_idx: int = None,
                             title: str = "RAMS — Global Feature Importance (SHAP)",
                             save_name: str = "shap_global_summary.png"):
        """
        Global SHAP summary beeswarm plot.
        Shows which features drive predictions across the entire test set.

        For multi-class: shows SHAP for a specific class (class_idx)
        or averaged across all classes if class_idx is None.
        """
        if shap_values is None:
            shap_values = self.shap_values_xgb
        if X is None:
            X = self.X_sample_xgb

        feature_names = (self.feature_names if self.feature_names
                         else [f"f{i}" for i in range(X.shape[1])])

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)

        # Handle multi-class SHAP (list of arrays, one per class)
        if isinstance(shap_values, list):
            if class_idx is not None:
                sv = shap_values[class_idx]
                class_label = (self.class_names[class_idx]
                               if class_idx < len(self.class_names)
                               else f"Class {class_idx}")
                title = f"{title}\n→ {class_label}"
            else:
                # Mean absolute SHAP across all classes
                sv = np.mean([np.abs(s) for s in shap_values], axis=0)
                title += "\n(Mean |SHAP| across all threat classes)"
        else:
            sv = shap_values

        shap.summary_plot(sv, X, feature_names=feature_names,
                          max_display=self.config.get("max_display", 20),
                          show=False, plot_type="dot")
        plt.title(title, fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XAI] Global summary plot saved: {save_path}")
        return save_path

    def plot_feature_importance_bar(self, shap_values=None,
                                     save_name: str = "shap_feature_importance.png"):
        """
        Bar plot: mean absolute SHAP value per feature.
        Provides a clean, non-technical summary for SOC analysts.
        """
        if shap_values is None:
            shap_values = self.shap_values_xgb

        feature_names = (self.feature_names if self.feature_names
                         else [f"f{i}" for i in range(
                             np.array(shap_values).shape[-1]
                         )])

        # Compute mean |SHAP| across samples (and classes if multi-class)
        if isinstance(shap_values, list):
            mean_abs = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        mean_abs = np.asarray(mean_abs)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1)

        top_n = self.config.get("max_display", 20)
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]
        top_feat = [feature_names[i] if i < len(feature_names)
                    else f"f{i}" for i in top_idx]
        top_vals = mean_abs[top_idx]

        # Colour-code by importance tier
        colours = ["#d62728" if v > top_vals[0] * 0.7 else
                   "#ff7f0e" if v > top_vals[0] * 0.4 else
                   "#1f77b4" for v in top_vals]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(top_n), top_vals[::-1], color=colours[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feat[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title("RAMS — Top Features for Threat Detection (SHAP)\n"
                     "Objective 4: Explainable AI Feature Importance",
                     fontsize=12, fontweight="bold")

        # Legend
        high = mpatches.Patch(color="#d62728", label="High importance (>70%)")
        med = mpatches.Patch(color="#ff7f0e", label="Medium importance (40-70%)")
        low = mpatches.Patch(color="#1f77b4", label="Lower importance (<40%)")
        ax.legend(handles=[high, med, low], loc="lower right", fontsize=9)

        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XAI] Feature importance bar plot saved: {save_path}")
        return save_path

    def plot_per_class_importance(self, shap_values: list = None,
                                   save_name: str = "shap_per_class.png"):
        """
        Heatmap: SHAP importance per feature per threat class.
        Reveals which features are diagnostic for each attack type.
        """
        if shap_values is None:
            shap_values = self.shap_values_xgb
        if not isinstance(shap_values, list):
            print("[XAI] Per-class plot requires multi-class SHAP values (list).")
            return None

        import seaborn as sns

        n_classes = len(shap_values)
        n_top = min(15, shap_values[0].shape[1])
        feature_names = (self.feature_names if self.feature_names
                         else [f"f{i}" for i in range(shap_values[0].shape[1])])

        # Mean |SHAP| per feature for each class
        class_importances = np.array(
            [np.abs(sv).mean(axis=0) for sv in shap_values]
        )

        # Select top features by overall importance
        overall = class_importances.mean(axis=0)
        top_idx = np.argsort(overall)[-n_top:][::-1]
        top_feat = [feature_names[i] if i < len(feature_names)
                    else f"f{i}" for i in top_idx]

        heatmap_data = pd.DataFrame(
            class_importances[:, top_idx],
            index=self.class_names[:n_classes] if self.class_names
                  else [f"Class {i}" for i in range(n_classes)],
            columns=top_feat
        )

        fig, ax = plt.subplots(figsize=(14, max(6, n_classes * 0.5 + 2)))
        sns.heatmap(heatmap_data, ax=ax, cmap="YlOrRd",
                    annot=False, linewidths=0.3, linecolor="gray",
                    cbar_kws={"label": "Mean |SHAP Value|"})
        ax.set_title("RAMS — SHAP Feature Importance per Threat Class\n"
                     "Objective 4: Class-Specific Explanations",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Network Flow Features", fontsize=10)
        ax.set_ylabel("Threat Category", fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XAI] Per-class SHAP heatmap saved: {save_path}")
        return save_path

    # ══════════════════════════════════════════════════════════════
    # LOCAL EXPLANATIONS (PER ALERT)
    # ══════════════════════════════════════════════════════════════

    def explain_single_alert(self, x: np.ndarray, prediction: int,
                              confidence: float,
                              alert_id: str = "ALERT-001") -> dict:
        """
        Generate a local SHAP explanation for a single detected threat.

        Returns:
          - Waterfall plot (visual explanation for SOC dashboard)
          - Top contributing features (positive and negative)
          - Human-readable explanation text
          - JSON export for API integration

        Args:
            x: (n_features,) — single flow feature vector
            prediction: predicted class index
            confidence: ensemble confidence score
            alert_id: unique alert identifier
        """
        if self.xgb_explainer is None:
            self.setup_tree_explainer()

        # Compute SHAP for this sample
        x_2d = x.reshape(1, -1)
        shap_vals = self.xgb_explainer.shap_values(x_2d)

        # For multi-class, get SHAP values for the predicted class
        if isinstance(shap_vals, list):
            pred_shap = shap_vals[prediction][0]
        else:
            pred_shap = shap_vals[0]

        feature_names = (self.feature_names if self.feature_names
                         else [f"f{i}" for i in range(len(pred_shap))])

        # Top positive contributors (pushing toward this threat class)
        top_pos_idx = np.argsort(pred_shap)[::-1][:10]
        top_neg_idx = np.argsort(pred_shap)[:10]

        top_positive = [
            {"feature": feature_names[i] if i < len(feature_names) else f"f{i}",
             "shap_value": float(pred_shap[i]),
             "feature_value": float(x[i])}
            for i in top_pos_idx if pred_shap[i] > 0
        ]
        top_negative = [
            {"feature": feature_names[i] if i < len(feature_names) else f"f{i}",
             "shap_value": float(pred_shap[i]),
             "feature_value": float(x[i])}
            for i in top_neg_idx if pred_shap[i] < 0
        ]

        threat_name = (self.class_names[prediction]
                       if prediction < len(self.class_names)
                       else f"Class {prediction}")

        # Generate waterfall plot
        plot_path = self._plot_waterfall(
            pred_shap, x, feature_names, threat_name,
            confidence, alert_id
        )

        # Human-readable explanation
        explanation_text = self._generate_explanation_text(
            threat_name, confidence, top_positive, top_negative
        )

        result = {
            "alert_id": alert_id,
            "threat_class": threat_name,
            "confidence": float(confidence),
            "shap_values": pred_shap.tolist(),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "explanation_text": explanation_text,
            "plot_path": plot_path,
        }

        # Save JSON explanation
        json_path = os.path.join(self.output_dir,
                                  f"explanation_{alert_id}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[XAI] Alert explanation saved: {json_path}")

        return result

    def _plot_waterfall(self, shap_vals: np.ndarray, x: np.ndarray,
                         feature_names: list, threat_name: str,
                         confidence: float, alert_id: str) -> str:
        """Generate waterfall plot for a single alert explanation."""
        top_n = 15
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
        sorted_shap = shap_vals[sorted_idx]
        sorted_feat = [feature_names[i] if i < len(feature_names)
                       else f"f{i}" for i in sorted_idx]
        sorted_vals = [x[i] for i in sorted_idx]

        colours = ["#d62728" if v > 0 else "#1f77b4" for v in sorted_shap]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(top_n), sorted_shap[::-1], color=colours[::-1])
        ax.set_yticks(range(top_n))
        labels = [f"{f}\n(val={v:.2f})" for f, v in
                  zip(sorted_feat[::-1], sorted_vals[::-1])]
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=10)
        ax.set_title(
            f"RAMS — Alert Explanation [{alert_id}]\n"
            f"Detected: {threat_name}  |  Confidence: {confidence:.1%}",
            fontsize=12, fontweight="bold"
        )

        # Annotation
        pos_patch = mpatches.Patch(color="#d62728",
                                    label="Pushes toward threat (↑ risk)")
        neg_patch = mpatches.Patch(color="#1f77b4",
                                    label="Pushes toward benign (↓ risk)")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        fname = f"waterfall_{alert_id}.png"
        save_path = os.path.join(self.output_dir, fname)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    def _generate_explanation_text(self, threat_name: str, confidence: float,
                                    top_positive: list,
                                    top_negative: list) -> str:
        """
        Generate human-readable threat explanation for SOC analysts.
        Implements the "interpretable outputs for detected threats"
        requirement from Tier 3 architecture.
        """
        lines = [
            f"THREAT DETECTED: {threat_name}",
            f"Confidence: {confidence:.1%}",
            f"",
            f"PRIMARY INDICATORS (features driving this detection):",
        ]

        for i, feat in enumerate(top_positive[:5], 1):
            direction = "elevated" if feat["feature_value"] > 0 else "abnormal"
            lines.append(
                f"  {i}. {feat['feature']}: value={feat['feature_value']:.3f} "
                f"— {direction} ({feat['shap_value']:+.4f} SHAP impact)"
            )

        if top_negative:
            lines += ["", "MITIGATING FACTORS (features suggesting lower risk):"]
            for i, feat in enumerate(top_negative[:3], 1):
                lines.append(
                    f"  {i}. {feat['feature']}: value={feat['feature_value']:.3f} "
                    f"({feat['shap_value']:+.4f} SHAP impact)"
                )

        lines += [
            "",
            "INTERPRETATION:",
            f"  The hybrid ML engine (Bi-LSTM + XGBoost) flagged this flow as "
            f"'{threat_name}' based on the above feature combination. "
            f"The top indicators suggest anomalous traffic patterns consistent "
            f"with known {threat_name} attack signatures.",
            "",
            "ACTION: Review MTD response recommendation in alert JSON.",
        ]
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # FULL XAI PIPELINE
    # ══════════════════════════════════════════════════════════════

    def run_full_xai_pipeline(self, X_test: np.ndarray,
                               y_test: np.ndarray,
                               predictions: np.ndarray,
                               confidences: np.ndarray,
                               X_seq_background: np.ndarray = None):
        """
        Run complete Objective 4 XAI pipeline:
          1. Setup TreeExplainer for XGBoost
          2. Compute global SHAP values
          3. Generate global summary + feature importance plots
          4. Generate per-class heatmap
          5. Explain top high-confidence alerts
          6. (Optional) DeepExplainer for Bi-LSTM

        Args:
            X_test: test features (flat, for XGBoost)
            y_test: true labels
            predictions: ensemble predictions
            confidences: ensemble confidence scores
            X_seq_background: background sequences for LSTM SHAP (optional)
        """
        print("\n" + "="*60)
        print(" RAMS — Objective 4: Explainable AI Pipeline")
        print("="*60)

        # Step 1: Setup
        self.setup_tree_explainer()

        # Step 2: Compute global SHAP
        shap_values = self.compute_xgb_shap(X_test)

        # Step 3: Global plots
        print("\n[XAI] Generating global plots...")
        self.plot_global_summary(shap_values, self.X_sample_xgb,
                                  save_name="shap_global_summary.png")
        self.plot_feature_importance_bar(shap_values,
                                          save_name="shap_feature_importance.png")

        # Step 4: Per-class heatmap
        if isinstance(shap_values, list):
            self.plot_per_class_importance(shap_values,
                                            save_name="shap_per_class.png")

        # Step 5: Local explanations for high-confidence detections
        print("\n[XAI] Generating local explanations for top alerts...")
        alert_explanations = []

        # Find high-confidence true positives (non-BENIGN predictions)
        if len(self.X_sample_xgb) > 0:
            n_explain = min(5, len(self.X_sample_xgb))
            high_conf_idx = np.argsort(confidences[:len(self.X_sample_xgb)])[::-1]

            for rank, idx in enumerate(high_conf_idx[:n_explain]):
                x_sample = self.X_sample_xgb[idx]
                pred_class = int(predictions[min(idx, len(predictions)-1)])
                conf = float(confidences[min(idx, len(confidences)-1)])

                if pred_class == 0:
                    continue   # Skip BENIGN explanations (focus on threats)

                explanation = self.explain_single_alert(
                    x_sample, pred_class, conf,
                    alert_id=f"ALERT-{rank+1:03d}"
                )
                alert_explanations.append(explanation)
                print(f"\n  Alert {rank+1}: {explanation['threat_class']} "
                      f"(conf={conf:.1%})")
                print(f"  Top feature: "
                      f"{explanation['top_positive_features'][0]['feature'] if explanation['top_positive_features'] else 'N/A'}")

        # Step 6: Optional LSTM SHAP
        if X_seq_background is not None and self.bilstm_trainer is not None:
            print("\n[XAI] Computing Bi-LSTM SHAP (DeepExplainer)...")
            try:
                n_bg = min(
                    self.config.get("n_background_samples", 100),
                    len(X_seq_background)
                )
                bg = X_seq_background[:n_bg]
                self.setup_deep_explainer(bg)
                lstm_shap = self.compute_lstm_shap(X_seq_background, max_samples=200)
                if lstm_shap is not None:
                    print("[XAI] LSTM SHAP computed (temporal feature attribution).")
            except Exception as e:
                print(f"[XAI] LSTM SHAP failed (non-critical): {e}")

        # Summary report
        self._save_xai_summary(alert_explanations)

        print(f"\n[XAI] Objective 4 complete. All outputs saved to: {self.output_dir}")
        return alert_explanations

    def _save_xai_summary(self, explanations: list):
        """Save a human-readable XAI summary report."""
        report_path = os.path.join(self.output_dir, "xai_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("RAMS FRAMEWORK — XAI THREAT EXPLANATION REPORT\n")
            f.write("Objective 4: Explainable AI for Transparent Decision-Making\n")
            f.write("=" * 70 + "\n\n")

            for exp in explanations:
                f.write(f"{'─'*60}\n")
                f.write(f"Alert ID: {exp['alert_id']}\n")
                f.write(exp["explanation_text"])
                f.write("\n\n")

        print(f"[XAI] XAI report saved: {report_path}")
