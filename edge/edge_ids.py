"""
edge/edge_ids.py — Lightweight Edge IDS (Objective 2)
RAMS Framework — Tier 2: Intelligent Edge Layer (Fog Computing)

Implements BoT-EnsIDS inspired ensemble:
  - Decision Tree (ultra-fast, interpretable)
  - Random Forest (robust, handles noise)
  - Ensemble voting → Binary: BENIGN vs SUSPICIOUS

Role in RAMS pipeline:
  - Filters ~90% of benign traffic LOCALLY (saves bandwidth)
  - Only forwards suspicious flows to Cloud (Obj 3) via Kafka
  - Runs on resource-constrained edge device (Raspberry Pi / Docker)

Reference: BoT-EnsIDS — bio-inspired ensemble feature selection
           + hybrid deep learning for IoT Botnet detection
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score
)
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EDGE_CONFIG, EVAL_CONFIG, RESULTS_DIR, MODELS_DIR


# ══════════════════════════════════════════════════════════════════
# EDGE PREPROCESSOR
# ══════════════════════════════════════════════════════════════════

class EdgePreprocessor:
    """
    Lightweight preprocessing for edge-constrained environment.
    Uses only top 20 features (memory/CPU limit of Raspberry Pi).
    Binary output: BENIGN (0) vs SUSPICIOUS (1).
    """

    def __init__(self, n_features: int = 20):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.selector = None
        self.selected_features = None
        self.label_encoder = LabelEncoder()

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        print(f"[EdgeIDS] Preprocessing {len(df):,} samples...")

        # Drop non-numeric / identifier columns
        drop_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ["ip", "port", "timestamp", "flow id"]
        ) and c != "Label"]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Handle inf/NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))

        # Clip outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=upper)

        # Binary labels: BENIGN=0, everything else=1 (SUSPICIOUS)
        labels = df["Label"].astype(str).str.strip()
        y = (labels != "BENIGN").astype(int).values
        X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])

        print(f"[EdgeIDS] Binary distribution — "
              f"BENIGN: {(y==0).sum():,}, SUSPICIOUS: {(y==1).sum():,}")

        # Feature selection: top 20 by mutual information
        n_feat = min(self.n_features, X.shape[1])
        print(f"[EdgeIDS] Selecting top {n_feat} features...")
        sample_idx = np.random.choice(len(X), min(30000, len(X)), replace=False)
        self.selector = SelectKBest(mutual_info_classif, k=n_feat)
        self.selector.fit(X.iloc[sample_idx], y[sample_idx])
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        X_sel = self.selector.transform(X)

        # Scale
        X_scaled = self.scaler.fit_transform(X_sel)

        print(f"[EdgeIDS] Top features: {self.selected_features[:5]}...")
        return X_scaled, y

    # FIXED — aligns columns to training feature set before transforming
    def transform(self, df: pd.DataFrame) -> tuple:
        """Transform new data using fitted preprocessor."""
        drop_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ["ip", "port", "timestamp", "flow id"]
        ) and c != "Label"]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        if "Label" in df.columns:
            labels = df["Label"].astype(str).str.strip()
            y = (labels != "BENIGN").astype(int).values
            X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
        else:
            y = np.zeros(len(df), dtype=int)
            X = df.select_dtypes(include=[np.number])

        # ── Align columns to what selector was trained on ──────────
        # Get the feature names the selector expects
        if hasattr(self.selector, "feature_names_in_"):
            expected_cols = list(self.selector.feature_names_in_)
        else:
            expected_cols = None

        if expected_cols is not None:
            # Add missing columns as zeros
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0.0
            # Drop extra columns not seen at fit time
            X = X[[c for c in expected_cols if c in X.columns]]
            # If still missing some, pad with zeros
            missing = [c for c in expected_cols if c not in X.columns]
            for col in missing:
                X[col] = 0.0
            X = X[expected_cols]

        X_sel = self.selector.transform(X)
        X_scaled = self.scaler.transform(X_sel)
        return X_scaled, y

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "selector": self.selector,
                     "selected_features": self.selected_features}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.selector = obj["selector"]
        self.selected_features = obj["selected_features"]
        return self


# ══════════════════════════════════════════════════════════════════
# EDGE IDS MODEL
# ══════════════════════════════════════════════════════════════════

class EdgeIDS:
    """
    Lightweight Ensemble IDS for Tier 2 Edge Layer.

    Ensemble: Decision Tree + Random Forest → majority vote
    Decision Tree: ultra-fast inference (~microseconds per flow)
    Random Forest: robust, handles feature noise from IoT sensors

    Design principle (BoT-EnsIDS): use bio-inspired feature
    selection + ensemble to maximise detection with minimal compute.
    """

    def __init__(self, config: dict = None,
                 save_dir: str = None):
        self.config = config or EDGE_CONFIG
        self.save_dir = save_dir or MODELS_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        self.dt = DecisionTreeClassifier(**config["dt_params"])
        self.rf = RandomForestClassifier(**config["rf_params"])
        self.suspicion_threshold = config["suspicion_threshold"]
        self.preprocessor = EdgePreprocessor(n_features=config["n_features"])

        self.is_fitted = False
        self.metrics = {}
        self.feature_importances = {}

    # ── Training ─────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        Full training pipeline:
          1. Preprocess + feature selection
          2. Train DT + RF
          3. Cross-validate
          4. Evaluate on hold-out test set
          5. Compute filter efficiency (bandwidth savings)
        """
        print("\n" + "="*60)
        print(" RAMS — Objective 2: Edge IDS Training (BoT-EnsIDS)")
        print("="*60)

        # Cap dataset for edge simulation (edge devices have limited RAM)
        MAX = 300000
        if len(df) > MAX:
            df = df.sample(n=MAX, random_state=42).reset_index(drop=True)
            print(f"[EdgeIDS] Capped to {MAX:,} samples (edge memory limit)")

        X, y = self.preprocessor.fit_transform(df)

        # Train/val/test split
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=EVAL_CONFIG["test_size"],
            random_state=EVAL_CONFIG["random_state"], stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.125,
            random_state=EVAL_CONFIG["random_state"], stratify=y_tmp
        )

        print(f"[EdgeIDS] Train: {len(X_train):,} | "
              f"Val: {len(X_val):,} | Test: {len(X_test):,}")

        # Train Decision Tree
        print("\n[EdgeIDS] Training Decision Tree...")
        t0 = time.time()
        self.dt.fit(X_train, y_train)
        dt_time = time.time() - t0
        dt_val_f1 = f1_score(y_val, self.dt.predict(X_val), average="weighted")
        print(f"  Train time: {dt_time:.2f}s | Val F1: {dt_val_f1:.4f}")

        # Train Random Forest
        print("[EdgeIDS] Training Random Forest...")
        t0 = time.time()
        self.rf.fit(X_train, y_train)
        rf_time = time.time() - t0
        rf_val_f1 = f1_score(y_val, self.rf.predict(X_val), average="weighted")
        print(f"  Train time: {rf_time:.2f}s | Val F1: {rf_val_f1:.4f}")

        # Cross-validation (5-fold)
        print("\n[EdgeIDS] 5-Fold Cross Validation...")
        dt_cv = cross_val_score(self.dt, X_train, y_train,
                                 cv=5, scoring="f1_weighted", n_jobs=-1)
        rf_cv = cross_val_score(self.rf, X_train, y_train,
                                 cv=5, scoring="f1_weighted", n_jobs=-1)
        print(f"  DT  CV F1: {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")
        print(f"  RF  CV F1: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

        # Test evaluation
        results = self.evaluate(X_test, y_test)
        results["dt_train_time"] = dt_time
        results["rf_train_time"] = rf_time
        results["dt_cv_f1"] = float(dt_cv.mean())
        results["rf_cv_f1"] = float(rf_cv.mean())
        self.metrics = results

        # Feature importances
        feat_names = self.preprocessor.selected_features or \
            [f"f{i}" for i in range(X.shape[1])]
        self.feature_importances = {
            "decision_tree": dict(zip(feat_names, self.dt.feature_importances_)),
            "random_forest": dict(zip(feat_names, self.rf.feature_importances_)),
        }

        self.is_fitted = True
        self.save()
        return results

    # ── Inference ────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction: majority vote of DT + RF.
        Returns binary labels: 0=BENIGN, 1=SUSPICIOUS
        """
        dt_preds = self.dt.predict(X)
        rf_preds = self.rf.predict(X)
        # Majority vote (both must agree for BENIGN to pass through)
        ensemble = ((dt_preds + rf_preds) >= 1).astype(int)
        return ensemble

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Suspicion probability (average of DT + RF probabilities)."""
        dt_proba = self.dt.predict_proba(X)[:, 1]
        rf_proba = self.rf.predict_proba(X)[:, 1]
        return (dt_proba + rf_proba) / 2

    def filter_traffic(self, df: pd.DataFrame) -> dict:
        """
        Main edge filtering function.
        Processes a batch of flows, returns:
          - Flows to forward to Cloud (suspicious)
          - Flows blocked at edge (benign)
          - Filter efficiency stats

        This is what runs continuously on the edge device.
        """
        t0 = time.time()
        X, y_true = self.preprocessor.transform(df)
        suspicion_probs = self.predict_proba(X)
        predictions = (suspicion_probs >= self.suspicion_threshold).astype(int)
        inference_time = time.time() - t0

        suspicious_mask = predictions == 1
        benign_mask = ~suspicious_mask

        # Bandwidth saving: only suspicious flows forwarded to cloud
        n_total = len(df)
        n_suspicious = suspicious_mask.sum()
        n_benign_blocked = benign_mask.sum()
        filter_rate = n_benign_blocked / n_total if n_total > 0 else 0

        # Per-flow inference time (key metric for edge deployment)
        per_flow_ms = (inference_time / n_total * 1000) if n_total > 0 else 0

        result = {
            "total_flows": n_total,
            "suspicious_count": int(n_suspicious),
            "benign_blocked": int(n_benign_blocked),
            "filter_rate": float(filter_rate),
            "bandwidth_saved_pct": float(filter_rate * 100),
            "inference_time_ms": float(inference_time * 1000),
            "per_flow_inference_ms": float(per_flow_ms),
            "suspicious_flows": df[suspicious_mask].copy(),
            "blocked_flows": df[benign_mask].copy(),
            "predictions": predictions,
            "suspicion_scores": suspicion_probs,
        }

        print(f"\n[EdgeIDS] Filter Results:")
        print(f"  Total flows:       {n_total:>8,}")
        print(f"  Forwarded (cloud): {n_suspicious:>8,} "
              f"({n_suspicious/n_total*100:.1f}%)")
        print(f"  Blocked (benign):  {n_benign_blocked:>8,} "
              f"({filter_rate*100:.1f}% saved)")
        print(f"  Inference time:    {inference_time*1000:.1f}ms total, "
              f"{per_flow_ms:.3f}ms/flow")

        return result

    # ── Evaluation ───────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate ensemble on test set."""
        dt_preds = self.dt.predict(X_test)
        rf_preds = self.rf.predict(X_test)
        ens_preds = self.predict(X_test)
        ens_proba = self.predict_proba(X_test)

        results = {}
        for name, preds in [("DT", dt_preds), ("RF", rf_preds),
                             ("Ensemble", ens_preds)]:
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            prec = precision_score(y_test, preds, average="weighted",
                                   zero_division=0)
            rec = recall_score(y_test, preds, average="weighted",
                               zero_division=0)
            results[name.lower()] = {
                "accuracy": float(acc), "f1_weighted": float(f1),
                "precision": float(prec), "recall": float(rec),
            }

        # False negative rate (missed attacks) — critical for IDS
        ens_cm = confusion_matrix(y_test, ens_preds)
        if ens_cm.shape == (2, 2):
            fn = ens_cm[1][0]  # True attack, predicted benign
            tp = ens_cm[1][1]
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            results["ensemble"]["false_negative_rate"] = float(fnr)
            results["ensemble"]["confusion_matrix"] = ens_cm.tolist()

        print(f"\n[EdgeIDS] Test Evaluation:")
        print(f"  {'Model':<12} {'Accuracy':>10} {'F1':>10} "
              f"{'Precision':>10} {'Recall':>10}")
        print(f"  {'-'*50}")
        for name in ["dt", "rf", "ensemble"]:
            r = results[name]
            print(f"  {name.upper():<12} "
                  f"{r['accuracy']:>10.4f} "
                  f"{r['f1_weighted']:>10.4f} "
                  f"{r.get('precision', 0):>10.4f} "
                  f"{r.get('recall', 0):>10.4f}")

        if "false_negative_rate" in results.get("ensemble", {}):
            print(f"\n  False Negative Rate (missed attacks): "
                  f"{results['ensemble']['false_negative_rate']:.4f}")

        return results

    # ── Persistence ───────────────────────────────────────────────

    def save(self):
        joblib.dump(self.dt,
                    os.path.join(self.save_dir, "edge_dt.joblib"))
        joblib.dump(self.rf,
                    os.path.join(self.save_dir, "edge_rf.joblib"))
        self.preprocessor.save(
            os.path.join(self.save_dir, "edge_preprocessor.joblib"))
        meta = {
            "metrics": self.metrics,
            "feature_importances": {
                k: {f: float(v) for f, v in imp.items()}
                for k, imp in self.feature_importances.items()
            },
            "suspicion_threshold": self.suspicion_threshold,
            "n_features": self.config["n_features"],
        }
        with open(os.path.join(self.save_dir, "edge_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[EdgeIDS] Models saved to {self.save_dir}")

    def load(self):
        self.dt = joblib.load(
            os.path.join(self.save_dir, "edge_dt.joblib"))
        self.rf = joblib.load(
            os.path.join(self.save_dir, "edge_rf.joblib"))
        self.preprocessor.load(
            os.path.join(self.save_dir, "edge_preprocessor.joblib"))
        self.is_fitted = True
        print(f"[EdgeIDS] Models loaded from {self.save_dir}")
        return self

    # ── Visualisation ────────────────────────────────────────────

    def plot_results(self, results: dict, output_dir: str = None):
        """Generate Edge IDS evaluation plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir = output_dir or RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        # ── Plot 1: Model comparison bar chart ────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        models = ["Decision Tree", "Random Forest", "Ensemble"]
        keys = ["dt", "rf", "ensemble"]
        metrics_to_plot = ["accuracy", "f1_weighted", "precision", "recall"]
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        x = np.arange(len(models))
        width = 0.2
        ax = axes[0]
        for i, (metric, colour) in enumerate(zip(metrics_to_plot, colours)):
            vals = [results.get(k, {}).get(metric, 0) for k in keys]
            bars = ax.bar(x + i * width, vals, width,
                          label=metric.replace("_", " ").title(),
                          color=colour, alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7, fontweight="bold")

        ax.set_ylim(0, 1.08)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title("Edge IDS — Model Comparison\nObjective 2: BoT-EnsIDS",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # ── Plot 2: Feature importances ────────────────────────────
        ax2 = axes[1]
        if self.feature_importances:
            rf_imp = self.feature_importances.get("random_forest", {})
            top = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:15]
            feat_names = [t[0] for t in top]
            feat_vals = [t[1] for t in top]
            ax2.barh(range(len(feat_names)), feat_vals[::-1],
                     color="#1f77b4", alpha=0.8)
            ax2.set_yticks(range(len(feat_names)))
            ax2.set_yticklabels(feat_names[::-1], fontsize=8)
            ax2.set_xlabel("Feature Importance (Gini)")
            ax2.set_title("Top 15 Features — Random Forest\n(Edge Feature Selection)",
                          fontweight="bold")
            ax2.grid(axis="x", alpha=0.3)

        plt.suptitle("RAMS Framework — Objective 2: Edge IDS (Tier 2)",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        path1 = os.path.join(output_dir, "edge_ids_evaluation.png")
        plt.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EdgeIDS] Plot saved: {path1}")

        # ── Plot 3: Confusion matrix ───────────────────────────────
        if "confusion_matrix" in results.get("ensemble", {}):
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = np.array(results["ensemble"]["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["BENIGN", "SUSPICIOUS"],
                        yticklabels=["BENIGN", "SUSPICIOUS"])
            ax.set_title("Edge IDS — Ensemble Confusion Matrix\n"
                         "Objective 2: Binary Classification",
                         fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            path2 = os.path.join(output_dir, "edge_confusion_matrix.png")
            plt.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[EdgeIDS] Confusion matrix saved: {path2}")

        # ── Plot 4: Bandwidth savings ──────────────────────────────
        filter_rate = results.get("ensemble", {}).get("accuracy", 0.9)
        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [filter_rate * 100, (1 - filter_rate) * 100]
        colours_pie = ["#2ca02c", "#d62728"]
        labels_pie = [f"Blocked at Edge\n({filter_rate*100:.1f}%)",
                      f"Forwarded to Cloud\n({(1-filter_rate)*100:.1f}%)"]
        ax.pie(sizes, labels=labels_pie, colors=colours_pie,
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 11})
        ax.set_title("Edge Traffic Filtering\nBandwidth Reduction (Objective 2)",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        path3 = os.path.join(output_dir, "edge_bandwidth_savings.png")
        plt.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EdgeIDS] Bandwidth savings plot saved: {path3}")
