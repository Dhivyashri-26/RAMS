"""
models/xgboost_model.py — XGBoost Classifier for Tabular Threat Detection
RAMS Framework — Objective 3

XGBoost handles tabular flow features exceptionally well, especially:
  - Non-linear feature interactions (packet size × flag combinations)
  - Categorical-like encoded features
  - Missing/noisy sensor data (common in IoT/edge environments)

Reference: BoT-EnsIDS — ensemble model leveraging gradient boosting
Reference: FUSE-Net — tabular component of hybrid framework
"""

import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")


class XGBoostDetector:
    """
    XGBoost-based network intrusion detector for the RAMS Cloud tier.

    Strengths over deep learning for tabular data:
      - Faster training and inference (critical for real-time cloud processing)
      - No need for large datasets to generalize
      - Naturally handles feature importance (feeds into SHAP — Obj 4)
      - Robust to outliers and noise (common in network traffic)
    """

    def __init__(self, config: dict, n_classes: int,
                 save_path: str = "results/saved_models/xgboost_model.joblib"):
        self.config = config
        self.n_classes = n_classes
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Build XGBoost with multi-class objective
        self.early_stopping_rounds = config.get("early_stopping_rounds", 20)

        xgb_params = {
            k: v for k, v in config.items()
            if k not in ("early_stopping_rounds",)
        }

        self.model = XGBClassifier(
            **xgb_params,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
        )
        self.feature_importances_ = None
        self.classes_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train XGBoost with early stopping on validation F1.
        Uses eval_set for built-in XGBoost early stopping.
        """
        print(f"\n[XGBoost] Training with {X_train.shape[0]:,} samples, "
              f"{X_train.shape[1]} features, {self.n_classes} classes...")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = np.unique(y_train)

        # Save
        joblib.dump(self.model, self.save_path)
        print(f"[XGBoost] Model saved to {self.save_path}")
        try:
            print(f"[XGBoost] Best iteration: {self.model.best_iteration}")
        except AttributeError:
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for hybrid ensemble."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 class_names: list = None) -> dict:
        """Full evaluation with multi-class metrics."""
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, preds, average="macro", zero_division=0)

        # ROC-AUC (one-vs-rest for multi-class)
        try:
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            if y_bin.shape[1] > 1:
                roc_auc = roc_auc_score(y_bin, probs,
                                         multi_class="ovr", average="weighted")
            else:
                roc_auc = roc_auc_score(y_test, probs[:, 1])
        except Exception:
            roc_auc = None

        unique_labels = np.unique(np.concatenate([y_test, preds]))
        filtered_names = ([class_names[i] for i in unique_labels]
                          if class_names else None)
        report = classification_report(
            y_test, preds,
            labels=unique_labels,
            target_names=filtered_names,
            zero_division=0,
            output_dict=True
        )

        print(f"\n[XGBoost] Test Results:")
        print(f"  Accuracy:       {acc:.4f}")
        print(f"  F1 (weighted):  {f1_w:.4f}")
        print(f"  F1 (macro):     {f1_m:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:        {roc_auc:.4f}")
        if class_names:
            print(classification_report(y_test, preds,
                                        labels=unique_labels,
                                        target_names=filtered_names,
                                        zero_division=0))

        return {
            "predictions": preds,
            "probabilities": probs,
            "labels": y_test,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "roc_auc": roc_auc,
            "report": report,
            "confusion_matrix": confusion_matrix(y_test, preds),
        }

    def get_top_features(self, feature_names: list = None,
                          top_n: int = 20) -> list:
        """Return top N features by XGBoost importance score."""
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        feat_imp = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        )
        return feat_imp[:top_n]

    def load(self):
        """Load saved model from disk."""
        self.model = joblib.load(self.save_path)
        print(f"[XGBoost] Loaded from {self.save_path}")
        return self
