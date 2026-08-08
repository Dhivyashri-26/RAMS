"""
models/xgboost_model.py — XGBoost Classifier (Objective 3)
All fixes applied: early_stopping_rounds in constructor, classification_report label mismatch
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
    def __init__(self, config, n_classes,
                 save_path="results/saved_models/xgboost_model.joblib"):
        self.config = config
        self.n_classes = n_classes
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.early_stopping_rounds = config.get("early_stopping_rounds", 20)
        xgb_params = {k: v for k, v in config.items()
                      if k != "early_stopping_rounds"}

        self.model = XGBClassifier(
            **xgb_params,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.feature_importances_ = None
        self.classes_ = None

    def train(self, X_train, y_train, X_val, y_val):
        print(f"\n[XGBoost] Training: {X_train.shape[0]:,} samples, "
              f"{X_train.shape[1]} features, {self.n_classes} classes")
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)], verbose=50)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = np.unique(y_train)
        joblib.dump(self.model, self.save_path)
        print(f"[XGBoost] Saved → {self.save_path}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, class_names=None):
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, preds, average="macro", zero_division=0)

        try:
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            roc_auc = (roc_auc_score(y_bin, probs, multi_class="ovr",
                                      average="weighted")
                       if y_bin.shape[1] > 1
                       else roc_auc_score(y_test, probs[:, 1]))
        except Exception:
            roc_auc = None

        # Fix: only report classes present in test + pred
        unique_labels = np.unique(np.concatenate([y_test, preds]))
        filtered_names = ([class_names[i] for i in unique_labels]
                          if class_names else None)

        report = classification_report(
            y_test, preds, labels=unique_labels,
            target_names=filtered_names, zero_division=0, output_dict=True
        )
        print(f"\n[XGBoost] Accuracy: {acc:.4f} | F1-W: {f1_w:.4f} | "
              f"F1-M: {f1_m:.4f}" +
              (f" | ROC-AUC: {roc_auc:.4f}" if roc_auc else ""))
        if class_names:
            print(classification_report(
                y_test, preds, labels=unique_labels,
                target_names=filtered_names, zero_division=0
            ))
        return {
            "predictions": preds, "probabilities": probs, "labels": y_test,
            "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
            "roc_auc": roc_auc, "report": report,
            "confusion_matrix": confusion_matrix(y_test, preds),
        }

    def get_top_features(self, feature_names=None, top_n=20):
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        return sorted(zip(feature_names, importances),
                      key=lambda x: x[1], reverse=True)[:top_n]

    def load(self):
        self.model = joblib.load(self.save_path)
        return self
