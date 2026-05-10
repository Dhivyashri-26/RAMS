"""
config.py — Global Configuration for RAMS Objectives 3 & 4
RAMS Framework: Hybrid ML Detection Engine + Explainable AI
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────

# CIC-IDS2017: 78 network flow features + Label column
# Download from: https://www.unb.ca/cic/datasets/ids-2017.html
CICIDS2017_LABEL_COL = " Label"
CICIDS2017_BENIGN_LABEL = "BENIGN"

# TON_IoT: IoT network dataset
# Download from: https://research.unsw.edu.au/projects/toniot-datasets
TONIOT_LABEL_COL = "label"
TONIOT_TYPE_COL = "type"

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Core flow-based features used across all tiers (Obj 3 — Tier 3 Cloud)
# These align with what the Edge tier (Obj 2) forwards for suspicious traffic
CORE_FLOW_FEATURES = [
    # Packet-level
    " Total Fwd Packets", " Total Backward Packets",
    "Total Length of Fwd Packets", " Total Length of Bwd Packets",
    " Fwd Packet Length Max", " Fwd Packet Length Min",
    " Fwd Packet Length Mean", " Fwd Packet Length Std",
    "Bwd Packet Length Max", " Bwd Packet Length Min",
    " Bwd Packet Length Mean", " Bwd Packet Length Std",
    # Flow-level
    " Flow Bytes/s", " Flow Packets/s",
    " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
    # Timing
    " Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std",
    " Fwd IAT Max", " Fwd IAT Min",
    " Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std",
    # Flags (encrypted traffic metadata — from META paper)
    " Fwd PSH Flags", " Bwd PSH Flags",
    " Fwd URG Flags", " Bwd URG Flags",
    " FIN Flag Count", " SYN Flag Count", " RST Flag Count",
    " PSH Flag Count", " ACK Flag Count", " URG Flag Count",
    " CWE Flag Count", " ECE Flag Count",
    # Window / Header
    " Fwd Header Length", " Bwd Header Length",
    " Fwd Packets/s", " Bwd Packets/s",
    " Min Packet Length", " Max Packet Length",
    " Packet Length Mean", " Packet Length Std", " Packet Length Variance",
    # Ratios
    " Down/Up Ratio", " Average Packet Size",
    " Avg Fwd Segment Size", " Avg Bwd Segment Size",
    " Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", " Bwd Avg Bulk Rate",
    # Subflows
    "Subflow Fwd Packets", " Subflow Fwd Bytes",
    "Subflow Bwd Packets", " Subflow Bwd Bytes",
    # Init window sizes (encrypted traffic signal)
    "Init_Win_bytes_forward", " Init_Win_bytes_backward",
    " act_data_pkt_fwd", " min_seg_size_forward",
    # Misc
    "Active Mean", " Active Std", " Active Max", " Active Min",
    "Idle Mean", " Idle Std", " Idle Max", " Idle Min",
]

# Threat label mapping (multi-class)
THREAT_LABELS = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
    "Bot": 9,
    "Web Attack \x96 Brute Force": 10,
    "Web Attack \x96 XSS": 11,
    "Infiltration": 12,
    "Web Attack \x96 Sql Injection": 13,
    "Heartbleed": 14,
}

# Severity mapping for MTD trigger (feeds into Obj 5)
THREAT_SEVERITY = {
    0: "none",       # BENIGN
    1: "medium",     # DoS Hulk
    2: "low",        # PortScan
    3: "critical",   # DDoS → triggers MTD IP shuffling
    4: "medium",     # DoS GoldenEye
    5: "medium",     # FTP-Patator
    6: "medium",     # SSH-Patator
    7: "low",        # DoS slowloris
    8: "low",        # DoS Slowhttptest
    9: "critical",   # Botnet → triggers MTD container mutation
    10: "high",      # Web Brute Force
    11: "high",      # XSS
    12: "critical",  # Infiltration → triggers MTD pod recreation
    13: "critical",  # SQL Injection
    14: "critical",  # Heartbleed
}

# ─────────────────────────────────────────────
# Bi-LSTM MODEL CONFIG (Objective 3)
# ─────────────────────────────────────────────
BILSTM_CONFIG = {
    "sequence_len": 10,        # Window of 10 consecutive flows (temporal context)
    "input_size": 60,          # Feature dimensionality (after preprocessing)
    "hidden_size": 128,        # LSTM hidden units
    "num_layers": 2,           # Stacked Bi-LSTM layers
    "num_classes": 15,         # Multi-class threat categories
    "dropout": 0.3,
    "batch_size": 256,
    "epochs": 30,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "device": "auto",          # "auto" = GPU if available, else CPU
}

# ─────────────────────────────────────────────
# XGBoost MODEL CONFIG (Objective 3)
# ─────────────────────────────────────────────
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 20,
    "tree_method": "hist",     # Fast histogram method
}

# ─────────────────────────────────────────────
# HYBRID ENSEMBLE CONFIG (Objective 3)
# ─────────────────────────────────────────────
HYBRID_CONFIG = {
    # Weighted voting: XGBoost stronger for tabular, Bi-LSTM for sequences
    "xgboost_weight": 0.55,
    "bilstm_weight": 0.45,
    "confidence_threshold": 0.7,   # Below this → flag as uncertain
    "high_risk_threshold": 0.85,   # Above this → immediate MTD trigger
}

# ─────────────────────────────────────────────
# SHAP CONFIG (Objective 4)
# ─────────────────────────────────────────────
SHAP_CONFIG = {
    "explainer_type": "tree",      # TreeExplainer for XGBoost (fast)
    "n_background_samples": 100,   # For DeepExplainer (Bi-LSTM)
    "max_display": 20,             # Top features to display
    "output_dir": RESULTS_DIR,
}

# ─────────────────────────────────────────────
# EVALUATION CONFIG
# ─────────────────────────────────────────────
EVAL_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "cv_folds": 5,
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
}
