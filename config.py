"""
config.py — Global Configuration for RAMS Full Framework
Objectives: 2 (Edge IDS), 3 (Hybrid ML), 4 (XAI/SHAP), 6 (SUMO Simulation)
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
CICIDS2017_LABEL_COL = " Label"
CICIDS2017_BENIGN_LABEL = "BENIGN"
TONIOT_LABEL_COL = "label"
TONIOT_TYPE_COL = "type"

THREAT_LABELS = {
    "BENIGN": 0, "DoS Hulk": 1, "PortScan": 2, "DDoS": 3,
    "DoS GoldenEye": 4, "FTP-Patator": 5, "SSH-Patator": 6,
    "DoS slowloris": 7, "DoS Slowhttptest": 8, "Bot": 9,
    "Web Attack \x96 Brute Force": 10, "Web Attack \x96 XSS": 11,
    "Infiltration": 12, "Web Attack \x96 Sql Injection": 13, "Heartbleed": 14,
}

THREAT_SEVERITY = {
    0: "none", 1: "medium", 2: "low", 3: "critical", 4: "medium",
    5: "medium", 6: "medium", 7: "low", 8: "low", 9: "critical",
    10: "high", 11: "high", 12: "critical", 13: "critical", 14: "critical",
}

# ─────────────────────────────────────────────
# OBJECTIVE 2 — EDGE IDS CONFIG
# ─────────────────────────────────────────────
EDGE_CONFIG = {
    # BoT-EnsIDS inspired lightweight models
    "models": ["decision_tree", "random_forest"],
    "dt_params": {
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "rf_params": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    },
    # Edge filtering: flag as suspicious if P(attack) > threshold
    "suspicion_threshold": 0.3,
    # Target: filter ~90% benign locally, forward only suspicious
    "target_filter_rate": 0.90,
    # Max features for edge (memory-constrained device simulation)
    "n_features": 20,
    # Output: binary classification at edge (BENIGN vs SUSPICIOUS)
    "binary_mode": True,
}

# ─────────────────────────────────────────────
# OBJECTIVE 6 — SUMO SIMULATION CONFIG
# ─────────────────────────────────────────────
SUMO_CONFIG = {
    # Simulation area: grid-based smart city intersection
    "grid_size": 5,           # 5x5 road grid
    "simulation_steps": 3600, # 1 hour of traffic
    "vehicles_per_hour": 200,
    "seed": 42,

    # Attack injection settings
    "attack_scenarios": {
        "ddos": {
            "enabled": True,
            "start_step": 600,    # Start at 10 min
            "duration": 300,      # 5 min attack
            "packet_rate": 5000,  # packets/sec during attack
            "target": "RSU_001",  # Roadside Unit targeted
        },
        "botnet": {
            "enabled": True,
            "start_step": 1200,
            "duration": 600,
            "beacon_interval": 30,  # Heartbeat every 30s
            "n_bots": 15,
        },
        "dos": {
            "enabled": True,
            "start_step": 1800,
            "duration": 200,
        },
        "port_scan": {
            "enabled": True,
            "start_step": 2400,
            "duration": 100,
            "port_range": [1, 1024],
        },
    },

    # Output
    "output_pcap": os.path.join(DATA_DIR, "sumo_traffic.csv"),
    "output_dir": os.path.join(RESULTS_DIR, "simulation"),
}

# ─────────────────────────────────────────────
# OBJECTIVE 3 — HYBRID ML ENGINE
# ─────────────────────────────────────────────
BILSTM_CONFIG = {
    "sequence_len": 10,
    "input_size": 60,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 15,
    "dropout": 0.3,
    "batch_size": 256,
    "epochs": 30,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "device": "auto",
}

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
    "tree_method": "hist",
}

HYBRID_CONFIG = {
    "xgboost_weight": 0.55,
    "bilstm_weight": 0.45,
    "confidence_threshold": 0.7,
    "high_risk_threshold": 0.85,
}

# ─────────────────────────────────────────────
# OBJECTIVE 4 — SHAP XAI
# ─────────────────────────────────────────────
SHAP_CONFIG = {
    "explainer_type": "tree",
    "n_background_samples": 100,
    "max_display": 20,
    "output_dir": RESULTS_DIR,
}

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
EVAL_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "cv_folds": 5,
}
