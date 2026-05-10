"""
data/data_loader.py — Dataset Loading & Preprocessing
RAMS Framework — Objective 3: Hybrid ML Detection Engine

Supports:
  - CIC-IDS2017 (primary recommended dataset)
  - TON_IoT
  - Synthetic data (for demo/testing without downloads)

Reference: BoT-EnsIDS paper — bio-inspired feature selection + preprocessing
Reference: FUSE-Net paper — hybrid ensemble data preparation for ITS
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

try:
    from config import (
        CICIDS2017_LABEL_COL, CICIDS2017_BENIGN_LABEL,
        TONIOT_LABEL_COL, TONIOT_TYPE_COL,
        CORE_FLOW_FEATURES, THREAT_LABELS, EVAL_CONFIG
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        CICIDS2017_LABEL_COL, CICIDS2017_BENIGN_LABEL,
        TONIOT_LABEL_COL, TONIOT_TYPE_COL,
        CORE_FLOW_FEATURES, THREAT_LABELS, EVAL_CONFIG
    )


# ══════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR (no download required for testing)
# ══════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 10000, n_features: int = 60,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic network flow data resembling CIC-IDS2017 structure.
    Useful for testing the full pipeline without downloading datasets.

    Attack types mimic: DDoS, Botnet, DoS, PortScan, Brute Force, BENIGN
    """
    np.random.seed(random_state)
    rng = np.random.RandomState(random_state)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    attack_types = ["BENIGN", "DDoS", "Bot", "DoS Hulk", "PortScan",
                    "FTP-Patator", "SSH-Patator", "Heartbleed",
                    "Web Attack \x96 XSS", "Web Attack \x96 Sql Injection"]

    # Class distribution mimicking CIC-IDS2017 (imbalanced, BENIGN majority)
    class_proportions = [0.50, 0.15, 0.10, 0.08, 0.06,
                         0.04, 0.03, 0.01, 0.02, 0.01]
    labels = rng.choice(attack_types, size=n_samples, p=class_proportions)

    data_rows = []
    for label in labels:
        if label == "BENIGN":
            # Normal traffic: moderate, consistent flows
            row = rng.normal(loc=50, scale=10, size=n_features)
            row = np.abs(row)
        elif label == "DDoS":
            # High packet rate, small sizes
            row = rng.normal(loc=200, scale=30, size=n_features)
            row[:5] = rng.uniform(1000, 5000, 5)   # packet counts spike
            row[10:15] = rng.uniform(0, 10, 5)     # tiny packet sizes
        elif label == "Bot":
            # Periodic, low-volume, encrypted-like
            row = rng.normal(loc=30, scale=5, size=n_features)
            row[20:25] = rng.uniform(500, 1000, 5)  # IAT features
        elif label == "DoS Hulk":
            # High bytes/s
            row = rng.normal(loc=150, scale=40, size=n_features)
            row[6:8] = rng.uniform(1e6, 5e6, 2)    # flow bytes/s
        elif label == "PortScan":
            # Many short flows
            row = rng.normal(loc=20, scale=5, size=n_features)
            row[0:2] = rng.uniform(1, 3, 2)         # few packets/flow
            row[4] = rng.uniform(0, 1)               # duration near 0
        else:
            # Other attacks: varied
            row = rng.normal(loc=70, scale=20, size=n_features)
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=feature_names)
    df["Label"] = labels
    print(f"[DataLoader] Synthetic dataset: {len(df):,} samples, "
          f"{n_features} features, {len(attack_types)} classes")
    print(f"[DataLoader] Class distribution: {dict(Counter(labels))}")
    return df


# ══════════════════════════════════════════════════════════════════
# CIC-IDS2017 LOADER
# ══════════════════════════════════════════════════════════════════

def load_cicids2017(data_path: str) -> pd.DataFrame:
    """
    Load CIC-IDS2017 dataset from directory containing CSV files.

    Download: https://www.unb.ca/cic/datasets/ids-2017.html
    → MachineLearningCSV.zip (contains daily CSV files)

    Files: Monday, Tuesday, Wednesday, Thursday-Morning,
           Thursday-Afternoon, Friday-Morning, Friday-Afternoon,
           Friday-Evening
    """
    data_path = Path(data_path)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_path}.\n"
            f"Download CIC-IDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html\n"
            f"Extract MachineLearningCSV.zip and point --data_path here."
        )

    print(f"[DataLoader] Loading {len(csv_files)} CIC-IDS2017 CSV files...")
    dfs = []
    for f in csv_files:
        print(f"  → {f.name}")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[DataLoader] Total rows: {len(df):,}")

    # Rename label column for uniformity
    df = df.rename(columns={CICIDS2017_LABEL_COL: "Label"})

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    return df


# ══════════════════════════════════════════════════════════════════
# TON_IoT LOADER
# ══════════════════════════════════════════════════════════════════

def load_toniot(data_path: str) -> pd.DataFrame:
    """
    Load TON_IoT dataset.

    Download: https://research.unsw.edu.au/projects/toniot-datasets
    → Network dataset (network_dataset_*.csv files)

    TON_IoT is ideal for smart mobility + IoT context
    43 attack categories including ransomware, scanning, injection, backdoor
    """
    data_path = Path(data_path)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_path}.\n"
            f"Download TON_IoT from: https://research.unsw.edu.au/projects/toniot-datasets"
        )

    print(f"[DataLoader] Loading {len(csv_files)} TON_IoT CSV files...")
    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    # Combine binary label + type into single multi-class label
    if TONIOT_TYPE_COL in df.columns:
        df["Label"] = df[TONIOT_TYPE_COL].str.lower().str.strip()
        df["Label"] = df["Label"].replace({"normal": "BENIGN"})
    elif TONIOT_LABEL_COL in df.columns:
        df["Label"] = df[TONIOT_LABEL_COL].apply(
            lambda x: "BENIGN" if x == 0 else "Attack"
        )

    print(f"[DataLoader] TON_IoT rows: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════
# CORE PREPROCESSOR
# ══════════════════════════════════════════════════════════════════

class RAMSDataPreprocessor:
    """
    Full preprocessing pipeline for Tier 3 (Cloud) detection.

    Steps:
      1. Drop useless columns (IPs, ports, timestamps)
      2. Handle infinities and NaN
      3. Clip extreme outliers
      4. Encode labels
      5. Feature selection (mutual information — from BoT-EnsIDS paper)
      6. Scale features
      7. Handle class imbalance (SMOTE)
      8. Create sequence windows for Bi-LSTM
    """

    def __init__(self, n_features: int = 60, sequence_len: int = 10,
                 use_smote: bool = True, random_state: int = 42):
        self.n_features = n_features
        self.sequence_len = sequence_len
        self.use_smote = use_smote
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        self.classes_ = None

    # ── Step 1: Clean raw dataframe ──────────────────────────────
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[Preprocessor] Cleaning data...")

        # Drop non-feature columns
        drop_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in
            ["ip", "port", "timestamp", "flow id", "src", "dst", "protocol_name"]
        ) and c != "Label"]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Replace inf with NaN, then fill
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))

        # Clip extreme outliers at 99.9th percentile per column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=upper)

        print(f"[Preprocessor] After cleaning: {df.shape}")
        return df

    # ── Step 2: Encode labels ────────────────────────────────────
    def encode_labels(self, df: pd.DataFrame) -> tuple:
        print("[Preprocessor] Encoding labels...")
        labels = df["Label"].astype(str).str.strip()
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        print(f"[Preprocessor] Classes: {list(self.classes_)}")
        X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
        return X, y

    # ── Step 3: Feature selection (Mutual Information) ───────────
    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                         fit: bool = True) -> np.ndarray:
        n_feat = min(self.n_features, X.shape[1])
        if fit:
            print(f"[Preprocessor] Selecting top {n_feat} features "
                  f"(Mutual Information — BoT-EnsIDS method)...")
            self.feature_selector = SelectKBest(mutual_info_classif, k=n_feat)
            # Use a subsample for speed if large dataset
            if len(X) > 50000:
                idx = np.random.choice(len(X), 50000, replace=False)
                self.feature_selector.fit(X.iloc[idx], y[idx])
            else:
                self.feature_selector.fit(X, y)
            mask = self.feature_selector.get_support()
            self.selected_features = X.columns[mask].tolist()
            print(f"[Preprocessor] Selected features: {self.selected_features[:5]}... "
                  f"(+{len(self.selected_features)-5} more)")

        return self.feature_selector.transform(X)

    # ── Step 4: Scale ────────────────────────────────────────────
    def scale(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    # ── Step 5: SMOTE (handle class imbalance) ───────────────────
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> tuple:
        print("[Preprocessor] Applying SMOTE to handle class imbalance...")
        before = dict(Counter(y))
        # Only oversample minority classes with < 500 samples
        min_samples = {cls: max(count, 500)
                       for cls, count in before.items()
                       if count < 500}
        try:
            smote = SMOTE(sampling_strategy=min_samples if min_samples else "auto",
                          random_state=self.random_state, k_neighbors=3)
            X_res, y_res = smote.fit_resample(X, y)
            print(f"[Preprocessor] SMOTE: {sum(before.values()):,} → "
                  f"{len(y_res):,} samples")
        except Exception as e:
            print(f"[Preprocessor] SMOTE skipped ({e}), using original data")
            X_res, y_res = X, y
        return X_res, y_res

    # ── Step 6: Create sequences for Bi-LSTM ────────────────────
    def create_sequences(self, X: np.ndarray, y: np.ndarray, max_sequences: int = 100000) -> tuple:
        """
        Create sliding window sequences for temporal Bi-LSTM analysis.
        Window of `sequence_len` consecutive flows = one sequence sample.
        Inspired by: FUSE-Net hybrid sequential modeling for ITS.
        """
        print(f"[Preprocessor] Creating sequences "
              f"(window={self.sequence_len})...")
        seq_len = self.sequence_len
        total = len(X) - seq_len
        if total > max_sequences:
            indices = np.linspace(0, total - 1, max_sequences, dtype=int)
            print(f"[Preprocessor] Subsampling {max_sequences:,} sequences "
                  f"from {total:,} (memory limit)")
        else:
            indices = np.arange(total)

        X_seq = np.array([X[i: i + seq_len] for i in indices], dtype=np.float32)
        y_seq = np.array([y[i + seq_len - 1] for i in indices], dtype=np.int64)
        print(f"[Preprocessor] Sequences shape: {X_seq.shape}")
        return X_seq, y_seq
        
    
    # ── Full pipeline ────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> dict:
        """
        Full preprocessing pipeline. Returns dict with:
          - X_flat: (n_samples, n_features) — for XGBoost
          - X_seq:  (n_samples, seq_len, n_features) — for Bi-LSTM
          - y_flat, y_seq: corresponding labels
          - X_train_flat, X_val_flat, X_test_flat (+ seq variants)
          - y_train_flat, y_val_flat, y_test_flat (+ seq variants)
        """
        # ADD THIS before clean()
        MAX_SAMPLES = 500000
        if len(df) > MAX_SAMPLES:
            print(f"[Preprocessor] Dataset too large ({len(df):,} rows), "
                  f"sampling {MAX_SAMPLES:,} for memory safety...")
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

        df = self.clean(df)
        X_df, y = self.encode_labels(df)
        X_sel = self.select_features(X_df, y, fit=True)
        X_scaled = self.scale(X_sel, fit=True)

        # Split BEFORE SMOTE to avoid data leakage
        test_size = EVAL_CONFIG["test_size"]
        val_size = EVAL_CONFIG["val_size"]

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_scaled, y, test_size=test_size,
            random_state=EVAL_CONFIG["random_state"], stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=val_size / (1 - test_size),
            random_state=EVAL_CONFIG["random_state"], stratify=y_tmp
        )

        # Apply SMOTE only on training set
        if self.use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)

        # Create sequences for Bi-LSTM
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)

        print(f"\n[Preprocessor] Final splits:")
        print(f"  Train (flat): {X_train.shape}, Train (seq): {X_train_seq.shape}")
        print(f"  Val   (flat): {X_val.shape},   Val (seq):   {X_val_seq.shape}")
        print(f"  Test  (flat): {X_test.shape},  Test (seq):  {X_test_seq.shape}")

        return {
            # Flat (XGBoost)
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            # Sequential (Bi-LSTM)
            "X_train_seq": X_train_seq, "X_val_seq": X_val_seq,
            "X_test_seq": X_test_seq,
            "y_train_seq": y_train_seq, "y_val_seq": y_val_seq,
            "y_test_seq": y_test_seq,
            # Meta
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y)),
            "classes": self.classes_,
            "label_encoder": self.label_encoder,
        }

    def transform(self, df: pd.DataFrame) -> dict:
        """Transform new data using fitted preprocessor (inference)."""
        df = self.clean(df)
        X_df, y = self.encode_labels(df)
        X_sel = self.select_features(X_df, y, fit=False)
        X_scaled = self.scale(X_sel, fit=False)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        return {
            "X_flat": X_scaled, "y_flat": y,
            "X_seq": X_seq, "y_seq": y_seq,
        }


# ══════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════

def load_dataset(dataset_name: str, data_path: str = None,
                 n_samples: int = 10000) -> pd.DataFrame:
    """
    Unified dataset loader.

    Args:
        dataset_name: "cicids2017" | "toniot" | "synthetic"
        data_path: path to CSV files (required for real datasets)
        n_samples: only used for synthetic

    Returns:
        Raw pandas DataFrame with "Label" column
    """
    name = dataset_name.lower().strip()
    if name == "cicids2017":
        return load_cicids2017(data_path)
    elif name in ("toniot", "ton_iot"):
        return load_toniot(data_path)
    elif name == "synthetic":
        return generate_synthetic_data(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose: cicids2017 | toniot | synthetic")
