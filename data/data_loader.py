"""
data/data_loader.py — Dataset Loading & Preprocessing
RAMS Framework — Objective 3 (with memory fixes applied)
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVAL_CONFIG


def generate_synthetic_data(n_samples=10000, n_features=60, random_state=42):
    np.random.seed(random_state)
    rng = np.random.RandomState(random_state)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    attack_types = ["BENIGN", "DDoS", "Bot", "DoS Hulk", "PortScan",
                    "FTP-Patator", "SSH-Patator", "Heartbleed",
                    "Web Attack \x96 XSS", "Web Attack \x96 Sql Injection"]
    class_proportions = [0.50, 0.15, 0.10, 0.08, 0.06,
                         0.04, 0.03, 0.01, 0.02, 0.01]
    labels = rng.choice(attack_types, size=n_samples, p=class_proportions)
    data_rows = []
    for label in labels:
        if label == "BENIGN":
            row = np.abs(rng.normal(50, 10, n_features))
        elif label == "DDoS":
            row = rng.normal(200, 30, n_features)
            row[:5] = rng.uniform(1000, 5000, 5)
            row[10:15] = rng.uniform(0, 10, 5)
        elif label == "Bot":
            row = rng.normal(30, 5, n_features)
            row[20:25] = rng.uniform(500, 1000, 5)
        elif label == "DoS Hulk":
            row = rng.normal(150, 40, n_features)
            row[6:8] = rng.uniform(1e6, 5e6, 2)
        elif label == "PortScan":
            row = rng.normal(20, 5, n_features)
            row[0:2] = rng.uniform(1, 3, 2)
        else:
            row = rng.normal(70, 20, n_features)
        data_rows.append(row)
    df = pd.DataFrame(data_rows, columns=feature_names)
    df["Label"] = labels
    print(f"[DataLoader] Synthetic: {len(df):,} samples, {len(attack_types)} classes")
    return df


def load_cicids2017(data_path):
    data_path = Path(data_path)
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_path}")
    print(f"[DataLoader] Loading {len(csv_files)} CIC-IDS2017 files...")
    dfs = []
    for f in csv_files:
        print(f"  → {f.name}")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={" Label": "Label"})
    df.columns = df.columns.str.strip()
    print(f"[DataLoader] Total rows: {len(df):,}")
    return df


def load_toniot(data_path):
    data_path = Path(data_path)
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_path}")
    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    if "type" in df.columns:
        df["Label"] = df["type"].str.lower().str.strip().replace({"normal": "BENIGN"})
    elif "label" in df.columns:
        df["Label"] = df["label"].apply(lambda x: "BENIGN" if x == 0 else "Attack")
    return df


def load_dataset(dataset_name, data_path=None, n_samples=10000):
    name = dataset_name.lower().strip()
    if name == "cicids2017":
        return load_cicids2017(data_path)
    elif name in ("toniot", "ton_iot"):
        return load_toniot(data_path)
    elif name == "synthetic":
        return generate_synthetic_data(n_samples=n_samples)
    elif name == "sumo":
        # Load from SUMO simulation output
        from config import SUMO_CONFIG
        path = SUMO_CONFIG["output_pcap"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SUMO output not found at {path}. Run simulation first."
            )
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class RAMSDataPreprocessor:
    def __init__(self, n_features=60, sequence_len=10,
                 use_smote=True, random_state=42):
        self.n_features = n_features
        self.sequence_len = sequence_len
        self.use_smote = use_smote
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        self.classes_ = None

    def clean(self, df):
        drop_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in
            ["ip", "port", "timestamp", "flow id", "src", "dst",
             "protocol_name"]
        ) and c != "Label"]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=upper)
        return df

    def encode_labels(self, df):
        labels = df["Label"].astype(str).str.strip()
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
        return X, y

    def select_features(self, X, y, fit=True):
        n_feat = min(self.n_features, X.shape[1])
        if fit:
            self.feature_selector = SelectKBest(mutual_info_classif, k=n_feat)
            idx = (np.random.choice(len(X), 50000, replace=False)
                   if len(X) > 50000 else np.arange(len(X)))
            self.feature_selector.fit(X.iloc[idx], y[idx])
            mask = self.feature_selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        return self.feature_selector.transform(X)

    def scale(self, X, fit=True):
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def apply_smote(self, X, y):
        before = dict(Counter(y))
        min_samples = {cls: max(count, 500)
                       for cls, count in before.items() if count < 500}
        try:
            smote = SMOTE(
                sampling_strategy=min_samples if min_samples else "auto",
                random_state=self.random_state, k_neighbors=3
            )
            X_res, y_res = smote.fit_resample(X, y)
            print(f"[Preprocessor] SMOTE: {sum(before.values()):,} → {len(y_res):,}")
        except Exception as e:
            print(f"[Preprocessor] SMOTE skipped ({e})")
            X_res, y_res = X, y
        return X_res, y_res

    def create_sequences(self, X, y, max_sequences=100000):
        """Memory-safe sequence creation with subsampling."""
        print(f"[Preprocessor] Creating sequences (window={self.sequence_len})...")
        seq_len = self.sequence_len
        total = len(X) - seq_len
        if total > max_sequences:
            indices = np.linspace(0, total - 1, max_sequences, dtype=int)
            print(f"[Preprocessor] Subsampled {max_sequences:,} from {total:,}")
        else:
            indices = np.arange(total)
        X_seq = np.array([X[i: i + seq_len] for i in indices], dtype=np.float32)
        y_seq = np.array([y[i + seq_len - 1] for i in indices], dtype=np.int64)
        print(f"[Preprocessor] Sequences shape: {X_seq.shape}")
        return X_seq, y_seq

    def fit_transform(self, df):
        # Memory cap
        MAX_SAMPLES = 500000
        if len(df) > MAX_SAMPLES:
            print(f"[Preprocessor] Capping {len(df):,} → {MAX_SAMPLES:,} samples")
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

        df = self.clean(df)
        X_df, y = self.encode_labels(df)
        X_sel = self.select_features(X_df, y, fit=True)
        X_scaled = self.scale(X_sel, fit=True)

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_scaled, y, test_size=EVAL_CONFIG["test_size"],
            random_state=EVAL_CONFIG["random_state"], stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=EVAL_CONFIG["val_size"] / (1 - EVAL_CONFIG["test_size"]),
            random_state=EVAL_CONFIG["random_state"], stratify=y_tmp
        )

        if self.use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)

        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)

        print(f"\n[Preprocessor] Splits — "
              f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "X_train_seq": X_train_seq, "X_val_seq": X_val_seq,
            "X_test_seq": X_test_seq,
            "y_train_seq": y_train_seq, "y_val_seq": y_val_seq,
            "y_test_seq": y_test_seq,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y)),
            "classes": self.classes_,
            "label_encoder": self.label_encoder,
        }

    def transform(self, df):
        df = self.clean(df)
        X_df, y = self.encode_labels(df)
        X_sel = self.select_features(X_df, y, fit=False)
        X_scaled = self.scale(X_sel, fit=False)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        return {"X_flat": X_scaled, "y_flat": y,
                "X_seq": X_seq, "y_seq": y_seq}
