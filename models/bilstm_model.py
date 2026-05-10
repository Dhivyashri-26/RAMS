"""
models/bilstm_model.py — Bidirectional LSTM for Sequential Threat Detection
RAMS Framework — Objective 3

Architecture inspired by:
  - FUSE-Net: Hybrid ensemble for cloud-based ITS security
  - META: Multi-classified encrypted traffic anomaly detection

The Bi-LSTM processes sequences of network flows (sliding window),
capturing temporal attack patterns like slow-loris, botnet heartbeats,
and DDoS ramp-up that are invisible in single-flow analysis.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════

class FlowSequenceDataset(Dataset):
    """PyTorch Dataset for sliding-window network flow sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, seq_len, n_features) float32
            y: (n_samples,) int64 class labels
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════
# BI-LSTM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class BiLSTMDetector(nn.Module):
    """
    Bidirectional LSTM for network intrusion detection.

    Architecture:
      Input → BiLSTM (2 layers) → Attention → Dropout → FC → Softmax

    Key design choices (from FUSE-Net & META papers):
      - Bidirectional: captures both forward (escalation) and backward
        (de-escalation) patterns in attack sequences
      - Attention mechanism: focuses on the most anomalous flows
        in the window, improving detection of slow/stealthy attacks
      - Multi-class output: 15 threat categories (not just binary)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.3):
        super(BiLSTMDetector, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # ── Bi-LSTM layers ────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ── Attention mechanism ───────────────────────────────────
        # Produces a single context vector by weighting each time step
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # ── Batch normalization ───────────────────────────────────
        self.bn = nn.BatchNorm1d(hidden_size * 2)

        # ── Classifier head ───────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: (batch, num_classes) logits
        """
        # Bi-LSTM: output = (batch, seq_len, hidden*2)
        lstm_out, _ = self.bilstm(x)

        # Attention weights: (batch, seq_len, 1)
        attn_weights = self.attention(lstm_out)

        # Context vector: (batch, hidden*2)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Normalize + classify
        context = self.bn(context)
        context = self.dropout(context)
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

    def get_attention_weights(self, x):
        """Extract attention weights for XAI analysis."""
        lstm_out, _ = self.bilstm(x)
        attn_weights = self.attention(lstm_out)
        return attn_weights.squeeze(-1).detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════

class BiLSTMTrainer:
    """
    Training wrapper for BiLSTMDetector with:
      - Class-weighted loss (handles imbalance)
      - Early stopping
      - Learning rate scheduling
      - Model checkpointing
    """

    def __init__(self, config: dict, n_classes: int, class_weights=None,
                 save_path: str = "results/saved_models/bilstm_best.pt"):
        self.config = config
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Device
        if config.get("device") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.get("device", "cpu"))
        print(f"[BiLSTM] Using device: {self.device}")

        # Model
        self.model = BiLSTMDetector(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=n_classes,
            dropout=config["dropout"]
        ).to(self.device)

        # Loss with class weights (handles imbalance from minority attack classes)
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer + scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=2, factor=0.5
        )

        self.best_val_f1 = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     shuffle: bool) -> DataLoader:
        ds = FlowSequenceDataset(X, y)
        return DataLoader(ds, batch_size=self.config["batch_size"],
                          shuffle=shuffle, num_workers=0, pin_memory=False)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _eval_epoch(self, loader: DataLoader) -> tuple:
        self.model.eval()
        all_preds, all_labels, total_loss = [], [], 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return total_loss / len(loader), f1, np.array(all_preds), np.array(all_labels)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Full training loop with early stopping."""
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        patience = self.config["early_stopping_patience"]
        no_improve = 0

        print(f"\n[BiLSTM] Training: {self.config['epochs']} epochs, "
              f"batch={self.config['batch_size']}")
        print(f"[BiLSTM] Model parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config["epochs"] + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_f1, _, _ = self._eval_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)

            self.scheduler.step(val_f1)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.save_path)
                no_improve = 0
                flag = "✓ saved"
            else:
                no_improve += 1
                flag = f"(no improve {no_improve}/{patience})"

            print(f"  Epoch {epoch:3d}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} {flag}")

            if no_improve >= patience:
                print(f"[BiLSTM] Early stopping at epoch {epoch}")
                break

        # Load best
        self.model.load_state_dict(torch.load(self.save_path,
                                               map_location=self.device))
        print(f"[BiLSTM] Training complete. Best Val F1: {self.best_val_f1:.4f}")
        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 class_names: list = None) -> dict:
        """Evaluate on test set, return metrics + predictions."""
        loader = self._make_loader(X_test, y_test, shuffle=False)
        _, f1, preds, labels = self._eval_epoch(loader)

        unique_labels = np.unique(np.concatenate([labels, preds]))
        filtered_names = ([class_names[i] for i in unique_labels]
                          if class_names else None)

        report = classification_report(
            labels, preds,
            labels=unique_labels,
            target_names=filtered_names,
            zero_division=0,
            output_dict=True
            )
        print(f"\n[BiLSTM] Test F1 (weighted): {f1:.4f}")
        if class_names:
            print(classification_report(labels, preds,
                                        labels=unique_labels,
                                        target_names=filtered_names,
                                        zero_division=0))
        return {"predictions": preds, "labels": labels,
                "f1_weighted": f1, "report": report}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for ensemble fusion."""
        self.model.eval()
        ds = FlowSequenceDataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(ds, batch_size=self.config["batch_size"],
                            shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                prob = torch.softmax(logits, dim=1)
                probs.append(prob.cpu().numpy())
        return np.vstack(probs)

    def load_best(self):
        """Load the best saved checkpoint."""
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        self.model.eval()
        return self


# ══════════════════════════════════════════════════════════════════
# UTILITY: Compute class weights for imbalanced dataset
# ══════════════════════════════════════════════════════════════════

def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse frequency class weights for weighted CrossEntropyLoss."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    # Normalize
    weights = weights / weights.sum() * len(classes)
    return weights.astype(np.float32)
