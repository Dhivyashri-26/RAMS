"""
models/bilstm_model.py — Bidirectional LSTM (Objective 3)
RAMS Framework — with all fixes applied
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


class FlowSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class BiLSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, 1), nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn = self.attention(lstm_out)
        context = torch.sum(attn * lstm_out, dim=1)
        context = self.bn(context)
        context = self.dropout(context)
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out)

    def get_attention_weights(self, x):
        lstm_out, _ = self.bilstm(x)
        return self.attention(lstm_out).squeeze(-1).detach().cpu().numpy()


class BiLSTMTrainer:
    def __init__(self, config, n_classes, class_weights=None,
                 save_path="results/saved_models/bilstm_best.pt"):
        self.config = config
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and
            config.get("device") == "auto" else "cpu"
        )
        print(f"[BiLSTM] Device: {self.device}")
        self.model = BiLSTMDetector(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=n_classes,
            dropout=config["dropout"]
        ).to(self.device)
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=2, factor=0.5
        )
        self.best_val_f1 = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

    def _loader(self, X, y, shuffle):
        return DataLoader(FlowSequenceDataset(X, y),
                          batch_size=self.config["batch_size"],
                          shuffle=shuffle, num_workers=0)

    def _train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def _eval_epoch(self, loader):
        self.model.eval()
        all_preds, all_labels, total = [], [], 0.0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                logits = self.model(X_b)
                total += self.criterion(logits, y_b).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return total / len(loader), f1, np.array(all_preds), np.array(all_labels)

    def train(self, X_train, y_train, X_val, y_val):
        train_loader = self._loader(X_train, y_train, True)
        val_loader = self._loader(X_val, y_val, False)
        patience = self.config["early_stopping_patience"]
        no_improve = 0
        print(f"\n[BiLSTM] Training {self.config['epochs']} epochs | "
              f"Params: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(1, self.config["epochs"] + 1):
            tl = self._train_epoch(train_loader)
            vl, vf1, _, _ = self._eval_epoch(val_loader)
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["val_f1"].append(vf1)
            self.scheduler.step(vf1)
            if vf1 > self.best_val_f1:
                self.best_val_f1 = vf1
                torch.save(self.model.state_dict(), self.save_path)
                no_improve = 0
                flag = "✓ saved"
            else:
                no_improve += 1
                flag = f"(no improve {no_improve}/{patience})"
            print(f"  Epoch {epoch:3d}/{self.config['epochs']} | "
                  f"Train: {tl:.4f} | Val: {vl:.4f} | F1: {vf1:.4f} {flag}")
            if no_improve >= patience:
                print(f"[BiLSTM] Early stopping at epoch {epoch}")
                break
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device))
        print(f"[BiLSTM] Best Val F1: {self.best_val_f1:.4f}")
        return self.history

    def evaluate(self, X_test, y_test, class_names=None):
        loader = self._loader(X_test, y_test, False)
        _, f1, preds, labels = self._eval_epoch(loader)
        unique_labels = np.unique(np.concatenate([labels, preds]))
        filtered_names = ([class_names[i] for i in unique_labels]
                          if class_names else None)
        report = classification_report(
            labels, preds, labels=unique_labels,
            target_names=filtered_names, zero_division=0, output_dict=True
        )
        if class_names:
            print(classification_report(
                labels, preds, labels=unique_labels,
                target_names=filtered_names, zero_division=0
            ))
        return {"predictions": preds, "labels": labels,
                "f1_weighted": f1, "report": report}

    def predict_proba(self, X):
        self.model.eval()
        loader = DataLoader(
            FlowSequenceDataset(X, np.zeros(len(X), dtype=np.int64)),
            batch_size=self.config["batch_size"], shuffle=False
        )
        probs = []
        with torch.no_grad():
            for X_b, _ in loader:
                logits = self.model(X_b.to(self.device))
                probs.append(torch.softmax(logits, 1).cpu().numpy())
        return np.vstack(probs)

    def load_best(self):
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device))
        self.model.eval()
        return self


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return (weights / weights.sum() * len(classes)).astype(np.float32)
