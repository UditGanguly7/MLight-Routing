"""
LSTM model for temporal PNT estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class TemporalPNTModel(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size_1: int = 64, hidden_size_2: int = 32, dense_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(out)
        h = h_n[-1]
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        return self.fc2(h)


class TemporalPNT:
    def __init__(self, input_size: int = 8, device: str = "cpu", learning_rate: float = 1e-3):
        self.device = torch.device(device)
        self.model = TemporalPNTModel(input_size=input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None

    def fit_normalizers(self, X: np.ndarray, y: np.ndarray) -> None:
        flat_X = X.reshape(-1, X.shape[-1])
        self.feat_mean = flat_X.mean(axis=0)
        self.feat_std = np.max(np.abs(flat_X - self.feat_mean), axis=0) + 1e-6
        self.target_mean = y.mean(axis=0)
        self.target_std = np.max(np.abs(y - self.target_mean), axis=0) + 1e-6

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / self.feat_std

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def _denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.target_std + self.target_mean

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        self.model.train()
        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)
        n = X_norm.shape[0]
        idx = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i + batch_size]
            xb = torch.tensor(X_norm[batch_idx], dtype=torch.float32, device=self.device)
            yb = torch.tensor(y_norm[batch_idx], dtype=torch.float32, device=self.device)
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(range(0, n, batch_size)), 1)

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        X_norm = self._normalize_X(X)
        n = X_norm.shape[0]
        preds = []
        for i in range(0, n, batch_size):
            xb = torch.tensor(X_norm[i:i + batch_size], dtype=torch.float32, device=self.device)
            pred_norm = self.model(xb).cpu().numpy()
            preds.append(pred_norm)
        y_norm = np.concatenate(preds, axis=0)
        return self._denormalize_y(y_norm) # <--- THE FIX IS HERE

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.feat_mean = ckpt['feat_mean']
        self.feat_std = ckpt['feat_std']
        self.target_mean = ckpt['target_mean']
        self.target_std = ckpt['target_std']
