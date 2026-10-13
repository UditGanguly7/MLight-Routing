"""
Topology predictor: predicts future ISL link states.

Uses a shared per-edge LSTM. Each edge is treated as an independent
time series and fed through the same LSTM (weight sharing across edges).

Input:  (batch, T_obs=30, num_edges=2145)
Output: (batch, T_pred=10, num_edges=2145) — binary predictions
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


# ============================================================
# MODEL
# ============================================================

class TopologyPredictor(nn.Module):
    """
    Shared per-edge LSTM predictor.

    Reshapes input so each edge becomes a separate batch item,
    runs through a small LSTM, then reshapes back.

    Architecture:
        Input:  (B, T_obs=30, E=2145)
        Reshape: (B*E, T_obs, 1)
        LSTM(input=1, hidden=16, layers=1)
        Linear(16 → 32) + ReLU
        Linear(32 → T_pred=10)
        Sigmoid
        Reshape: (B, T_pred, E)
    """

    def __init__(
        self,
        obs_steps: int = 30,
        pred_steps: int = 10,
        hidden_size: int = 16,
        fc_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, pred_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_obs, E)
        Returns:
            (B, T_pred, E)
        """
        B, T, E = x.shape

        # Reshape to (B*E, T, 1) — one sequence per edge
        x_reshaped = x.transpose(1, 2).reshape(B * E, T, 1)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x_reshaped)
        h = h_n[-1]                        # (B*E, hidden_size)

        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)                  # (B*E, T_pred)
        out = torch.sigmoid(out)

        # Reshape back to (B, T_pred, E)
        out = out.reshape(B, E, self.pred_steps).transpose(1, 2)
        return out


# ============================================================
# HIGH-LEVEL WRAPPER
# ============================================================

class TopologyWrapper:
    """
    Wrapper with training utilities and normalization.
    """

    def __init__(
        self,
        obs_steps: int = 30,
        pred_steps: int = 10,
        device: str = "cpu",
        learning_rate: float = 1e-3,
    ):
        self.device = torch.device(device)
        self.model = TopologyPredictor(
            obs_steps=obs_steps, pred_steps=pred_steps
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.BCELoss()

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 8,
    ) -> float:
        """
        One training epoch.

        Note: batch_size is small (8) because each batch item contains
        30 × 2145 = 64,350 values. Larger batches use too much memory.
        """
        self.model.train()
        n = X.shape[0]
        idx = np.random.permutation(n)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = idx[i:i + batch_size]
            xb = torch.tensor(
                X[batch_idx], dtype=torch.float32, device=self.device
            )
            yb = torch.tensor(
                y[batch_idx], dtype=torch.float32, device=self.device
            )

            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """Predict link states."""
        self.model.eval()
        n = X.shape[0]
        preds = []

        for i in range(0, n, batch_size):
            xb = torch.tensor(
                X[i:i + batch_size], dtype=torch.float32, device=self.device
            )
            pred = self.model(xb).cpu().numpy()
            preds.append(pred)

        return np.concatenate(preds, axis=0)

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Binary accuracy between predicted and true link states."""
        y_bin = (y_pred > threshold).astype(np.float32)
        return float((y_bin == y_true).mean())

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """F1 score for predicting active links (class 1)."""
        y_bin = (y_pred > threshold).astype(np.float32)
        tp = ((y_bin == 1) & (y_true == 1)).sum()
        fp = ((y_bin == 1) & (y_true == 0)).sum()
        fn = ((y_bin == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return float(f1)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': self.model.state_dict()}, path)
        print(f"Topology model saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"Topology model loaded: {path}")

    def model_size_kb(self) -> float:
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / 1024.0
