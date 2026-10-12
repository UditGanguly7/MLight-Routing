"""
LSTM model for temporal PNT estimation.

Input:  (batch, T=120, F=7) sequence of
        [sat_x_ref, sat_y_ref, sat_z_ref, doppler, delay, snr, elevation]
Output: (batch, 2) [lat_deg, lon_deg]
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


# ============================================================
# NEURAL NETWORK
# ============================================================

class TemporalPNTModel(nn.Module):
    """
    2-layer LSTM for sequence-to-position regression.

    Architecture:
        LSTM(7  -> 64, return_sequences=True)
        LSTM(64 -> 32, return_sequences=False)
        Dropout(0.2)
        Linear(32 -> 32) + ReLU
        Linear(32 -> 2)
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dense_size: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, 7)
        Returns:
            (batch, 2)
        """
        # LSTM 1: keep all timesteps
        out, _ = self.lstm1(x)

        # LSTM 2: only final timestep
        _, (h_n, _) = self.lstm2(out)
        h = h_n[-1]

        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out


# ============================================================
# HIGH-LEVEL WRAPPER
# ============================================================

class TemporalPNT:
    """
    High-level wrapper with normalization + training utilities.
    """

    def __init__(
        self,
        input_size: int = 7,
        device: str = "cpu",
        learning_rate: float = 1e-3,
    ):
        self.device = torch.device(device)
        self.model = TemporalPNTModel(input_size=input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Feature and target normalizers (computed on training data)
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    def fit_normalizers(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute normalization stats from training data.

        Uses max-abs deviation (not std) to robustly handle
        features with near-zero variance.

        X: (N, T, F)
        y: (N, 2)
        """
        flat_X = X.reshape(-1, X.shape[-1])
        self.feat_mean = flat_X.mean(axis=0)
        # Robust scale: max absolute deviation from mean
        self.feat_std = np.max(np.abs(flat_X - self.feat_mean), axis=0) + 1e-6

        self.target_mean = y.mean(axis=0)
        self.target_std = np.max(np.abs(y - self.target_mean), axis=0) + 1e-6

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / self.feat_std

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def _denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.target_std + self.target_mean

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
    ) -> float:
        """One training epoch."""
        self.model.train()

        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)

        n = X_norm.shape[0]
        idx = np.random.permutation(n)
        total_loss = 0.0
        num_batches = 0

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
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict positions in degrees."""
        self.model.eval()

        X_norm = self._normalize_X(X)
        n = X_norm.shape[0]
        preds = []

        for i in range(0, n, batch_size):
            xb = torch.tensor(
                X_norm[i:i + batch_size],
                dtype=torch.float32,
                device=self.device,
            )
            pred_norm = self.model(xb).cpu().numpy()
            preds.append(pred_norm)

        y_norm = np.concatenate(preds, axis=0)
        return self._denormalize_y(y_norm)

    def position_error_km(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMS position error in km."""
        err_lat_km = (y_true[:, 0] - y_pred[:, 0]) * 111.0
        err_lon_km = (y_true[:, 1] - y_pred[:, 1]) * 111.0 * np.cos(np.deg2rad(np.mean(y_true[:, 0])))
        return float(np.sqrt(np.mean(err_lat_km**2 + err_lon_km**2)))

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

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
        print(f"Model saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.feat_mean = ckpt['feat_mean']
        self.feat_std = ckpt['feat_std']
        self.target_mean = ckpt['target_mean']
        self.target_std = ckpt['target_std']
        print(f"Model loaded: {path}")

    def model_size_kb(self) -> float:
        """Approximate model size in kB (float32 = 4 bytes/param)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / 1024.0
