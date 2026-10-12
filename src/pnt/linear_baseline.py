"""
Linear regression baseline for temporal PNT.

Provides a simple, fast linear model that maps flattened sequences
to position predictions. Used as a baseline and as a starting point
for LSTM residual learning.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pickle
from sklearn.linear_model import Ridge


class LinearPNT:
    """
    Ridge regression on flattened temporal features.

    Input:  (N, T, F) sequences
    Output: (N, 2) predictions
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearPNT':
        X_flat = self._flatten(X)

        # Standardize features
        self.feat_mean = X_flat.mean(axis=0)
        self.feat_std = X_flat.std(axis=0) + 1e-8
        X_norm = (X_flat - self.feat_mean) / self.feat_std

        self.model.fit(X_norm, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        X_norm = (X_flat - self.feat_mean) / self.feat_std
        return self.model.predict(X_norm)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Linear PNT saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'LinearPNT':
        with open(path, 'rb') as f:
            return pickle.load(f)
