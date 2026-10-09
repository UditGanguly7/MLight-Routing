"""
Gradient Boosted Tree model for position estimation.

Learns: (Doppler, delay, SNR, elevation) → (lat, lon)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PNTModel:
    """
    Gradient Boosted Tree regressor for device position estimation.

    Pipeline:
        1. Standardize features (zero mean, unit variance)
        2. Multi-output GBT (one tree per target: lat, lon)
        3. Inverse-transform predictions if needed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state

        # Scaler for input features
        self.scaler = StandardScaler()

        # GBT: one model per target (lat, lon)
        base_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
        )
        self.model = MultiOutputRegressor(base_model, n_jobs=-1)

        self.is_fitted = False

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PNTModel':
        """
        Train the model.

        Args:
            X: (n_samples, 4) features
            y: (n_samples, 2) targets [lat, lon]
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict positions.

        Args:
            X: (n_samples, 4) features
        Returns:
            (n_samples, 2) predicted [lat, lon]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            dict with MAE, RMSE, and position error (km)
        """
        y_pred = self.predict(X)

        mae_lat = mean_absolute_error(y[:, 0], y_pred[:, 0])
        mae_lon = mean_absolute_error(y[:, 1], y_pred[:, 1])
        rmse_lat = np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0]))
        rmse_lon = np.sqrt(mean_squared_error(y[:, 1], y_pred[:, 1]))

        # Convert degree error to km (rough approximation)
        # 1 deg latitude ≈ 111 km
        # 1 deg longitude ≈ 111 * cos(lat) km
        lat_err_km = mae_lat * 111.0
        lon_err_km = mae_lon * 111.0 * np.cos(np.deg2rad(np.mean(y[:, 0])))
        pos_err_km = np.sqrt(lat_err_km**2 + lon_err_km**2)

        return {
            'mae_lat_deg': float(mae_lat),
            'mae_lon_deg': float(mae_lon),
            'rmse_lat_deg': float(rmse_lat),
            'rmse_lon_deg': float(rmse_lon),
            'pos_error_km': float(pos_err_km),
        }

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'PNTModel':
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_model_size_kb(self, path: str) -> float:
        """Get serialized model size in kilobytes."""
        return Path(path).stat().st_size / 1024.0
