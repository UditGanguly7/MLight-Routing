"""
Kalman filter for device position tracking.

Fuses noisy GBT position estimates over time to reduce error.

State vector: [lat, lon, v_lat, v_lon]
Measurement: [lat, lon] from GBT
Model: constant velocity
"""

import numpy as np
from typing import Optional


class PositionKalmanFilter:
    """
    4D Kalman filter for 2D position + velocity tracking.

    Usage:
        kf = PositionKalmanFilter(dt_s=1.0)
        for lat, lon in measurements:
            kf.update(lat, lon)      # measurement update
            kf.predict()              # propagate to next step
    """

    def __init__(
        self,
        dt_s: float = 1.0,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        initial_uncertainty: float = 1e2,
    ):
        self.dt = dt_s

        # State transition: constant velocity
        self.F = np.array([
            [1, 0, dt_s, 0],
            [0, 1, 0, dt_s],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix: observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise (small for smooth motion)
        self.Q = process_noise * np.eye(4)

        # Measurement noise
        self.R = measurement_noise * np.eye(2)

        # Initial uncertainty
        self.P0 = initial_uncertainty * np.eye(4)

        # State
        self.x = np.zeros(4)
        self.P = self.P0.copy()
        self.initialized = False

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def initialize(self, lat: float, lon: float) -> None:
        """Initialize state from first measurement."""
        self.x = np.array([lat, lon, 0.0, 0.0], dtype=float)
        self.P = self.P0.copy()
        self.initialized = True

    # --------------------------------------------------------
    # PREDICT & UPDATE
    # --------------------------------------------------------

    def predict(self) -> np.ndarray:
        """
        Prediction step (propagate state forward).

        Safe to call even if not initialized — auto-initializes at origin.
        """
        if not self.initialized:
            # Auto-initialize with zero state and large uncertainty
            self.x = np.zeros(4)
            self.P = self.P0.copy()
            self.initialized = True

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, lat: float, lon: float) -> np.ndarray:
        """
        Update step (correct with measurement).

        Auto-initializes on first call.
        """
        if not self.initialized:
            self.initialize(lat, lon)
            return self.x[:2]

        z = np.array([lat, lon])
        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2]

    # --------------------------------------------------------
    # ACCESSORS
    # --------------------------------------------------------

    def get_position(self) -> np.ndarray:
        """Return current position estimate [lat, lon]."""
        return self.x[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity estimate [v_lat, v_lon]."""
        return self.x[2:].copy()

    def get_uncertainty_km(self) -> float:
        """Approximate position uncertainty in km (1-sigma)."""
        pos_std_deg = np.sqrt(np.maximum(np.diag(self.P)[:2], 0.0))
        # 1 deg ≈ 111 km
        return float(np.linalg.norm(pos_std_deg) * 111.0)


# ============================================================
# HIGH-LEVEL TRACKING FUNCTION
# ============================================================

def run_kalman_tracking(
    measurements: np.ndarray,
    dt_s: float = 1.0,
) -> np.ndarray:
    """
    Apply Kalman filter over sequence of GBT position measurements.

    For each measurement:
        1. Update with new measurement
        2. Predict for next step

    Args:
        measurements: (T, 2) array of [lat, lon]
        dt_s: time step between measurements

    Returns:
        smoothed: (T, 2) filtered positions
    """
    kf = PositionKalmanFilter(dt_s=dt_s)
    smoothed = np.zeros_like(measurements)

    for t, (lat, lon) in enumerate(measurements):
        smoothed[t] = kf.update(float(lat), float(lon))
        kf.predict()

    return smoothed
