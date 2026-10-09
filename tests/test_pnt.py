"""
Unit tests for PNT estimation module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.pnt.data_generator import (
    generate_sample,
    generate_dataset,
    random_device_position,
)
from src.pnt.gbt_model import PNTModel
from src.pnt.kalman_filter import PositionKalmanFilter, run_kalman_tracking
from src.simulation import constants as C


# ============================================================
# DATA GENERATOR TESTS
# ============================================================

class TestDataGenerator:
    def test_device_position_on_earth(self):
        _, _, pos = random_device_position()
        r = np.linalg.norm(pos)
        assert abs(r - C.EARTH_RADIUS_KM) < 0.1

    def test_generate_sample_shapes(self):
        X, y, A = generate_sample()
        assert X.shape == (7,)
        assert y.shape == (2,)
        assert A.shape == (2,)

    def test_generate_dataset_shapes(self):
        X, y, A = generate_dataset(n_samples=50, seed=42)
        assert X.shape == (50, 7)
        assert y.shape == (50, 2)
        assert A.shape == (50, 2)

    def test_doppler_bounded(self):
        X, _, _ = generate_dataset(n_samples=200, seed=42)
        assert np.all(np.abs(X[:, 3]) < C.MAX_DOPPLER_HZ * 1.1)

    def test_delay_in_range(self):
        X, _, _ = generate_dataset(n_samples=200, seed=42)
        assert np.all(X[:, 4] >= C.MIN_DELAY_MS * 0.95)
        assert np.all(X[:, 4] <= C.MAX_DELAY_MS * 1.05)

    def test_reproducibility(self):
        X1, y1, A1 = generate_dataset(n_samples=20, seed=42)
        X2, y2, A2 = generate_dataset(n_samples=20, seed=42)
        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)
        assert np.allclose(A1, A2)

    def test_satellite_position_in_features(self):
        X, _, _ = generate_dataset(n_samples=100, seed=42)
        sat_norm = np.linalg.norm(X[:, :3], axis=1)
        assert np.all(sat_norm > 6000)
        assert np.all(sat_norm < 7500)


# ============================================================
# GBT MODEL TESTS
# ============================================================

class TestGBTModel:
    def setup_method(self):
        self.X, self.y, self.A = generate_dataset(n_samples=500, seed=42)
        self.model = PNTModel(n_estimators=20, max_depth=4)
        self.model.fit(self.X[:400], self.y[:400])

    def test_model_fitted(self):
        assert self.model.is_fitted

    def test_predict_shape(self):
        pred = self.model.predict(self.X[400:])
        assert pred.shape == (100, 2)

    def test_predict_without_fit_raises(self):
        model = PNTModel()
        with pytest.raises(RuntimeError):
            model.predict(self.X)


# ============================================================
# KALMAN FILTER TESTS
# ============================================================

class TestKalmanFilter:
    def test_initialization(self):
        kf = PositionKalmanFilter()
        kf.initialize(20.0, 80.0)
        assert kf.initialized

    def test_predict_auto_initializes(self):
        kf = PositionKalmanFilter()
        pos = kf.predict()
        assert pos.shape == (2,)
        assert kf.initialized

    def test_update_returns_position(self):
        kf = PositionKalmanFilter()
        pos = kf.update(20.0, 80.0)
        assert pos.shape == (2,)

    def test_filter_reduces_noise(self):
        true = np.column_stack([
            np.linspace(20, 21, 100),
            np.linspace(80, 81, 100),
        ])
        np.random.seed(42)
        noisy = true + np.random.randn(*true.shape) * 0.1
        filtered = run_kalman_tracking(noisy)

        err_noisy = np.mean(np.linalg.norm(noisy - true, axis=1))
        err_filtered = np.mean(np.linalg.norm(filtered - true, axis=1))

        assert err_filtered < err_noisy

    def test_uncertainty_decreases(self):
        kf = PositionKalmanFilter()
        kf.update(20.0, 80.0)
        u1 = kf.get_uncertainty_km()
        for _ in range(20):
            kf.update(20.0, 80.0)
            kf.predict()
        u2 = kf.get_uncertainty_km()
        assert u2 < u1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
