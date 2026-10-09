"""
Training pipeline for PNT module (two-stage estimator).

Usage:
    python3 -m src.pnt.train
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split

from .data_generator import generate_dataset, save_dataset
from .gbt_model import PNTModel


def position_error_km(y_true, y_pred):
    """Root-mean-square position error in km."""
    err_lat_km = (y_true[:, 0] - y_pred[:, 0]) * 111.0
    err_lon_km = (y_true[:, 1] - y_pred[:, 1]) * 111.0 * np.cos(np.deg2rad(np.mean(y_true[:, 0])))
    return float(np.sqrt(np.mean(err_lat_km**2 + err_lon_km**2)))


def main():
    print("=" * 60)
    print("PNT Module Training (Two-Stage)")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/4] Generating training data...")
    t0 = time.time()
    X, y_res, A = generate_dataset(n_samples=10000, seed=42)
    print(f"  Generated {X.shape[0]} samples in {time.time()-t0:.1f}s")

    save_dataset(X, y_res, A, out_dir="data/training")

    # Split (keeping three aligned arrays)
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y_res[idx_train], y_res[idx_test]
    A_train, A_test = A[idx_train], A[idx_test]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Train GBT on residuals
    print("\n[2/4] Training GBT on residuals...")
    t0 = time.time()
    model = PNTModel(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # 3. Evaluate
    print("\n[3/4] Evaluating...")

    # Analytical-only baseline
    err_analytical = position_error_km(y_test + A_test, A_test)

    # GBT-corrected
    y_pred_res = model.predict(X_test)
    y_pred_corrected = A_test + y_pred_res

    # True positions
    y_true = y_test + A_test

    err_corrected = position_error_km(y_true, y_pred_corrected)

    print(f"  Analytical baseline error:  {err_analytical:.2f} km")
    print(f"  GBT-corrected error:        {err_corrected:.2f} km")
    print(f"  Improvement:                {(1 - err_corrected/err_analytical)*100:.1f}%")

    # 4. Save
    print("\n[4/4] Saving model...")
    model_path = "models/pnt/gbt_model.pkl"
    model.save(model_path)
    size_kb = model.get_model_size_kb(model_path)
    print(f"  Model size: {size_kb:.2f} kB")

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

