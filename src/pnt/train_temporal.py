"""
Training pipeline for temporal PNT with LSTM residual learning.

Strategy:
    1. Train Ridge regression on flattened sequences (baseline)
    2. Train LSTM to predict residual (true - linear_prediction)
    3. Combine at inference: linear_pred + lstm_residual

Usage:
    python3 -m src.pnt.train_temporal
"""

import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .temporal_generator import generate_temporal_dataset, save_temporal_dataset
from .linear_baseline import LinearPNT
from .lstm_model import TemporalPNT


N_SAMPLES = 5000
N_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
SEED = 42
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


def angular_error_km(true_abs, pred_abs):
    lat_err_km = (true_abs[:, 0] - pred_abs[:, 0]) * 111.0
    lon_diff = true_abs[:, 1] - pred_abs[:, 1]
    lon_diff = ((lon_diff + 180) % 360) - 180
    lon_err_km = lon_diff * 111.0 * np.cos(np.deg2rad(np.mean(true_abs[:, 0])))
    return float(np.sqrt(np.mean(lat_err_km**2 + lon_err_km**2)))


def main():
    print("=" * 60)
    print("Temporal PNT Training (LSTM on Linear Residual)")
    print("=" * 60)

    np.random.seed(SEED)

    # 1. Generate data
    print(f"\n[1/6] Generating {N_SAMPLES} satellite passes...")
    t0 = time.time()
    X, y, R = generate_temporal_dataset(n_samples=N_SAMPLES, seed=SEED)
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  X: {X.shape}  y: {y.shape}  R: {R.shape}")
    save_temporal_dataset(X, y, R, out_dir="data/training")

    # 2. Split
    print(f"\n[2/6] Splitting...")
    idx = np.arange(N_SAMPLES)
    idx_temp, idx_test = train_test_split(idx, test_size=TEST_SPLIT, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=SEED
    )

    X_train, y_train, R_train = X[idx_train], y[idx_train], R[idx_train]
    X_val, y_val, R_val = X[idx_val], y[idx_val], R[idx_val]
    X_test, y_test, R_test = X[idx_test], y[idx_test], R[idx_test]

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # 3. Train linear baseline
    print(f"\n[3/6] Training linear baseline (Ridge)...")
    t0 = time.time()
    linear = LinearPNT(alpha=1.0)
    linear.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Evaluate linear
    y_pred_lin_val = linear.predict(X_val)
    y_pred_lin_test = linear.predict(X_test)

    val_err_lin = angular_error_km(y_val + R_val, y_pred_lin_val + R_val)
    test_err_lin = angular_error_km(y_test + R_test, y_pred_lin_test + R_test)

    print(f"  Linear — val error:  {val_err_lin:.2f} km")
    print(f"  Linear — test error: {test_err_lin:.2f} km")

    linear.save("models/pnt/linear_pnt.pkl")

    # 4. Compute residuals
    print(f"\n[4/6] Computing residuals...")
    y_train_lin = linear.predict(X_train)
    y_val_lin = linear.predict(X_val)

    residual_train = y_train - y_train_lin
    residual_val = y_val - y_val_lin

    print(f"  Train residual mean: {residual_train.mean(axis=0).round(3)}")
    print(f"  Train residual std:  {residual_train.std(axis=0).round(3)}")

    # 5. Train LSTM on residual
    print(f"\n[5/6] Training LSTM on residuals ({N_EPOCHS} epochs)...")
    model = TemporalPNT(device="cpu", learning_rate=LEARNING_RATE)
    model.fit_normalizers(X_train, residual_train)

    print(f"  Parameters:    {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"  Model size:    {sum(p.numel() for p in model.model.parameters()) * 4 / 1024:.1f} kB")

    best_val_err = float('inf')
    best_model_path = "models/pnt/lstm_temporal_best.pt"
    Path("models/pnt").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = model.train_epoch(X_train, residual_train, batch_size=BATCH_SIZE)

        # Val: LSTM predicts residual, add to linear pred
        residual_pred_val = model.predict(X_val)
        y_pred_val = y_val_lin + residual_pred_val
        val_err = angular_error_km(y_val + R_val, y_pred_val + R_val)

        if val_err < best_val_err:
            best_val_err = val_err
            model.save(best_model_path)

        epoch_time = time.time() - t0
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}: "
              f"loss={train_loss:.5f}, "
              f"val_err={val_err:7.2f} km, "
              f"time={epoch_time:.1f}s")

    # 6. Final evaluation
    print(f"\n[6/6] Final evaluation...")
    model.load(best_model_path)

    # Combined prediction: linear + LSTM residual
    residual_pred_test = model.predict(X_test)
    y_pred_combined = y_pred_lin_test + residual_pred_test

    err_combined = angular_error_km(y_test + R_test, y_pred_combined + R_test)

    print(f"\n  === FINAL RESULTS ===")
    print(f"  Linear only:              {test_err_lin:.2f} km")
    print(f"  Linear + LSTM residual:   {err_combined:.2f} km")
    print(f"  Improvement:              {(1 - err_combined/test_err_lin)*100:.1f}%")
    print(f"  Best val error:           {best_val_err:.2f} km")

    # Distribution
    lat_err = (y_test[:,0] + R_test[:,0] - y_pred_combined[:,0] - R_test[:,0]) * 111.0
    lon_diff = (y_test[:,1] - y_pred_combined[:,1])
    lon_diff = ((lon_diff + 180) % 360) - 180
    lon_err = lon_diff * 111.0 * np.cos(np.deg2rad(y_test[:,0]))
    err_km = np.sqrt(lat_err**2 + lon_err**2)

    print(f"\n  Error distribution (combined):")
    print(f"    median: {np.median(err_km):7.2f} km")
    print(f"    90th %: {np.percentile(err_km, 90):7.2f} km")
    print(f"    95th %: {np.percentile(err_km, 95):7.2f} km")
    print(f"    max:    {np.max(err_km):7.2f} km")

    print("=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
