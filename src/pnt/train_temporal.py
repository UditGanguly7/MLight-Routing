"""
Training pipeline for temporal PNT module.

Usage:
    python3 -m src.pnt.train_temporal
"""

import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .temporal_generator import generate_temporal_dataset, save_temporal_dataset
from .lstm_model import TemporalPNT


# ============================================================
# CONFIG
# ============================================================

N_SAMPLES = 5000
N_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Temporal PNT Training (LSTM)")
    print("=" * 60)

    np.random.seed(SEED)

    # --------------------------------------------------------
    # 1. Generate data
    # --------------------------------------------------------
    print(f"\n[1/5] Generating {N_SAMPLES} satellite passes...")
    t0 = time.time()
    X, y = generate_temporal_dataset(n_samples=N_SAMPLES, seed=SEED)
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  X: {X.shape}  y: {y.shape}")

    save_temporal_dataset(X, y, out_dir="data/training")

    # --------------------------------------------------------
    # 2. Split
    # --------------------------------------------------------
    print(f"\n[2/5] Splitting (train/val/test = {1-VAL_SPLIT-TEST_SPLIT:.1f}/{VAL_SPLIT}/{TEST_SPLIT})")

    idx = np.arange(N_SAMPLES)
    idx_temp, idx_test = train_test_split(idx, test_size=TEST_SPLIT, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED,
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # --------------------------------------------------------
    # 3. Build model
    # --------------------------------------------------------
    print(f"\n[3/5] Building LSTM...")
    model = TemporalPNT(device="cpu", learning_rate=LEARNING_RATE)
    model.fit_normalizers(X_train, y_train)

    print(f"  Feature means: {model.feat_mean.round(2)}")
    print(f"  Feature stds:  {model.feat_std.round(2)}")
    print(f"  Target means:  {model.target_mean.round(2)}")
    print(f"  Target stds:   {model.target_std.round(2)}")

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters:    {n_params:,}")
    print(f"  Model size:    {n_params * 4 / 1024:.1f} kB")

    # --------------------------------------------------------
    # 4. Train
    # --------------------------------------------------------
    print(f"\n[4/5] Training ({N_EPOCHS} epochs)...")

    best_val_err = float('inf')
    best_model_path = "models/pnt/lstm_temporal_best.pt"
    Path("models/pnt").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # Train epoch
        train_loss = model.train_epoch(X_train, y_train, batch_size=BATCH_SIZE)

        # Validate
        val_pred = model.predict(X_val)
        val_err_km = model.position_error_km(y_val, val_pred)

        # Track best
        if val_err_km < best_val_err:
            best_val_err = val_err_km
            model.save(best_model_path)

        # Log
        epoch_time = time.time() - t0
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}: "
              f"loss={train_loss:.5f}, "
              f"val_err={val_err_km:7.2f} km, "
              f"time={epoch_time:.1f}s")

    # --------------------------------------------------------
    # 5. Final evaluation on test set
    # --------------------------------------------------------
    print(f"\n[5/5] Final evaluation...")
    model.load(best_model_path)
    test_pred = model.predict(X_test)
    test_err_km = model.position_error_km(y_test, test_pred)

    print(f"  Best val error: {best_val_err:.2f} km")
    print(f"  Test error:     {test_err_km:.2f} km")

    # Compute distribution of errors
    err_lat = (y_test[:, 0] - test_pred[:, 0]) * 111.0
    err_lon = (y_test[:, 1] - test_pred[:, 1]) * 111.0 * np.cos(np.deg2rad(y_test[:, 0]))
    err_km = np.sqrt(err_lat**2 + err_lon**2)

    print(f"  Test error distribution:")
    print(f"    median: {np.median(err_km):7.2f} km")
    print(f"    90th %: {np.percentile(err_km, 90):7.2f} km")
    print(f"    95th %: {np.percentile(err_km, 95):7.2f} km")
    print(f"    max:    {np.max(err_km):7.2f} km")

    print(f"\n  Model saved: {best_model_path}")
    print("=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
