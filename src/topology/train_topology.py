"""
Training pipeline for topology predictor.

Usage:
    python3 -m src.topology.train_topology
"""

import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .topology_generator import (
    generate_topology_dataset,
    save_topology_dataset,
    OBSERVATION_STEPS,
    PREDICTION_STEPS,
)
from .topology_model import TopologyWrapper


# ============================================================
# CONFIG
# ============================================================

N_SAMPLES = 500
N_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
SEED = 42

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Topology Predictor Training")
    print("=" * 60)

    np.random.seed(SEED)

    # --------------------------------------------------------
    # 1. Generate data
    # --------------------------------------------------------
    print(f"\n[1/5] Generating {N_SAMPLES} topology samples...")
    print(f"  (This may take ~{N_SAMPLES * 0.2:.0f} seconds)")
    t0 = time.time()
    X, y = generate_topology_dataset(n_samples=N_SAMPLES, seed=SEED)
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  X: {X.shape}  y: {y.shape}")

    save_topology_dataset(X, y, out_dir="data/training")

    # --------------------------------------------------------
    # 2. Split
    # --------------------------------------------------------
    print(f"\n[2/5] Splitting...")
    idx = np.arange(N_SAMPLES)
    idx_temp, idx_test = train_test_split(idx, test_size=TEST_SPLIT, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=SEED
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
    print(f"\n[3/5] Building topology predictor...")
    model = TopologyWrapper(
        obs_steps=OBSERVATION_STEPS,
        pred_steps=PREDICTION_STEPS,
        device="cpu",
        learning_rate=LEARNING_RATE,
    )
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Model size: {model.model_size_kb():.2f} kB")

    # --------------------------------------------------------
    # 4. Train
    # --------------------------------------------------------
    print(f"\n[4/5] Training ({N_EPOCHS} epochs)...")

    best_val_acc = 0.0
    best_model_path = "models/topology/topology_best.pt"
    Path("models/topology").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = model.train_epoch(X_train, y_train, batch_size=BATCH_SIZE)

        # Validation
        val_pred = model.predict(X_val)
        val_acc = model.accuracy(y_val, val_pred)
        val_f1 = model.f1_score(y_val, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_model_path)

        epoch_time = time.time() - t0
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}: "
              f"loss={train_loss:.5f}, "
              f"val_acc={val_acc:.4f}, "
              f"val_f1={val_f1:.4f}, "
              f"time={epoch_time:.1f}s")

    # --------------------------------------------------------
    # 5. Final evaluation on test set
    # --------------------------------------------------------
    print(f"\n[5/5] Final evaluation...")
    model.load(best_model_path)

    test_pred = model.predict(X_test)
    test_acc = model.accuracy(y_test, test_pred)
    test_f1 = model.f1_score(y_test, test_pred)

    # Per-timestep accuracy
    print(f"  Overall test accuracy: {test_acc:.4f}")
    print(f"  Overall test F1:       {test_f1:.4f}")
    print(f"  Best val accuracy:     {best_val_acc:.4f}")
    print()
    print(f"  Per-timestep test accuracy:")
    for k in range(PREDICTION_STEPS):
        acc_k = model.accuracy(y_test[:, k, :], test_pred[:, k, :])
        print(f"    Step +{k+1:2d}: acc={acc_k:.4f}")

    print(f"\n  Model saved: {best_model_path}")
    print("=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

