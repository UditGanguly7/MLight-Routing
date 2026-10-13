"""
Topology data generator for link-state prediction.

Generates sequences of ISL (inter-satellite link) states over time.
The predictor learns to forecast future link states from the
current topology evolution.

For each sample:
    Input:  (T_obs, E) sequence of edge states
    Output: (T_pred, E) future edge states

Where E = number of unordered satellite pairs = N*(N-1)/2.
"""

import numpy as np
from typing import Tuple
from pathlib import Path

from ..simulation.constellation import create_default_constellation
from ..simulation.link import compute_adjacency_matrix


# ============================================================
# CONFIGURATION
# ============================================================

OBSERVATION_STEPS = 30       # 30 timesteps of history
PREDICTION_STEPS = 10        # predict 10 steps ahead
TIME_STEP_S = 3.0            # 3 seconds per step


# ============================================================
# EDGE EXTRACTION
# ============================================================

def num_edges(N: int) -> int:
    """Number of unordered pairs in constellation."""
    return N * (N - 1) // 2


def adjacency_to_edge_list(adj: np.ndarray) -> np.ndarray:
    """Convert (N, N) adjacency to flat edge vector of length N*(N-1)/2."""
    N = adj.shape[0]
    iu = np.triu_indices(N, k=1)
    return adj[iu].astype(np.float32)


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_topology_sample(
    constellation,
    t_start: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one (observation, prediction) sample.

    Args:
        constellation: Constellation object
        t_start: starting time [s]

    Returns:
        X: (OBSERVATION_STEPS, num_edges) — past link states
        y: (PREDICTION_STEPS, num_edges) — future link states
    """
    total_steps = OBSERVATION_STEPS + PREDICTION_STEPS
    N = len(constellation)
    num_pairs = num_edges(N)

    edges = np.zeros((total_steps, num_pairs), dtype=np.float32)

    for i in range(total_steps):
        t = t_start + i * TIME_STEP_S
        positions = constellation.get_positions(t)
        adj = compute_adjacency_matrix(positions)
        edges[i] = adjacency_to_edge_list(adj)

    X = edges[:OBSERVATION_STEPS]
    y = edges[OBSERVATION_STEPS:]
    return X, y


def generate_topology_dataset(
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset.

    Returns:
        X: (n_samples, OBSERVATION_STEPS, num_edges)
        y: (n_samples, PREDICTION_STEPS, num_edges)
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    constellation = create_default_constellation()
    N = len(constellation)
    num_pairs = num_edges(N)

    X = np.zeros((n_samples, OBSERVATION_STEPS, num_pairs), dtype=np.float32)
    y = np.zeros((n_samples, PREDICTION_STEPS, num_pairs), dtype=np.float32)

    for i in range(n_samples):
        t_start = np.random.uniform(0, 3600)
        X[i], y[i] = generate_topology_sample(constellation, t_start)

    np.random.set_state(rng_state)
    return X, y


# ============================================================
# PERSISTENCE
# ============================================================

def save_topology_dataset(X, y, out_dir: str = "data/training") -> None:
    """Save dataset to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "topology_X.npy", X)
    np.save(out_path / "topology_y.npy", y)
    print(f"Saved: {out_path / 'topology_X.npy'}  (shape: {X.shape})")
    print(f"Saved: {out_path / 'topology_y.npy'}  (shape: {y.shape})")


def load_topology_dataset(in_dir: str = "data/training"):
    """Load dataset from disk."""
    in_path = Path(in_dir)
    X = np.load(in_path / "topology_X.npy")
    y = np.load(in_path / "topology_y.npy")
    return X, y

