"""
End-to-end integration demo for MLight-Routing.

Demonstrates the complete pipeline:
    1. PNT estimation (device position from satellite observables)
    2. Topology prediction (future ISL states)
    3. DRL routing (packet forwarding)

Usage:
    python3 -m src.integration.demo
"""

import numpy as np
import time
from pathlib import Path

from ..simulation.constellation import create_default_constellation
from ..simulation.link import compute_adjacency_matrix
from ..simulation import constants as C

from ..pnt.temporal_generator import (
    generate_temporal_sample,
    PASS_NUM_STEPS,
)
from ..pnt.linear_baseline import LinearPNT
from ..pnt.lstm_model import TemporalPNT

from ..topology.topology_generator import (
    generate_topology_sample,
    OBSERVATION_STEPS,
    PREDICTION_STEPS,
)
from ..topology.topology_model import TopologyWrapper

from ..router.env_v2 import RoutingEnvV2
from ..router.dqn_agent import DQNAgent


# ============================================================
# CONFIG
# ============================================================

SEED = 42


# ============================================================
# HEADERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_step(n: int, title: str) -> None:
    print()
    print(f"--- Step {n}: {title} ---")


# ============================================================
# PHASE 1: PNT ESTIMATION
# ============================================================

def phase_1_pnt(seed: int = SEED):
    """
    Demonstrate PNT: estimate device position from satellite pass observables.

    Returns:
        dict with pnt results
    """
    print_step(1, "PNT Estimation (device position from link observables)")

    # Generate a synthetic satellite pass
    seq, target_delta, sat_subpoint = generate_temporal_sample()

    # Convert delta target back to absolute
    true_lat = target_delta[0] + sat_subpoint[0]
    true_lon = target_delta[1] + sat_subpoint[1]

    print(f"  True device position:     ({true_lat:+.2f}°, {true_lon:+.2f}°)")
    print(f"  Satellite subpoint t=0:   ({sat_subpoint[0]:+.2f}°, {sat_subpoint[1]:+.2f}°)")
    print(f"  Sequence shape:           {seq.shape}")

    # Load models
    linear_path = Path("models/pnt/linear_pnt.pkl")
    lstm_path = Path("models/pnt/lstm_temporal_best.pt")

    if not linear_path.exists() or not lstm_path.exists():
        print(f"  ⚠️  PNT models not found — skipping this phase")
        return None

    linear = LinearPNT.load(str(linear_path))
    lstm = TemporalPNT(device="cpu")
    lstm.load(str(lstm_path))

    # Predict
    X = seq[np.newaxis, :, :]     # add batch dim
    delta_lin = linear.predict(X)
    delta_res = lstm.predict(X)
    delta_pred = delta_lin + delta_res

    pred_lat = delta_pred[0, 0] + sat_subpoint[0]
    pred_lon = delta_pred[0, 1] + sat_subpoint[1]

    # Convert to km
    lat_err_km = abs(pred_lat - true_lat) * 111.0
    lon_err_km = abs(pred_lon - true_lon) * 111.0 * np.cos(np.deg2rad(true_lat))
    pos_err_km = np.sqrt(lat_err_km**2 + lon_err_km**2)

    print(f"  Predicted position:       ({pred_lat:+.2f}°, {pred_lon:+.2f}°)")
    print(f"  Position error:           {pos_err_km:.1f} km")
    print(f"  ✓ PNT estimation complete")

    return {
        'true_lat': true_lat,
        'true_lon': true_lon,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon,
        'error_km': pos_err_km,
    }


# ============================================================
# PHASE 2: TOPOLOGY PREDICTION
# ============================================================

def phase_2_topology(seed: int = SEED):
    """
    Demonstrate topology prediction: forecast future ISL states.

    Returns:
        dict with topology results
    """
    print_step(2, "Topology Prediction (future ISL link states)")

    constellation = create_default_constellation()
    t_start = 0.0

    # Generate a topology sample
    X, y_true = generate_topology_sample(constellation, t_start)
    X = X[np.newaxis, :, :]        # (1, 30, 2145)
    y_true = y_true[np.newaxis, :, :]  # (1, 10, 2145)

    print(f"  Input shape:              {X.shape}")
    print(f"  Prediction horizon:       {PREDICTION_STEPS * 3}s ({PREDICTION_STEPS} steps × 3s)")
    print(f"  Current active links:     {int(X[0, -1].sum())}")

    # Load model
    topo_path = Path("models/topology/topology_best.pt")
    if not topo_path.exists():
        print(f"  ⚠️  Topology model not found — skipping this phase")
        return None

    wrapper = TopologyWrapper(device="cpu")
    wrapper.load(str(topo_path))

    # Predict
    y_pred = wrapper.predict(X)
    acc = wrapper.accuracy(y_true, y_pred)
    f1 = wrapper.f1_score(y_true, y_pred)

    # Per-step
    print(f"  Per-step accuracy:")
    for k in range(PREDICTION_STEPS):
        step_acc = wrapper.accuracy(y_true[:, k, :], y_pred[:, k, :])
        print(f"    Step +{k+1:2d}: {step_acc:.4f}")

    print(f"  Overall accuracy:         {acc:.4f}")
    print(f"  Overall F1 (link class):  {f1:.4f}")
    print(f"  ✓ Topology prediction complete")

    return {
        'accuracy': acc,
        'f1': f1,
    }


# ============================================================
# PHASE 3: DRL ROUTING
# ============================================================

def phase_3_routing(seed: int = SEED):
    """
    Demonstrate DRL routing: forward a packet through the constellation.

    Returns:
        dict with routing results
    """
    print_step(3, "DRL Routing (packet forwarding through constellation)")

    env = RoutingEnvV2(seed=seed)
    obs, info = env.reset()

    src = info['current_sat']
    dst = info['dest_sat']
    print(f"  Source satellite:         Sat {src}")
    print(f"  Destination satellite:    Sat {dst}")
    print(f"  Initial distance:         {np.linalg.norm(env.constellation.get_positions(env.time_s)[src] - env.constellation.get_positions(env.time_s)[dst]):.1f} km")

    # Load router
    router_path = Path("models/router/dqn_v2_best.pt")
    if not router_path.exists():
        print(f"  ⚠️  Router model not found — skipping this phase")
        return None

    agent = DQNAgent(obs_dim=35, action_dim=6, device="cpu")
    agent.load(str(router_path))

    # Route packet
    hop_count = 0
    max_hops = 20
    reason = 'unknown'

    while hop_count < max_hops:
        mask = env.action_masks()
        action = agent.select_action(obs, mask, greedy=True)

        obs, reward, term, trunc, info = env.step(action)
        hop_count += 1

        if term or trunc:
            reason = info.get('reason', 'unknown')
            break

    print(f"  Hops taken:               {hop_count}")
    print(f"  Termination reason:       {reason}")
    print(f"  Final cumulative reward:  {reward:.3f}")

    if reason == 'delivered':
        print(f"  ✓ Packet delivered")
    else:
        print(f"  ✗ Packet not delivered ({reason})")

    return {
        'hops': hop_count,
        'delivered': reason == 'delivered',
        'reason': reason,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("MLight-Routing: End-to-End Integration Demo")

    t_total = time.time()

    # ============================================================
    # Phase 1: PNT — evaluate over 50 samples
    # ============================================================
    print_step(1, "PNT Estimation (device position from link observables)")

    linear_path = Path("models/pnt/linear_pnt.pkl")
    lstm_path = Path("models/pnt/lstm_temporal_best.pt")

    if linear_path.exists() and lstm_path.exists():
        linear = LinearPNT.load(str(linear_path))
        lstm = TemporalPNT(device="cpu")
        lstm.load(str(lstm_path))

        n_test = 50
        errors = []
        for i in range(n_test):
            seq, target_delta, subpoint = generate_temporal_sample()
            true_lat = target_delta[0] + subpoint[0]
            true_lon = target_delta[1] + subpoint[1]

            X = seq[np.newaxis, :, :]
            delta_pred = linear.predict(X) + lstm.predict(X)
            pred_lat = delta_pred[0, 0] + subpoint[0]
            pred_lon = delta_pred[0, 1] + subpoint[1]

            lat_err = abs(pred_lat - true_lat) * 111.0
            lon_err = abs(pred_lon - true_lon) * 111.0 * np.cos(np.deg2rad(true_lat))
            errors.append(np.sqrt(lat_err**2 + lon_err**2))

        errors = np.array(errors)
        print(f"  Evaluated on {n_test} random samples")
        print(f"  PNT error — RMS:     {np.sqrt(np.mean(errors**2)):7.1f} km")
        print(f"  PNT error — median:  {np.median(errors):7.1f} km")
        print(f"  PNT error — 90th %:  {np.percentile(errors, 90):7.1f} km")
        print(f"  ✓ PNT estimation complete")
    else:
        print(f"    PNT models not found")

    # ============================================================
    # Phase 2: Topology — evaluate over 10 samples
    # ============================================================
    print_step(2, "Topology Prediction (future ISL link states)")

    topo_path = Path("models/topology/topology_best.pt")
    if topo_path.exists():
        constellation = create_default_constellation()
        wrapper = TopologyWrapper(device="cpu")
        wrapper.load(str(topo_path))

        n_test = 10
        accs = []
        f1s = []
        for i in range(n_test):
            X_t, y_t = generate_topology_sample(constellation, t_start=float(i) * 100.0)
            X_b = X_t[np.newaxis, :, :]
            y_b = y_t[np.newaxis, :, :]
            y_pred = wrapper.predict(X_b)
            accs.append(wrapper.accuracy(y_b, y_pred))
            f1s.append(wrapper.f1_score(y_b, y_pred))

        print(f"  Evaluated on {n_test} random samples")
        print(f"  Topology prediction accuracy:  {np.mean(accs):.4f}")
        print(f"  Topology prediction F1:        {np.mean(f1s):.4f}")
        print(f"  ✓ Topology prediction complete")
    else:
        print(f"    Topology model not found")

    # ============================================================
    # Phase 3: Routing — evaluate over 20 episodes
    # ============================================================
    print_step(3, "DRL Routing (packet forwarding through constellation)")

    router_path = Path("models/router/dqn_v2_best.pt")
    if router_path.exists():
        agent = DQNAgent(obs_dim=35, action_dim=6, device="cpu")
        agent.load(str(router_path))

        n_episodes = 20
        delivered_count = 0
        hop_counts = []

        for i in range(n_episodes):
            env = RoutingEnvV2(seed=SEED + i)
            obs, info = env.reset()
            hops = 0
            reason = 'unknown'

            while hops < 20:
                mask = env.action_masks()
                action = agent.select_action(obs, mask, greedy=True)
                obs, reward, term, trunc, info = env.step(action)
                hops += 1
                if term or trunc:
                    reason = info.get('reason', 'unknown')
                    break

            if reason == 'delivered':
                delivered_count += 1
                hop_counts.append(hops)

        print(f"  Evaluated on {n_episodes} random routing tasks")
        print(f"  Delivered:      {delivered_count}/{n_episodes}")
        if hop_counts:
            print(f"  Mean hops:      {np.mean(hop_counts):.2f}")
        print(f"  ✓ DRL routing complete")
    else:
        print(f"    Router model not found")

    # ============================================================
    # Summary
    # ============================================================
    print_header("Summary")
    print(f"  Total time:  {time.time() - t_total:.2f} s")
    print()
    print("=" * 72)
    print("  END-TO-END PIPELINE COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    main()
