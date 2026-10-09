"""
Quick demo of the constellation simulator.

Run: python3 src/simulation/demo.py
"""

import numpy as np
from .constellation import create_default_constellation
from .link import compute_adjacency_matrix, count_isl_links
from .channel import compute_link_metrics, compute_doppler_hz
from . import constants as C


def main():
    print("=" * 60)
    print("MLight-Routing: Constellation Simulator Demo")
    print("=" * 60)

    # Build constellation
    const = create_default_constellation()
    print(f"\n{const}")

    # Snapshot at t = 0
    t = 0.0
    positions = const.get_positions(t)
    velocities = const.get_velocities(t)

    # Adjacency matrix
    adj = compute_adjacency_matrix(positions)
    num_links = count_isl_links(adj)

    degrees = np.sum(adj, axis=1)
    print(f"\nAt t = {t:.1f} s:")
    print(f"  Active ISL links:  {num_links}")
    print(f"  Mean satellite degree: {np.mean(degrees):.2f}")
    print(f"  Min degree:        {int(np.min(degrees))}")
    print(f"  Max degree:        {int(np.max(degrees))}")
    print(f"  Isolated sats:     {int(np.sum(degrees == 0))}")

    # Test channel metrics on a few links
    print(f"\nChannel metrics (sample links):")
    for i, j in [(0, 5), (0, 10), (0, 15)]:
        if adj[i, j]:
            m = compute_link_metrics(positions[i], velocities[i], positions[j], velocities[j])
            print(f"  Sat {i:2d} <-> Sat {j:2d}: "
                  f"d={m['distance_km']:7.1f} km, "
                  f"τ={m['delay_ms']:5.2f} ms, "
                  f"f_d={m['doppler_hz']:8.1f} Hz, "
                  f"SNR={m['snr_db']:6.2f} dB")
        else:
            print(f"  Sat {i:2d} <-> Sat {j:2d}: no link")

    # Time evolution
    print(f"\nTopology evolution over 60 seconds:")
    for t in [0, 15, 30, 45, 60]:
        pos = const.get_positions(t)
        adj = compute_adjacency_matrix(pos)
        n_links = count_isl_links(adj)
        print(f"  t={t:3.0f}s: {n_links} links")

    print("\n" + "=" * 60)
    print("Simulation ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
