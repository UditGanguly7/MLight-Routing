"""
Heuristic greedy-distance baseline for routing.

Strategy: at each step, choose the neighbor closest to destination
(in Euclidean ECEF distance).

Usage:
    python3 -m src.router.heuristic_baseline
"""

import numpy as np
from src.router.env import RoutingEnv
from src.simulation.link import compute_adjacency_matrix


def run_greedy_episode(env):
    """Run one episode using greedy-distance heuristic."""
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        # Get current positions
        positions = env.constellation.get_positions(env.time_s)
        cur_pos = positions[env.current_sat]
        dest_pos = positions[env.dest_sat]

        # Get valid neighbors and their distances to destination
        valid = [i for i in range(6) if env.neighbor_valid[i] > 0.5]
        if not valid:
            break

        neighbor_dists = []
        for i in valid:
            n_id = env.neighbor_ids[i]
            n_pos = positions[n_id]
            dist_to_dest = np.linalg.norm(n_pos - dest_pos)
            neighbor_dists.append((i, dist_to_dest))

        # Choose neighbor closest to destination
        neighbor_dists.sort(key=lambda x: x[1])
        action = neighbor_dists[0][0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {
        'reward': total_reward,
        'steps': steps,
        'delivered': info.get('reason') == 'delivered',
        'reason': info.get('reason', 'unknown'),
    }


def main():
    print("=" * 60)
    print("Greedy-Distance Heuristic Baseline")
    print("=" * 60)

    env = RoutingEnv(seed=0)

    n_episodes = 500
    results = []

    for i in range(n_episodes):
        if (i + 1) % 100 == 0:
            print(f"  Episode {i+1}/{n_episodes}...")
        results.append(run_greedy_episode(env))

    # Aggregate
    delivered = sum(1 for r in results if r['delivered'])
    rewards = [r['reward'] for r in results]
    steps = [r['steps'] for r in results]

    reasons = {}
    for r in results:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Episodes:            {n_episodes}")
    print(f"Delivery rate:       {delivered/n_episodes:.3f} ({delivered}/{n_episodes})")
    print(f"Mean reward:         {np.mean(rewards):.3f}")
    print(f"Mean hops:           {np.mean(steps):.2f}")
    print()
    print("Termination reasons:")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason:15s}: {count:4d} ({count/n_episodes*100:.1f}%)")


if __name__ == "__main__":
    main()
