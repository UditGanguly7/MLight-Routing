"""
Random policy baseline for the routing environment.

Runs N episodes with random valid actions to establish
a performance floor before DRL training.

Usage:
    python3 -m src.router.random_baseline
"""

import numpy as np
from src.router.env import RoutingEnv


def run_random_episode(env, rng):
    """Run one episode with random valid actions."""
    obs, info = env.reset(seed=int(rng.integers(0, 1000000)))
    total_reward = 0.0
    steps = 0

    while True:
        # Get valid actions
        valid = [i for i in range(6) if env.neighbor_valid[i] > 0.5]
        if not valid:
            break
        action = valid[int(rng.integers(0, len(valid)))]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return {
        'reward': total_reward,
        'steps': steps,
        'reason': info.get('reason', 'unknown'),
        'delivered': info.get('reason') == 'delivered',
    }


def main():
    print("=" * 60)
    print("Random Policy Baseline")
    print("=" * 60)

    env = RoutingEnv(seed=0)
    rng = np.random.default_rng(42)

    n_episodes = 500
    results = []

    for i in range(n_episodes):
        if (i + 1) % 100 == 0:
            print(f"  Episode {i+1}/{n_episodes}...")
        results.append(run_random_episode(env, rng))

    # Aggregate
    delivered = sum(1 for r in results if r['delivered'])
    reasons = {}
    for r in results:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1

    rewards = [r['reward'] for r in results]
    steps = [r['steps'] for r in results]

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

