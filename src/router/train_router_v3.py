"""
Training pipeline for DRL router v3 (multi-flow, congestion, failures).

Usage:
    python3 -m src.router.train_router_v3
"""

import time
import numpy as np
from pathlib import Path

from .env_v3 import RoutingEnvV3
from .dqn_agent import DQNAgent


# ============================================================
# CONFIG
# ============================================================

N_EPISODES = 2000
EVAL_EVERY = 200
EVAL_EPISODES = 50
SEED = 42
SAVE_PATH = "models/router/dqn_v3_best.pt"

WARMUP_STEPS = 2000
TARGET_UPDATE = 500


# ============================================================
# HELPERS
# ============================================================

def run_episode(env, agent, train=True):
    obs, info = env.reset()
    total_reward = 0.0
    losses = []

    while True:
        mask = env.action_masks()
        action = agent.select_action(obs, mask, greedy=not train)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = env.action_masks()

        if train:
            agent.store(
                obs, action, reward, next_obs, float(done), mask, next_mask
            )
            if agent.total_steps >= WARMUP_STEPS:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        obs = next_obs
        total_reward += reward

        if done:
            break

    return {
        'reward': total_reward,
        'delivered': info.get('delivered_total', 0),
        'dropped': info.get('dropped_total', 0),
        'mean_loss': float(np.mean(losses)) if losses else 0.0,
    }


def evaluate(env, agent, n_episodes=EVAL_EPISODES):
    results = [run_episode(env, agent, train=False) for _ in range(n_episodes)]
    return {
        'delivery': float(np.mean([r['delivered'] for r in results])),
        'drops': float(np.mean([r['dropped'] for r in results])),
        'reward': float(np.mean([r['reward'] for r in results])),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DRL Router Training v3 (Multi-Flow)")
    print("=" * 60)

    np.random.seed(SEED)
    import random as _random
    _random.seed(SEED)

    env = RoutingEnvV3(seed=SEED)
    agent = DQNAgent(
        obs_dim=46,
        action_dim=6,
        lr=3e-4,
        gamma=0.95,
        buffer_capacity=100000,
        batch_size=64,
        target_update=TARGET_UPDATE,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=30000,
        device="cpu",
    )

    print(f"\nAgent config:")
    print(f"  Parameters: {sum(p.numel() for p in agent.q_net.parameters()):,}")
    print(f"  Model size: {agent.model_size_kb():.2f} kB")

    print(f"\nTraining {N_EPISODES} episodes (eval every {EVAL_EVERY})...")
    print("-" * 60)

    Path("models/router").mkdir(parents=True, exist_ok=True)

    best_delivered = 0.0
    train_start = time.time()

    for ep in range(1, N_EPISODES + 1):
        result = run_episode(env, agent, train=True)

        if ep % 100 == 0:
            print(
                f"  Ep {ep:5d} | reward={result['reward']:8.3f} | "
                f"delivered={result['delivered']:3d} | "
                f"dropped={result['dropped']:3d} | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={result['mean_loss']:.4f}"
            )

        if ep % EVAL_EVERY == 0:
            eval_result = evaluate(env, agent, n_episodes=EVAL_EPISODES)
            print(f"\n  === EVAL at episode {ep} ===")
            print(f"    Mean delivered:  {eval_result['delivery']:.2f}")
            print(f"    Mean dropped:    {eval_result['drops']:.2f}")
            print(f"    Mean reward:     {eval_result['reward']:.3f}")
            print()

            if eval_result['delivery'] > best_delivered:
                best_delivered = eval_result['delivery']
                agent.save(SAVE_PATH)
                print(f"    ✅ New best! ({best_delivered:.2f} delivered)")

    train_time = time.time() - train_start
    print("-" * 60)
    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"Best mean delivered: {best_delivered:.2f}")
    print(f"Model saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
