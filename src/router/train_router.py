"""
Training pipeline for DRL router.

Usage:
    python3 -m src.router.train_router
"""

import time
import numpy as np
from pathlib import Path

from .env import RoutingEnv
from .dqn_agent import DQNAgent


# ============================================================
# CONFIG
# ============================================================

N_EPISODES = 2000 
EVAL_EVERY = 100
EVAL_EPISODES = 100
SEED = 42
SAVE_PATH = "models/router/dqn_best.pt"

# Warmup: how many env steps before we start training the network
WARMUP_STEPS = 1000

# How often we do a gradient update
TRAIN_EVERY = 1

# How often we sync target network
TARGET_UPDATE = 500


# ============================================================
# HELPERS
# ============================================================

def run_episode(env, agent, train=True):
    """Run one episode, optionally storing transitions and training."""
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    losses = []

    while True:
        mask = env.action_masks()
        action = agent.select_action(obs, mask, greedy=not train)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_mask = env.action_masks()

        if train:
            agent.store(obs, action, reward, next_obs, float(done), mask, next_mask)
            if agent.total_steps >= WARMUP_STEPS and agent.total_steps % TRAIN_EVERY == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        obs = next_obs
        total_reward += reward
        steps += 1

        if done:
            break

    return {
        'reward': total_reward,
        'steps': steps,
        'delivered': info.get('reason') == 'delivered',
        'reason': info.get('reason', 'unknown'),
        'mean_loss': float(np.mean(losses)) if losses else 0.0,
    }


def evaluate(env, agent, n_episodes=EVAL_EPISODES):
    """Run N evaluation episodes with greedy policy."""
    results = [run_episode(env, agent, train=False) for _ in range(n_episodes)]
    delivery = sum(r['delivered'] for r in results) / n_episodes
    mean_reward = float(np.mean([r['reward'] for r in results]))
    mean_hops = float(np.mean([r['steps'] for r in results]))
    return {
        'delivery': delivery,
        'mean_reward': mean_reward,
        'mean_hops': mean_hops,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DRL Router Training")
    print("=" * 60)

    np.random.seed(SEED)
    import random as _random
    _random.seed(SEED)

    env = RoutingEnv(seed=SEED)
    agent = DQNAgent(
        obs_dim=21,
        action_dim=6,
        lr=1e-3,
        gamma=0.95,
        buffer_capacity=50000,
        batch_size=64,
        target_update=TARGET_UPDATE,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20000,
        device="cpu",
    )

    print(f"\nAgent config:")
    print(f"  Parameters: {sum(p.numel() for p in agent.q_net.parameters()):,}")
    print(f"  Model size: {agent.model_size_kb():.2f} kB")
    print(f"  Buffer:     {agent.replay_buffer.buffer.maxlen}")
    print(f"  Batch:      {agent.batch_size}")
    print(f"  Warmup:     {WARMUP_STEPS} steps")

    print(f"\nTraining {N_EPISODES} episodes (eval every {EVAL_EVERY})...")
    print("-" * 60)

    Path("models/router").mkdir(parents=True, exist_ok=True)

    best_delivery = 0.0
    train_start = time.time()

    for ep in range(1, N_EPISODES + 1):
        result = run_episode(env, agent, train=True)

        if ep % 50 == 0:
            print(f"  Ep {ep:5d} | "
                  f"reward={result['reward']:7.3f} | "
                  f"steps={result['steps']:3d} | "
                  f"delivered={result['delivered']} | "
                  f"eps={agent.epsilon:.3f} | "
                  f"loss={result['mean_loss']:.4f}")

        if ep % EVAL_EVERY == 0:
            eval_result = evaluate(env, agent, n_episodes=EVAL_EPISODES)
            print(f"\n  === EVAL at episode {ep} ===")
            print(f"    Delivery rate:  {eval_result['delivery']:.3f}")
            print(f"    Mean reward:    {eval_result['mean_reward']:.3f}")
            print(f"    Mean hops:      {eval_result['mean_hops']:.2f}")
            print(f"    Epsilon:        {agent.epsilon:.3f}")
            print()

            if eval_result['delivery'] > best_delivery:
                best_delivery = eval_result['delivery']
                agent.save(SAVE_PATH)
                print(f"    ✅ New best! Saved to {SAVE_PATH}")

    train_time = time.time() - train_start
    print("-" * 60)
    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"Best eval delivery rate: {best_delivery:.3f}")
    print(f"Model saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()

