"""
Lightweight DQN agent for MLight-Routing.

Supports action masking (invalid next-hops masked out during action selection
and Q-value updates). Handles edge case where all actions are invalid.

Architecture:
    Input(obs_dim) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(action_dim)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from collections import deque
import random


# ============================================================
# Q-NETWORK
# ============================================================

class QNetwork(nn.Module):
    """Small MLP for Q-value estimation with action masking."""

    def __init__(self, obs_dim: int = 46, action_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, action_mask, next_mask):
        self.buffer.append(
            (state, action, reward, next_state, done, action_mask, next_mask)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(masks, dtype=np.bool_),
            np.array(next_masks, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:
    """
    DQN agent with action masking and safe target computation.

    Key bug fix: when all actions in a next-state are invalid,
    the max over Q-values must not be -1e9 (which would explode
    the squared loss). We use -100 as the mask value and zero out
    the target when no valid actions exist.
    """

    def __init__(
        self,
        obs_dim: int = 46,
        action_dim: int = 6,
        lr: float = 3e-4,
        gamma: float = 0.95,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 30000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    @property
    def epsilon(self) -> float:
        frac = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            return 0

        if not greedy and random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))

        with torch.no_grad():
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        q_values[~action_mask] = -np.inf
        return int(np.argmax(q_values))

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    def store(self, state, action, reward, next_state, done, action_mask, next_mask):
        self.replay_buffer.push(
            state, action, reward, next_state, done, action_mask, next_mask
        )
        self.total_steps += 1

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            masks,
            next_masks,
        ) = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        next_masks_t = torch.tensor(next_masks, device=self.device)

        # Current Q-values
        q_values = self.q_net(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target computation with safe masking
        with torch.no_grad():
            next_q = self.target_net(next_states_t)

            # Mask invalid actions with a moderate value
            next_q[~next_masks_t] = -100.0

            # Max over actions
            max_next_q = next_q.max(dim=1).values

            # If a state has no valid actions at all, use 0 as target
            has_valid = next_masks_t.any(dim=1)
            max_next_q = torch.where(
                has_valid,
                max_next_q,
                torch.zeros_like(max_next_q),
            )

            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        # Clip targets to reasonable range
        target_q = torch.clamp(target_q, -100.0, 100.0)

        # Loss
        loss = self.loss_fn(q_selected, target_q)

        # Backward with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Sync target network
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
        print(f"DQN saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt['total_steps']
        print(f"DQN loaded: {path}")

    def model_size_kb(self) -> float:
        total_params = sum(p.numel() for p in self.q_net.parameters())
        return total_params * 4 / 1024.0
