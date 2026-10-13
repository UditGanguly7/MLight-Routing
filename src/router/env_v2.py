"""
DRL routing environment v2: with congestion and link failures.

Extends the base environment with:
- Per-satellite queue tracking (drop packets when full)
- Random ISL failures (per-link reliability)
- Larger observation space (~35 dims)
- More realistic reward signal

The agent must learn to:
- Balance load across satellites (avoid congestion)
- Route around failed links
- Minimize hop count
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List

from ..simulation.constellation import create_default_constellation
from ..simulation.link import compute_adjacency_matrix
from ..simulation import constants as C


# ============================================================
# CONFIG
# ============================================================

MAX_NEIGHBORS = 6
MAX_HOPS = 15
TIME_STEP_S = 3.0

# Congestion parameters
MAX_QUEUE = 5                    # max packets per satellite
PACKET_ARRIVAL_RATE = 0.3        # probability of new packet arriving at source per step

# Link failure parameters
LINK_FAILURE_PROB = 0.02         # 2% chance a link is down at any moment
LINK_RECOVERY_PROB = 0.10        # 10% chance a down link recovers


# ============================================================
# ENVIRONMENT
# ============================================================

class RoutingEnvV2(gym.Env):
    """
    Routing environment with congestion and link failures.

    Observation (35 dims):
        [0]     hops_so_far_normalized
        [1-2]   dest_direction (x, y)
        [3-8]   per-neighbor load (queue fill fraction at each neighbor)
        [9-14]  per-neighbor dist_to_dest (normalized)
        [15-20] per-neighbor valid_mask
        [21-26] per-neighbor link_quality (0=down, 1=up)
        [27-32] per-neighbor queue length (normalized)
        [33]    own queue length normalized
        [34]    mean network queue length normalized

    Action space: Discrete(6) — same as v1
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()

        self.constellation = create_default_constellation()
        self.num_sats = len(self.constellation)

        # 35-dim observation
        self.obs_dim = 35
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_NEIGHBORS)

        # Episode state
        self.current_sat: int = 0
        self.dest_sat: int = 0
        self.hops_taken: int = 0
        self.time_s: float = 0.0
        self.done: bool = False

        # Per-satellite queues (packets waiting)
        self.queues: np.ndarray = np.zeros(self.num_sats, dtype=np.float32)

        # Link state: N×N binary (1=up, 0=down)
        self.link_state: np.ndarray = np.ones(
            (self.num_sats, self.num_sats), dtype=np.float32
        )

        # Current neighbors
        self.neighbor_ids: List[int] = []
        self.neighbor_valid: np.ndarray = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_loads: np.ndarray = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_dist: np.ndarray = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_queue: np.ndarray = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_link_quality: np.ndarray = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Random source/destination
        self.current_sat = int(self.rng.integers(0, self.num_sats))
        self.dest_sat = int(self.rng.integers(0, self.num_sats))
        while self.dest_sat == self.current_sat:
            self.dest_sat = int(self.rng.integers(0, self.num_sats))

        self.hops_taken = 0
        self.time_s = float(self.rng.uniform(0, 3600))
        self.done = False

        # Reset queues to small random values
        self.queues = self.rng.uniform(0, 2, size=self.num_sats).astype(np.float32)

        # Reset link states (random initial failures)
        self.link_state = (self.rng.random((self.num_sats, self.num_sats)) > LINK_FAILURE_PROB).astype(np.float32)
        np.fill_diagonal(self.link_state, 0.0)
        self.link_state = np.triu(self.link_state, 1) + np.triu(self.link_state, 1).T

        # Update neighbors
        self._update_neighbors()

        obs = self._get_obs()
        info = {'current_sat': self.current_sat, 'dest_sat': self.dest_sat}
        return obs, info

    # --------------------------------------------------------
    # DYNAMICS: LINK FAILURES
    # --------------------------------------------------------

    def _evolve_links(self) -> None:
        """Random link failures and recoveries."""
        N = self.num_sats
        # Random failures on currently-up links
        fail_mask = (self.rng.random((N, N)) < LINK_FAILURE_PROB) & (self.link_state > 0.5)
        self.link_state[fail_mask] = 0.0

        # Random recoveries on currently-down links
        recover_mask = (self.rng.random((N, N)) < LINK_RECOVERY_PROB) & (self.link_state < 0.5)
        self.link_state[recover_mask] = 1.0

        # Keep symmetric and no self-loops
        self.link_state = ((self.link_state + self.link_state.T) > 1.0).astype(np.float32)
        np.fill_diagonal(self.link_state, 0.0)

    # --------------------------------------------------------
    # NEIGHBORS
    # --------------------------------------------------------

    def _update_neighbors(self) -> None:
        """Find current satellite's neighbors (active ISLs)."""
        positions = self.constellation.get_positions(self.time_s)
        adj_geom = compute_adjacency_matrix(positions)

        # Active links = geometric adjacency AND not failed
        adj = adj_geom & (self.link_state > 0.5)

        neighbors = np.where(adj[self.current_sat])[0].tolist()

        # Cap at MAX_NEIGHBORS by distance
        if len(neighbors) > MAX_NEIGHBORS:
            cur_pos = positions[self.current_sat]
            dists = [(n, np.linalg.norm(positions[n] - cur_pos)) for n in neighbors]
            dists.sort(key=lambda x: x[1])
            neighbors = [n for n, _ in dists[:MAX_NEIGHBORS]]

        self.neighbor_ids = neighbors
        self.neighbor_valid = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_loads = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_dist = np.ones(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_queue = np.ones(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_link_quality = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

        cur_pos = positions[self.current_sat]
        dest_pos = positions[self.dest_sat]
        cur_to_dest = np.linalg.norm(cur_pos - dest_pos)

        for i, n_id in enumerate(neighbors):
            if i >= MAX_NEIGHBORS:
                break
            self.neighbor_valid[i] = 1.0

            # Load = number of active ISLs at neighbor (proxy for traffic)
            n_degree = int(adj[n_id].sum())
            self.neighbor_loads[i] = min(n_degree / 6.0, 1.0)

            # Queue length normalized
            self.neighbor_queue[i] = min(self.queues[n_id] / MAX_QUEUE, 1.0)

            # Distance to destination via this neighbor
            n_pos = positions[n_id]
            dist_via_n = np.linalg.norm(n_pos - dest_pos)
            self.neighbor_dist[i] = min(dist_via_n / max(cur_to_dest, 1.0), 2.0) - 1.0

            # Link quality (always 1 for active neighbors, 0 for invalid)
            self.neighbor_link_quality[i] = 1.0 if self.link_state[self.current_sat, n_id] > 0.5 else 0.0

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        hops_norm = self.hops_taken / MAX_HOPS

        positions = self.constellation.get_positions(self.time_s)
        cur_pos = positions[self.current_sat]
        dest_pos = positions[self.dest_sat]
        delta = dest_pos - cur_pos
        delta_norm = delta / (np.linalg.norm(delta) + 1e-8)
        dest_x = delta_norm[0]
        dest_y = delta_norm[1]

        own_q = min(self.queues[self.current_sat] / MAX_QUEUE, 1.0)
        mean_q = min(float(self.queues.mean()) / MAX_QUEUE, 1.0)

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = hops_norm
        obs[1] = dest_x
        obs[2] = dest_y
        obs[3:9] = self.neighbor_loads
        obs[9:15] = self.neighbor_dist
        obs[15:21] = self.neighbor_valid
        obs[21:27] = self.neighbor_link_quality
        obs[27:33] = self.neighbor_queue
        obs[33] = own_q
        obs[34] = mean_q
        return obs

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        # Validate
        if action < 0 or action >= MAX_NEIGHBORS or self.neighbor_valid[action] < 0.5:
            self.done = True
            return self._get_obs(), -1.0, True, False, {'reason': 'invalid_action'}

        chosen_sat = self.neighbor_ids[action]
        self.hops_taken += 1
        self.time_s += TIME_STEP_S

        # Check delivery BEFORE moving
        if chosen_sat == self.dest_sat:
            self.done = True
            # Delivery reward: base + efficiency bonus - congestion penalty
            hop_penalty = 0.05 * self.hops_taken
            congestion_penalty = 0.3 * min(self.queues[chosen_sat] / MAX_QUEUE, 1.0)
            reward = 1.0 - hop_penalty - congestion_penalty
            obs = self._get_obs()
            return obs, reward, True, False, {
                'reason': 'delivered',
                'hops': self.hops_taken,
            }

        # Check hop limit
        if self.hops_taken >= MAX_HOPS:
            self.done = True
            return self._get_obs(), -1.0, True, False, {'reason': 'max_hops'}

        # CONGESTION CHECK: does the chosen satellite have room?
        if self.queues[chosen_sat] >= MAX_QUEUE:
            # Packet dropped
            self.done = True
            return self._get_obs(), -1.0, True, False, {
                'reason': 'dropped_congestion',
                'queue': float(self.queues[chosen_sat]),
            }

        # Move and add to queue
        self.queues[chosen_sat] += 1.0
        self.current_sat = chosen_sat

        # Simulate queue drain (packets leave at rate 0.3/step)
        self.queues = np.maximum(0.0, self.queues - 0.3)

        # Evolve link states (failures/recoveries)
        self._evolve_links()

        # Update neighbors at new location
        self._update_neighbors()

        # Dead end check
        if self.neighbor_valid.sum() < 0.5:
            self.done = True
            return self._get_obs(), -1.0, True, False, {'reason': 'dead_end'}

        # Intermediate reward
        # - Hop cost
        # - Congestion penalty based on chosen neighbor's queue
        hop_penalty = 0.05
        congestion_penalty = 0.3 * min(self.queues[chosen_sat] / MAX_QUEUE, 1.0)
        reward = -hop_penalty - congestion_penalty

        obs = self._get_obs()
        return obs, reward, False, False, {
            'hops': self.hops_taken,
            'current_sat': self.current_sat,
        }

    # --------------------------------------------------------
    # ACTION MASKING
    # --------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        return self.neighbor_valid > 0.5

    def render(self):
        print(f"t={self.time_s:.1f} | "
              f"cur=Sat{self.current_sat} | dest=Sat{self.dest_sat} | "
              f"hops={self.hops_taken} | "
              f"own_q={self.queues[self.current_sat]:.1f}/{MAX_QUEUE} | "
              f"valid={int(self.neighbor_valid.sum())}")

