"""
DRL routing environment for MLight-Routing.

Gym-compatible single-agent environment where the agent routes
a packet from a source satellite to a destination satellite
through a dynamic LEO constellation.

State space:
    - current_sat_id:  which satellite holds the packet
    - dest_sat_id:     where the packet wants to go
    - neighbors:       list of reachable next-hop satellites
    - neighbor_loads:  queue length at each neighbor (0-1 normalized)
    - neighbor_dist:   estimated distance to dest via each neighbor (0-1)
    - hops_so_far:     number of hops taken (0-1 normalized)
    - dest_direction:  azimuth to destination (unit circle)

Action space:
    Discrete(6) — choose one of up to 6 neighbors. Invalid actions
    are masked. If no valid actions, episode ends with failure.
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

MAX_NEIGHBORS = 6           # Max ISL degree we'll consider
MAX_HOPS = 15               # Episode ends if this many hops taken
TIME_STEP_S = 3.0           # Time advances this much per action


# ============================================================
# ENVIRONMENT
# ============================================================

class RoutingEnv(gym.Env):
    """
    Single-packet routing environment.

    The agent routes one packet from a random source to a random
    destination satellite. At each step, the agent selects one of
    the current satellite's ISL neighbors as next hop.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()

        self.constellation = create_default_constellation()
        self.num_sats = len(self.constellation)

        # Observation: 2 (src/dest one-hot is too big; use relative features)
        # We use a compact representation:
        #   [hops_so_far_normalized, dest_azimuth_x, dest_azimuth_y,
        #    neighbor_load_0..5, neighbor_dist_0..5, neighbor_valid_0..5]
        # = 1 + 2 + 6 + 6 + 6 = 21
        self.obs_dim = 1 + 2 + MAX_NEIGHBORS * 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action: choose neighbor index 0-5 (invalid masked)
        self.action_space = spaces.Discrete(MAX_NEIGHBORS)

        # Episode state
        self.current_sat: int = 0
        self.dest_sat: int = 0
        self.hops_taken: int = 0
        self.time_s: float = 0.0
        self.done: bool = False

        # Track neighbors and their metadata for action masking
        self.neighbor_ids: List[int] = []
        self.neighbor_loads: np.ndarray = np.zeros(MAX_NEIGHBORS)
        self.neighbor_dist: np.ndarray = np.zeros(MAX_NEIGHBORS)
        self.neighbor_valid: np.ndarray = np.zeros(MAX_NEIGHBORS)

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset episode with new random source/destination."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Random source and destination (must be different)
        self.current_sat = int(self.rng.integers(0, self.num_sats))
        self.dest_sat = int(self.rng.integers(0, self.num_sats))
        while self.dest_sat == self.current_sat:
            self.dest_sat = int(self.rng.integers(0, self.num_sats))

        self.hops_taken = 0
        self.time_s = float(self.rng.uniform(0, 3600))
        self.done = False

        # Refresh neighbors
        self._update_neighbors()

        obs = self._get_obs()
        info = {'current_sat': self.current_sat, 'dest_sat': self.dest_sat}
        return obs, info

    # --------------------------------------------------------
    # NEIGHBOR DISCOVERY
    # --------------------------------------------------------

    def _update_neighbors(self) -> None:
        """Find current satellite's active ISL neighbors."""
        positions = self.constellation.get_positions(self.time_s)
        adj = compute_adjacency_matrix(positions)

        # Neighbors are active links from current satellite
        neighbors = np.where(adj[self.current_sat])[0].tolist()

        # Cap at MAX_NEIGHBORS (take closest by distance)
        if len(neighbors) > MAX_NEIGHBORS:
            cur_pos = positions[self.current_sat]
            dists = [(n, np.linalg.norm(positions[n] - cur_pos)) for n in neighbors]
            dists.sort(key=lambda x: x[1])
            neighbors = [n for n, _ in dists[:MAX_NEIGHBORS]]

        self.neighbor_ids = neighbors

        # Fill arrays
        self.neighbor_valid = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_loads = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_dist = np.ones(MAX_NEIGHBORS, dtype=np.float32)

        cur_pos = positions[self.current_sat]
        dest_pos = positions[self.dest_sat]
        cur_to_dest = np.linalg.norm(cur_pos - dest_pos)

        for i, n_id in enumerate(neighbors):
            if i >= MAX_NEIGHBORS:
                break
            self.neighbor_valid[i] = 1.0

            # Queue load: proxy via number of incoming links (more links = more traffic)
            n_degree = int(adj[n_id].sum())
            self.neighbor_loads[i] = min(n_degree / 6.0, 1.0)

            # Distance to destination via this neighbor
            n_pos = positions[n_id]
            dist_via_n = np.linalg.norm(n_pos - dest_pos)
            # Normalize by current-to-dest distance
            self.neighbor_dist[i] = min(dist_via_n / max(cur_to_dest, 1.0), 2.0) - 1.0

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Hops taken (normalized)
        hops_norm = self.hops_taken / MAX_HOPS

        # Destination direction (unit vector in local frame)
        # Simplified: use ECEF delta direction, projected to 2D
        positions = self.constellation.get_positions(self.time_s)
        cur_pos = positions[self.current_sat]
        dest_pos = positions[self.dest_sat]
        delta = dest_pos - cur_pos
        delta_norm = delta / (np.linalg.norm(delta) + 1e-8)

        # 2D projection using arbitrary basis
        dest_x = delta_norm[0]
        dest_y = delta_norm[1]

        # Assemble
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = hops_norm
        obs[1] = dest_x
        obs[2] = dest_y
        obs[3:3 + MAX_NEIGHBORS] = self.neighbor_loads
        obs[3 + MAX_NEIGHBORS:3 + 2 * MAX_NEIGHBORS] = self.neighbor_dist
        obs[3 + 2 * MAX_NEIGHBORS:3 + 3 * MAX_NEIGHBORS] = self.neighbor_valid

        return obs

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one routing action.

        Args:
            action: index in [0, MAX_NEIGHBORS) selecting next-hop neighbor

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Validate action
        if action < 0 or action >= MAX_NEIGHBORS or self.neighbor_valid[action] < 0.5:
            # Invalid action — penalty and end
            self.done = True
            obs = self._get_obs()
            return obs, -1.0, True, False, {'reason': 'invalid_action'}

        # Take action: move to selected neighbor
        chosen_sat = self.neighbor_ids[action]
        self.hops_taken += 1

        # Advance time
        self.time_s += TIME_STEP_S

        # Check if we arrived
        if chosen_sat == self.dest_sat:
            self.done = True
            obs = self._get_obs()
            reward = 1.0 - 0.05 * self.hops_taken
            info = {
                'reason': 'delivered',
                'hops': self.hops_taken,
                'time_s': self.time_s,
            }
            return obs, reward, True, False, info

        # Check hop limit
        if self.hops_taken >= MAX_HOPS:
            self.done = True
            obs = self._get_obs()
            reward = -1.0
            info = {'reason': 'max_hops', 'hops': self.hops_taken}
            return obs, reward, True, False, info

        # Move to new satellite
        self.current_sat = chosen_sat

        # Update neighbors at new satellite
        self._update_neighbors()

        # Check if stuck (no valid neighbors)
        if self.neighbor_valid.sum() < 0.5:
            self.done = True
            obs = self._get_obs()
            return obs, -1.0, True, False, {'reason': 'dead_end'}

        # Compute reward for intermediate hop
        # Penalty: hop cost + load cost
        load_penalty = 0.1 * self.neighbor_loads[action]
        hop_penalty = 0.05
        reward = -hop_penalty - load_penalty

        obs = self._get_obs()
        info = {'hops': self.hops_taken, 'current_sat': self.current_sat}
        return obs, reward, False, False, info

    # --------------------------------------------------------
    # ACTION MASKING
    # --------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions (used by SB3)."""
        return self.neighbor_valid > 0.5

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    def render(self):
        """Simple text render."""
        print(f"Time: {self.time_s:.1f}s | "
              f"Current: Sat {self.current_sat} | "
              f"Dest: Sat {self.dest_sat} | "
              f"Hops: {self.hops_taken} | "
              f"Valid actions: {self.neighbor_valid.sum():.0f}")

