"""
DRL routing environment v3: multi-flow packet streaming.

Features:
- Multiple concurrent packets
- Per-satellite queues with overflow drops
- Random ISL failures
- Trapped-packet detection
- Continuous routing: current packet stays selected until delivered/dropped
- Observation clipped to [-1, 1]
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
TIME_STEP_S = 3.0

INITIAL_PACKETS = 3
PACKET_ARRIVAL_RATE = 0.1
MAX_QUEUE = 8
MAX_EPISODE_STEPS = 200
PACKET_TTL = 200

LINK_FAILURE_PROB = 0.05
LINK_RECOVERY_PROB = 0.03

# Scaled rewards (per-step reward clipped to [-1, 1])
REWARD_DELIVERED = 1.0
PENALTY_DROPPED = -0.5
PENALTY_QUEUE = -0.02
PENALTY_ACTIVE = -0.001


# ============================================================
# PACKET
# ============================================================

class Packet:
    """A packet being routed through the constellation."""

    def __init__(self, pkt_id: int, src: int, dst: int, birth_step: int):
        self.id = pkt_id
        self.src = src
        self.dst = dst
        self.current_sat = src
        self.birth_step = birth_step
        self.age = 0
        self.alive = True
        self.last_step_routed = -1
        self.stuck_counter = 0


# ============================================================
# ENVIRONMENT
# ============================================================

class RoutingEnvV3(gym.Env):
    """Multi-flow routing environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()

        self.constellation = create_default_constellation()
        self.num_sats = len(self.constellation)

        self.obs_dim = 46
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_NEIGHBORS)

        self.time_s = 0.0
        self.done = False
        self.step_count = 0

        self.packets: List[Packet] = []
        self.packet_id_counter = 0

        self.queues: List[List[Packet]] = [[] for _ in range(self.num_sats)]

        self.link_state: np.ndarray = np.ones(
            (self.num_sats, self.num_sats), dtype=np.float32
        )

        self.current_packet: Optional[Packet] = None

        self.neighbor_ids: List[int] = []
        self.neighbor_valid = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_loads = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_dist = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_queue = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_link_quality = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

        self.total_delivered = 0
        self.total_dropped = 0

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.time_s = float(self.rng.uniform(0, 3600))
        self.done = False
        self.step_count = 0
        self.packets = []
        self.packet_id_counter = 0
        self.queues = [[] for _ in range(self.num_sats)]
        self.total_delivered = 0
        self.total_dropped = 0
        self.current_packet = None

        self.link_state = np.ones(
            (self.num_sats, self.num_sats), dtype=np.float32
        )
        np.fill_diagonal(self.link_state, 0.0)

        fail_mask = (
            self.rng.random((self.num_sats, self.num_sats)) < LINK_FAILURE_PROB
        )
        self.link_state[fail_mask] = 0.0
        self.link_state = (
            (self.link_state + self.link_state.T) > 1.5
        ).astype(np.float32)
        np.fill_diagonal(self.link_state, 0.0)

        for _ in range(INITIAL_PACKETS):
            self._spawn_packet()

        self._select_next_packet()
        self._update_neighbors()

        obs = self._get_obs()
        info = {
            'active_packets': len([p for p in self.packets if p.alive]),
            'queues_total': sum(len(q) for q in self.queues),
        }
        return obs, info

    # --------------------------------------------------------
    # PACKET MANAGEMENT
    # --------------------------------------------------------

    def _spawn_packet(self) -> None:
        src = int(self.rng.integers(0, self.num_sats))
        dst = int(self.rng.integers(0, self.num_sats))
        while dst == src:
            dst = int(self.rng.integers(0, self.num_sats))

        pkt = Packet(self.packet_id_counter, src, dst, self.step_count)
        self.packet_id_counter += 1

        if len(self.queues[src]) < MAX_QUEUE:
            self.queues[src].append(pkt)
            self.packets.append(pkt)
        else:
            self.total_dropped += 1

    def _select_next_packet(self) -> None:
        positions = self.constellation.get_positions(self.time_s)
        adj_geom = compute_adjacency_matrix(positions)
        adj_active = adj_geom & (self.link_state > 0.5)

        for pkt in self.packets:
            if not pkt.alive:
                continue
            n_neighbors = int(adj_active[pkt.current_sat].sum())
            if n_neighbors == 0:
                pkt.stuck_counter += 1
                if pkt.stuck_counter >= 3:
                    pkt.alive = False
                    self.total_dropped += 1
                    if pkt in self.queues[pkt.current_sat]:
                        self.queues[pkt.current_sat].remove(pkt)
            else:
                pkt.stuck_counter = 0

        if self.current_packet is not None and self.current_packet.alive:
            return

        candidates = []
        for sat_id, q in enumerate(self.queues):
            for p in q:
                if p.alive:
                    candidates.append(p)

        if not candidates:
            self._spawn_packet()
            candidates = [p for q in self.queues for p in q if p.alive]

        if not candidates:
            self.current_packet = None
            return

        candidates.sort(key=lambda p: p.age, reverse=True)
        self.current_packet = candidates[0]

    def _advance_current_packet(self, next_sat: int) -> str:
        pkt = self.current_packet
        if pkt is None:
            return 'in_flight'

        cur_sat = pkt.current_sat
        if pkt in self.queues[cur_sat]:
            self.queues[cur_sat].remove(pkt)

        if len(self.queues[next_sat]) >= MAX_QUEUE:
            pkt.alive = False
            return 'dropped'

        if next_sat == pkt.dst:
            pkt.alive = False
            return 'delivered'

        pkt.current_sat = next_sat
        self.queues[next_sat].append(pkt)
        return 'in_flight'

    # --------------------------------------------------------
    # DYNAMICS
    # --------------------------------------------------------

    def _evolve_links(self) -> None:
        N = self.num_sats
        positions = self.constellation.get_positions(self.time_s)
        adj_geom = compute_adjacency_matrix(positions)

        fail_mask = (
            (self.rng.random((N, N)) < LINK_FAILURE_PROB)
            & (self.link_state > 0.5)
            & adj_geom
        )
        self.link_state[fail_mask] = 0.0

        recover_mask = (
            (self.rng.random((N, N)) < LINK_RECOVERY_PROB)
            & (self.link_state < 0.5)
            & adj_geom
        )
        self.link_state[recover_mask] = 1.0

        self.link_state = (
            (self.link_state + self.link_state.T) > 1.5
        ).astype(np.float32)
        np.fill_diagonal(self.link_state, 0.0)

    def _age_packets(self) -> None:
        for pkt in self.packets:
            if pkt.alive:
                pkt.age += 1
                if pkt.age > PACKET_TTL:
                    pkt.alive = False
                    self.total_dropped += 1
                    if pkt in self.queues[pkt.current_sat]:
                        self.queues[pkt.current_sat].remove(pkt)

    def _arrive_packets(self) -> None:
        n_new = self.rng.poisson(PACKET_ARRIVAL_RATE)
        for _ in range(n_new):
            self._spawn_packet()

    # --------------------------------------------------------
    # NEIGHBORS
    # --------------------------------------------------------

    def _update_neighbors(self) -> None:
        if self.current_packet is None:
            self.neighbor_ids = []
            self.neighbor_valid = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
            return

        cur = self.current_packet.current_sat
        positions = self.constellation.get_positions(self.time_s)
        adj_geom = compute_adjacency_matrix(positions)
        adj = adj_geom & (self.link_state > 0.5)

        neighbors = np.where(adj[cur])[0].tolist()

        if len(neighbors) > MAX_NEIGHBORS:
            cur_pos = positions[cur]
            dists = [
                (n, np.linalg.norm(positions[n] - cur_pos)) for n in neighbors
            ]
            dists.sort(key=lambda x: x[1])
            neighbors = [n for n, _ in dists[:MAX_NEIGHBORS]]

        self.neighbor_ids = neighbors
        self.neighbor_valid = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_loads = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_dist = np.ones(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_queue = np.ones(MAX_NEIGHBORS, dtype=np.float32)
        self.neighbor_link_quality = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

        dest = self.current_packet.dst
        dest_pos = positions[dest]
        cur_pos = positions[cur]
        cur_to_dest = np.linalg.norm(cur_pos - dest_pos) + 1e-6

        for i, n_id in enumerate(neighbors):
            if i >= MAX_NEIGHBORS:
                break
            self.neighbor_valid[i] = 1.0
            n_degree = int(adj[n_id].sum())
            self.neighbor_loads[i] = min(n_degree / 6.0, 1.0)
            self.neighbor_queue[i] = min(
                len(self.queues[n_id]) / MAX_QUEUE, 1.0
            )
            n_pos = positions[n_id]
            dist_via_n = np.linalg.norm(n_pos - dest_pos)
            ratio = min(dist_via_n / cur_to_dest, 2.0)
            self.neighbor_dist[i] = ratio - 1.0
            self.neighbor_link_quality[i] = 1.0

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.current_packet is None:
            return obs

        pkt = self.current_packet
        positions = self.constellation.get_positions(self.time_s)
        cur_pos = positions[pkt.current_sat]
        dest_pos = positions[pkt.dst]
        delta = dest_pos - cur_pos
        dn = delta / (np.linalg.norm(delta) + 1e-8)

        obs[0] = 0.0
        obs[1:4] = dn[:3]
        obs[4] = min(pkt.age / PACKET_TTL, 1.0)
        obs[5] = 0.0
        obs[6] = min(len(self.queues[pkt.current_sat]) / MAX_QUEUE, 1.0)
        obs[7:13] = self.neighbor_loads
        obs[13:19] = self.neighbor_dist
        obs[19:25] = self.neighbor_valid
        obs[25:31] = self.neighbor_link_quality
        obs[31:37] = self.neighbor_queue

        active_count = len([p for p in self.packets if p.alive])
        mean_q = float(np.mean([len(q) for q in self.queues]))
        max_q = float(np.max([len(q) for q in self.queues]))
        obs[37] = min(active_count / 50.0, 1.0)
        obs[38] = min(mean_q / MAX_QUEUE, 1.0)
        obs[39] = min(max_q / MAX_QUEUE, 1.0)

        adj_geom = compute_adjacency_matrix(positions)
        adj = adj_geom & (self.link_state > 0.5)
        obs[40] = min(int(adj[pkt.dst].sum()) / 6.0, 1.0)
        obs[41] = min(len(self.queues[pkt.dst]) / MAX_QUEUE, 1.0)
        obs[42] = min(np.linalg.norm(cur_pos - dest_pos) / 10000.0, 1.0)
        obs[43] = 0.0
        obs[44] = 0.0
        obs[45] = 0.0

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        self.step_count += 1
        self.time_s += TIME_STEP_S

        self._evolve_links()

        reward = 0.0
        event = 'none'

        if self.current_packet is None:
            event = 'no_packet'
        elif action < 0 or action >= MAX_NEIGHBORS or self.neighbor_valid[action] < 0.5:
            reward -= 0.1
            event = 'invalid'
        else:
            next_sat = self.neighbor_ids[action]
            result = self._advance_current_packet(next_sat)

            if result == 'delivered':
                reward += REWARD_DELIVERED
                self.total_delivered += 1
                event = 'delivered'
            elif result == 'dropped':
                reward += PENALTY_DROPPED
                self.total_dropped += 1
                event = 'dropped'
            else:
                event = 'in_flight'

        self._age_packets()
        self._arrive_packets()

        mean_q = float(np.mean([len(q) for q in self.queues]))
        reward += PENALTY_QUEUE * mean_q

        active_count = len([p for p in self.packets if p.alive])
        reward += PENALTY_ACTIVE * active_count

        self._select_next_packet()
        self._update_neighbors()

        truncated = False
        terminated = False
        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True
            self.done = True

        reward = float(np.clip(reward, -1.0, 1.0))

        obs = self._get_obs()
        info = {
            'event': event,
            'delivered_total': self.total_delivered,
            'dropped_total': self.total_dropped,
            'active_packets': active_count,
            'mean_queue': mean_q,
        }
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------
    # MASKING & RENDER
    # --------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        return self.neighbor_valid > 0.5

    def render(self):
        print(
            f"t={self.time_s:.1f} | step={self.step_count}/{MAX_EPISODE_STEPS} | "
            f"active={len([p for p in self.packets if p.alive])} | "
            f"delivered={self.total_delivered} | dropped={self.total_dropped}"
        )
