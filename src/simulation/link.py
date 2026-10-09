"""
Inter-satellite link (ISL) geometry.

Determines if two satellites can communicate based on:
    - Distance range (max ISL range)
    - Earth blockage (line of sight)

Also provides link quality metrics.
"""

import numpy as np
from typing import Tuple, Optional
from . import constants as C


# ============================================================
# RANGE CHECK
# ============================================================

def is_in_range(pos_a_km: np.ndarray, pos_b_km: np.ndarray) -> bool:
    """Check if two satellites are within max ISL range."""
    dist = np.linalg.norm(pos_a_km - pos_b_km)
    return dist <= C.MAX_ISL_RANGE_KM


def link_distance_km(pos_a_km: np.ndarray, pos_b_km: np.ndarray) -> float:
    """Euclidean distance between two satellites [km]."""
    return float(np.linalg.norm(pos_a_km - pos_b_km))


# ============================================================
# EARTH BLOCKAGE CHECK
# ============================================================

def is_line_of_sight_clear(
    pos_a_km: np.ndarray,
    pos_b_km: np.ndarray,
    earth_radius_km: float = C.EARTH_RADIUS_KM,
) -> bool:
    """
    Check if line of sight between two satellites is clear of Earth.

    Method: Find minimum distance from Earth's center to the line segment
    between A and B. If this minimum distance < Earth radius, Earth blocks.

    Args:
        pos_a_km, pos_b_km: (3,) positions in ECI [km]
        earth_radius_km: Earth radius [km]

    Returns:
        True if line of sight is clear
    """
    # Parametric line: P(s) = A + s * (B - A), s ∈ [0, 1]
    ab = pos_b_km - pos_a_km

    # Special case: A == B
    norm_ab_sq = np.dot(ab, ab)
    if norm_ab_sq < 1e-9:
        return np.linalg.norm(pos_a_km) > earth_radius_km

    # Find s that minimizes distance from origin to line
    s = -np.dot(pos_a_km, ab) / norm_ab_sq

    # Clamp to line segment
    s = np.clip(s, 0.0, 1.0)

    # Closest point on segment to Earth's center
    closest_point = pos_a_km + s * ab
    min_dist = np.linalg.norm(closest_point)

    return min_dist >= earth_radius_km


# ============================================================
# COMBINED LINK CHECK
# ============================================================

def is_link_available(
    pos_a_km: np.ndarray,
    pos_b_km: np.ndarray,
) -> Tuple[bool, Optional[float]]:
    """
    Check if an ISL between two satellites is available.

    Returns:
        (is_available, distance_km)
        distance_km is None if link unavailable
    """
    dist = link_distance_km(pos_a_km, pos_b_km)

    if dist > C.MAX_ISL_RANGE_KM:
        return False, None

    if not is_line_of_sight_clear(pos_a_km, pos_b_km):
        return False, None

    return True, dist


# ============================================================
# VISIBILITY MATRIX
# ============================================================

def compute_adjacency_matrix(
    positions_km: np.ndarray,
) -> np.ndarray:
    """
    Compute N×N adjacency matrix for a constellation snapshot.

    Args:
        positions_km: (N, 3) array of satellite positions

    Returns:
        adjacency: (N, N) boolean matrix
                   adjacency[i, j] = True if ISL between i and j is available
    """
    N = positions_km.shape[0]
    adjacency = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i + 1, N):
            available, _ = is_link_available(positions_km[i], positions_km[j])
            if available:
                adjacency[i, j] = True
                adjacency[j, i] = True

    return adjacency


def count_isl_links(adjacency: np.ndarray) -> int:
    """Count total bidirectional ISL links (each counted once)."""
    return int(np.sum(adjacency) // 2)


def link_statistics(
    positions_km: np.ndarray,
) -> dict:
    """Compute summary statistics on ISLs for a constellation snapshot."""
    adj = compute_adjacency_matrix(positions_km)
    N = positions_km.shape[0]

    num_links = count_isl_links(adj)
    degrees = np.sum(adj, axis=1)

    return {
        'num_satellites': N,
        'num_isl_links': num_links,
        'mean_degree': float(np.mean(degrees)),
        'min_degree': int(np.min(degrees)),
        'max_degree': int(np.max(degrees)),
        'isolated_sats': int(np.sum(degrees == 0)),
    }
