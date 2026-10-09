"""
LEO Walker-Delta constellation model.

Creates 66 satellites (Iridium-like) in 6 orbital planes.
Each satellite position/velocity computed via orbit.py.
"""

import numpy as np
from typing import List, Dict
from . import constants as C
from .orbit import Satellite, compute_mean_motion


class Constellation:
    """
    Walker-Delta constellation.

    Configuration:
        - num_planes: number of orbital planes
        - sats_per_plane: satellites per plane
        - inclination: orbital tilt
        - altitude: orbital altitude

    Total = num_planes × sats_per_plane satellites.
    """

    def __init__(
        self,
        num_planes: int = C.NUM_ORBITAL_PLANES,
        sats_per_plane: int = C.SATS_PER_PLANE,
        inclination_rad: float = C.INCLINATION_RAD,
        altitude_km: float = C.ALTITUDE_KM,
        phasing_factor: int = C.WALKER_PHASING_FACTOR,
    ):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.inclination = inclination_rad
        self.altitude = altitude_km
        self.phasing_factor = phasing_factor

        self.total_sats = num_planes * sats_per_plane
        self.semi_major_axis = C.EARTH_RADIUS_KM + altitude_km
        self.mean_motion = compute_mean_motion(self.semi_major_axis)

        # Build satellites
        self.satellites: List[Satellite] = []
        self._build_constellation()

    def _build_constellation(self) -> None:
        """
        Create satellites using Walker-Delta pattern.

        For plane p (0 to num_planes-1):
            RAAN_p = p * (2π / num_planes)

        For sat s in plane p (0 to sats_per_plane-1):
            phase_ps = s * (2π / sats_per_plane)
                     + p * phasing_factor * (2π / total_sats)
        """
        for p in range(self.num_planes):
            raan = p * (2 * np.pi / self.num_planes)

            for s in range(self.sats_per_plane):
                # Base phase within plane
                phase = s * (2 * np.pi / self.sats_per_plane)

                # Add inter-plane phasing to avoid alignment
                phase += p * self.phasing_factor * (2 * np.pi / self.total_sats)

                sat_id = p * self.sats_per_plane + s

                sat = Satellite(
                    sat_id=sat_id,
                    semi_major_axis_km=self.semi_major_axis,
                    inclination_rad=self.inclination,
                    raan_rad=raan,
                    phase_rad=phase,
                    mean_motion_rad_s=self.mean_motion,
                )
                self.satellites.append(sat)

    # --------------------------------------------------------
    # QUERY METHODS
    # --------------------------------------------------------

    def get_positions(self, t_sec: float) -> np.ndarray:
        """
        Get positions of all satellites at time t.

        Returns:
            positions: (N, 3) array [km]
        """
        return np.array([sat.get_position_eci(t_sec) for sat in self.satellites])

    def get_velocities(self, t_sec: float) -> np.ndarray:
        """
        Get velocities of all satellites at time t.

        Returns:
            velocities: (N, 3) array [km/s]
        """
        return np.array([sat.get_velocity_eci(t_sec) for sat in self.satellites])

    def get_satellite(self, sat_id: int) -> Satellite:
        """Get satellite by ID."""
        return self.satellites[sat_id]

    def get_satellite_plane(self, sat_id: int) -> int:
        """Get orbital plane index for a satellite."""
        return sat_id // self.sats_per_plane

    def __len__(self) -> int:
        return len(self.satellites)

    def __repr__(self) -> str:
        return (
            f"Constellation({self.total_sats} sats, "
            f"{self.num_planes} planes × {self.sats_per_plane} sats, "
            f"alt={self.altitude:.0f}km, i={np.rad2deg(self.inclination):.1f}°)"
        )


# ============================================================
# FACTORY
# ============================================================

def create_default_constellation() -> Constellation:
    """Create the standard MLight-Routing constellation (Iridium-like)."""
    return Constellation()


# ============================================================
# ANALYTICS
# ============================================================

def constellation_stats(constellation: Constellation, t_sec: float = 0.0) -> Dict:
    """
    Compute basic stats about the constellation at time t.

    Returns:
        dict with min/max/mean altitude, positions spread, etc.
    """
    positions = constellation.get_positions(t_sec)
    radii = np.linalg.norm(positions, axis=1)

    return {
        'num_satellites': len(constellation),
        'num_planes': constellation.num_planes,
        'sats_per_plane': constellation.sats_per_plane,
        'mean_radius_km': float(np.mean(radii)),
        'std_radius_km': float(np.std(radii)),
        'min_radius_km': float(np.min(radii)),
        'max_radius_km': float(np.max(radii)),
        'expected_radius_km': constellation.semi_major_axis,
    }
