"""
Orbital mechanics for LEO satellites.

Implements Keplerian orbit propagation for circular orbits.
Used by constellation.py to compute satellite positions over time.

Reference: Vallado, "Fundamentals of Astrodynamics and Applications"
"""

import numpy as np
from typing import Tuple
from . import constants as C


class Satellite:
    """
    Represents a single LEO satellite with circular orbit.

    Position is computed using Keplerian elements:
        - semi_major_axis_km: orbit size
        - inclination_rad: orbital plane tilt
        - raan_rad: right ascension of ascending node (plane orientation)
        - phase_rad: initial position within orbit

    All angles in radians.
    """

    def __init__(
        self,
        sat_id: int,
        semi_major_axis_km: float,
        inclination_rad: float,
        raan_rad: float,
        phase_rad: float,
        mean_motion_rad_s: float,
    ):
        self.sat_id = sat_id
        self.a = semi_major_axis_km
        self.i = inclination_rad
        self.raan = raan_rad
        self.phase_0 = phase_rad
        self.n = mean_motion_rad_s  # mean motion [rad/s]

        # Cache for last position (avoid recomputation)
        self._last_t = None
        self._last_position = None
        self._last_velocity = None

    # --------------------------------------------------------
    # POSITION COMPUTATION
    # --------------------------------------------------------

    def get_position_eci(self, t_sec: float) -> np.ndarray:
        """
        Compute satellite position in Earth-Centered Inertial (ECI) frame.

        For circular orbit:
            true_anomaly(t) = phase_0 + n * t
            position in orbit plane: [a*cos(ν), a*sin(ν), 0]
            rotate by inclination and RAAN to get ECI coordinates

        Args:
            t_sec: Time since epoch [seconds]

        Returns:
            position: (3,) array in ECI frame [km]
        """
        # True anomaly at time t (circular orbit: ν = phase + mean_motion * t)
        nu = self.phase_0 + self.n * t_sec

        # Position in orbital plane (perifocal frame)
        x_orbit = self.a * np.cos(nu)
        y_orbit = self.a * np.sin(nu)
        z_orbit = 0.0

        # Rotation matrix: perifocal → ECI
        # R = R_z(-Ω) * R_x(-i) * R_z(-ω)
        # For circular orbit: ω = 0
        cos_raan = np.cos(self.raan)
        sin_raan = np.sin(self.raan)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)

        # R_z(-Ω) * R_x(-i) applied to [x, y, z]
        # Full transformation:
        x_eci = (
            cos_raan * x_orbit
            - sin_raan * cos_i * y_orbit
        )
        y_eci = (
            sin_raan * x_orbit
            + cos_raan * cos_i * y_orbit
        )
        z_eci = sin_i * y_orbit

        return np.array([x_eci, y_eci, z_eci])

    def get_velocity_eci(self, t_sec: float) -> np.ndarray:
        """
        Compute satellite velocity in ECI frame.

        For circular orbit: v = sqrt(mu/a), direction is perpendicular to position.

        Args:
            t_sec: Time since epoch [seconds]

        Returns:
            velocity: (3,) array in ECI frame [km/s]
        """
        nu = self.phase_0 + self.n * t_sec

        # Velocity magnitude for circular orbit
        v_mag = np.sqrt(C.EARTH_MU_KM3_S2 / self.a)

        # Velocity in orbital plane (perpendicular to position)
        vx_orbit = -v_mag * np.sin(nu)
        vy_orbit = v_mag * np.cos(nu)
        vz_orbit = 0.0

        # Same rotation as position
        cos_raan = np.cos(self.raan)
        sin_raan = np.sin(self.raan)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)

        vx_eci = (
            cos_raan * vx_orbit
            - sin_raan * cos_i * vy_orbit
        )
        vy_eci = (
            sin_raan * vx_orbit
            + cos_raan * cos_i * vy_orbit
        )
        vz_eci = sin_i * vy_orbit

        return np.array([vx_eci, vy_eci, vz_eci])

    # --------------------------------------------------------
    # GROUND-RELATIVE GEOMETRY
    # --------------------------------------------------------

    def get_position_ecef(self, t_sec: float) -> np.ndarray:
        """
        Convert ECI position to Earth-Centered Earth-Fixed (ECEF) frame.

        Accounts for Earth's rotation.

        Args:
            t_sec: Time since epoch [seconds]

        Returns:
            position_ecef: (3,) array [km]
        """
        pos_eci = self.get_position_eci(t_sec)

        # Rotation angle of Earth at time t
        theta = C.EARTH_ROTATION_RATE_RAD_S * t_sec
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # R_z(-θ) applied to ECI position
        x_ecef = cos_t * pos_eci[0] + sin_t * pos_eci[1]
        y_ecef = -sin_t * pos_eci[0] + cos_t * pos_eci[1]
        z_ecef = pos_eci[2]

        return np.array([x_ecef, y_ecef, z_ecef])

    # --------------------------------------------------------
    # UTILITY
    # --------------------------------------------------------

    def distance_to(self, other_pos_km: np.ndarray, t_sec: float) -> float:
        """Distance to a fixed point (e.g., ground device) [km]."""
        pos = self.get_position_eci(t_sec)
        return float(np.linalg.norm(pos - other_pos_km))

    def __repr__(self) -> str:
        return (
            f"Satellite(id={self.sat_id}, "
            f"a={self.a:.1f}km, i={np.rad2deg(self.i):.1f}°, "
            f"raan={np.rad2deg(self.raan):.1f}°)"
        )


# ============================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================

def compute_mean_motion(a_km: float) -> float:
    """Kepler's third law: n = sqrt(mu / a^3) [rad/s]."""
    return np.sqrt(C.EARTH_MU_KM3_S2 / a_km**3)


def eci_to_geodetic(pos_eci_km: np.ndarray, t_sec: float) -> Tuple[float, float, float]:
    """
    Convert ECI position to geodetic (lat, lon, alt).

    Args:
        pos_eci_km: (3,) array in ECI frame [km]
        t_sec: Time since epoch [seconds]

    Returns:
        (lat_deg, lon_deg, alt_km)
    """
    # Rotate to ECEF
    theta = C.EARTH_ROTATION_RATE_RAD_S * t_sec
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = cos_t * pos_eci_km[0] + sin_t * pos_eci_km[1]
    y = -sin_t * pos_eci_km[0] + cos_t * pos_eci_km[1]
    z = pos_eci_km[2]

    # Geodetic conversion (simplified spherical Earth)
    r = np.sqrt(x**2 + y**2 + z**2)
    lat_rad = np.arcsin(z / r)
    lon_rad = np.arctan2(y, x)

    lat_deg = np.rad2deg(lat_rad)
    lon_deg = np.rad2deg(lon_rad)
    alt_km = r - C.EARTH_RADIUS_KM

    return lat_deg, lon_deg, alt_km
