"""
Training data generator for PNT estimation.

Two-stage approach:
    Stage 1 (analytical): crude position from satellite geometry
    Stage 2 (GBT):        correct residual error

Feature vector (7 dims):
    [sat_x, sat_y, sat_z, doppler, delay, snr, elevation]

Target vector (2 dims):
    [residual_lat, residual_lon] = true - analytical
"""

import numpy as np
from typing import Tuple, Dict
from pathlib import Path

from ..simulation import constants as C


# ============================================================
# RANDOM GEOMETRY
# ============================================================

def random_device_position(
    lat_range_deg: Tuple[float, float] = (-60.0, 60.0),
    lon_range_deg: Tuple[float, float] = (-180.0, 180.0),
) -> Tuple[float, float, np.ndarray]:
    """Random device position on Earth's surface."""
    lat = np.random.uniform(*lat_range_deg)
    lon = np.random.uniform(*lon_range_deg)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    r = C.EARTH_RADIUS_KM
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return float(lat), float(lon), np.array([x, y, z])


def random_satellite_overhead(
    device_lat_deg: float,
    device_lon_deg: float,
    min_elevation_deg: float = C.MIN_ELEVATION_DEG,
    max_elevation_deg: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random satellite position visible from device."""
    elev_deg = np.random.uniform(min_elevation_deg, max_elevation_deg)
    azim_deg = np.random.uniform(0.0, 360.0)

    elev_rad = np.deg2rad(elev_deg)
    azim_rad = np.deg2rad(azim_deg)

    lat_rad = np.deg2rad(device_lat_deg)
    lon_rad = np.deg2rad(device_lon_deg)

    device_pos = C.EARTH_RADIUS_KM * np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])

    up = device_pos / np.linalg.norm(device_pos)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    north = np.cross(up, east)

    a_coef = 1.0
    b_coef = 2 * C.EARTH_RADIUS_KM * np.sin(elev_rad)
    c_coef = C.EARTH_RADIUS_KM**2 - C.SEMI_MAJOR_AXIS_KM**2
    d_slant = (-b_coef + np.sqrt(b_coef**2 - 4 * a_coef * c_coef)) / (2 * a_coef)

    direction = (
        np.cos(elev_rad) * (np.cos(azim_rad) * north + np.sin(azim_rad) * east)
        + np.sin(elev_rad) * up
    )

    sat_pos = device_pos + d_slant * direction

    v_mag = C.SATELLITE_VELOCITY_KM_S
    tangent = np.cross(sat_pos, np.random.randn(3))
    tangent = tangent / np.linalg.norm(tangent)
    sat_vel = v_mag * tangent

    return sat_pos, sat_vel


# ============================================================
# OBSERVABLES
# ============================================================

def compute_observables(
    device_pos_ecef: np.ndarray,
    sat_pos_ecef: np.ndarray,
    sat_vel_ecef: np.ndarray,
) -> Dict[str, float]:
    """Compute Doppler, delay, SNR, elevation."""
    delta = sat_pos_ecef - device_pos_ecef
    dist_km = np.linalg.norm(delta)

    if dist_km < 1e-6:
        return {'doppler_hz': 0.0, 'delay_ms': 0.0, 'snr_db': -100.0, 'elevation_deg': 90.0}

    u = delta / dist_km

    device_vel = np.cross([0.0, 0.0, C.EARTH_ROTATION_RATE_RAD_S], device_pos_ecef)
    v_rel = sat_vel_ecef - device_vel
    v_radial_m_s = np.dot(v_rel, u) * 1000.0

    doppler_hz = (v_radial_m_s / C.SPEED_OF_LIGHT_M_S) * C.CARRIER_FREQUENCY_HZ
    delay_ms = (dist_km / C.SPEED_OF_LIGHT_KM_S) * 1000.0

    up = device_pos_ecef / np.linalg.norm(device_pos_ecef)
    sin_elev = np.dot(u, up)
    elevation_deg = np.rad2deg(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))

    fspl_db = 20 * np.log10(4 * np.pi * dist_km * 1000 * C.CARRIER_FREQUENCY_HZ / C.SPEED_OF_LIGHT_M_S)
    p_rx_dbm = C.TX_POWER_DBM + C.TX_ANTENNA_GAIN_DBI + C.RX_ANTENNA_GAIN_DBI - fspl_db
    snr_db = p_rx_dbm - (C.NOISE_POWER_DBM + C.NOISE_FIGURE_DB)

    return {
        'doppler_hz': float(doppler_hz),
        'delay_ms': float(delay_ms),
        'snr_db': float(snr_db),
        'elevation_deg': float(elevation_deg),
    }


# ============================================================
# ANALYTICAL POSITION ESTIMATE
# ============================================================

def analytical_position_estimate(
    sat_pos_ecef: np.ndarray,
    delay_ms: float,
) -> Tuple[float, float]:
    """
    Crude device position from satellite position + delay.

    Device is at distance d = c * delay from satellite, on Earth's surface.

    Method: Find intersection of:
        - Sphere of radius d centered at satellite
        - Sphere of radius Earth_radius centered at origin

    This is a circle. We pick the point closest to satellite's ground track
    (approximation — the real solution requires elevation for uniqueness).
    """
    d_km = (delay_ms / 1000.0) * C.SPEED_OF_LIGHT_KM_S

    # Vector from origin to satellite
    sat_norm = np.linalg.norm(sat_pos_ecef)
    sat_unit = sat_pos_ecef / sat_norm

    # Distance from origin to the circle of intersection (along sat direction)
    # Use law of cosines: R² = d² + sat_r² - 2*d*sat_r*cos(angle_from_sat)
    # a = (sat_r² + R² - d²) / (2 * sat_r) — projection along sat_unit
    a = (sat_norm**2 + C.EARTH_RADIUS_KM**2 - d_km**2) / (2 * sat_norm)
    a = np.clip(a, -C.EARTH_RADIUS_KM, C.EARTH_RADIUS_KM)

    # Perpendicular distance from sat direction to circle
    h_sq = C.EARTH_RADIUS_KM**2 - a**2
    h = np.sqrt(max(h_sq, 0.0))

    # Point on the line from origin in sat direction at distance a
    base_point = a * sat_unit

    # Build tangent frame perpendicular to sat_unit
    if abs(sat_unit[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(sat_unit, tmp)
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(sat_unit, t1)

    # We can't pick a unique point from this circle. Pick the one in the t1 direction.
    # This is a rough approximation; GBT will correct the error.
    device_pos_est = base_point + h * t1

    # Convert ECEF to lat/lon
    r = np.linalg.norm(device_pos_est)
    lat_est = np.rad2deg(np.arcsin(device_pos_est[2] / r))
    lon_est = np.rad2deg(np.arctan2(device_pos_est[1], device_pos_est[0]))

    return float(lat_est), float(lon_est)


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_sample() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one (features, target, analytical_estimate) triple.

    Returns:
        features:  (7,) — [sat_x, sat_y, sat_z, doppler, delay, snr, elevation]
        target:    (2,) — [residual_lat, residual_lon] = true - analytical
        analytical: (2,) — [analytical_lat, analytical_lon]
    """
    lat, lon, device_pos = random_device_position()
    sat_pos, sat_vel = random_satellite_overhead(lat, lon)
    obs = compute_observables(device_pos, sat_pos, sat_vel)

    # Analytical estimate
    lat_est, lon_est = analytical_position_estimate(sat_pos, obs['delay_ms'])

    # Residual target
    residual_lat = lat - lat_est
    residual_lon = lon - lon_est

    features = np.array([
        sat_pos[0], sat_pos[1], sat_pos[2],
        obs['doppler_hz'],
        obs['delay_ms'],
        obs['snr_db'],
        obs['elevation_deg'],
    ], dtype=np.float64)

    target = np.array([residual_lat, residual_lon], dtype=np.float64)
    analytical = np.array([lat_est, lon_est], dtype=np.float64)

    return features, target, analytical


def generate_dataset(
    n_samples: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate full dataset.

    Returns:
        X: (n_samples, 7) features
        y: (n_samples, 2) residual targets
        A: (n_samples, 2) analytical estimates
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X = np.zeros((n_samples, 7))
    y = np.zeros((n_samples, 2))
    A = np.zeros((n_samples, 2))

    for i in range(n_samples):
        X[i], y[i], A[i] = generate_sample()

    np.random.set_state(rng_state)
    return X, y, A


def save_dataset(X, y, A, out_dir: str = "data/training") -> None:
    """Save dataset to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "pnt_features.npy", X)
    np.save(out_path / "pnt_targets.npy", y)
    np.save(out_path / "pnt_analytical.npy", A)
    print(f"Saved: {out_path / 'pnt_features.npy'}  (shape: {X.shape})")
    print(f"Saved: {out_path / 'pnt_targets.npy'}  (shape: {y.shape})")
    print(f"Saved: {out_path / 'pnt_analytical.npy'}  (shape: {A.shape})")


def load_dataset(in_dir: str = "data/training"):
    """Load dataset from disk."""
    in_path = Path(in_dir)
    X = np.load(in_path / "pnt_features.npy")
    y = np.load(in_path / "pnt_targets.npy")
    A = np.load(in_path / "pnt_analytical.npy")
    return X, y, A
