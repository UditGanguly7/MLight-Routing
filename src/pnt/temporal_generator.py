"""
Temporal training data generator for PNT estimation.

Generates satellite pass sequences (90 s) instead of single snapshots.
The evolution of Doppler/delay/SNR/elevation over time uniquely
identifies device position -- resolving the ambiguity of single-snapshot PNT.

Feature sequence (T=90, F=4):
    [doppler_hz, delay_ms, snr_db, elevation_deg] at each time step

Target (2,):
    [lat_deg, lon_deg]
"""

import numpy as np
from typing import Tuple, Dict
from pathlib import Path

from ..simulation import constants as C


# ============================================================
# CONFIGURATION
# ============================================================

PASS_DURATION_S = 480.0        # 8 min — real LEO visibility window
PASS_TIME_STEP_S = 4.0         # 4 s steps → 120 timesteps
PASS_NUM_STEPS = int(PASS_DURATION_S / PASS_TIME_STEP_S)   # 120

# ============================================================
# RANDOM DEVICE POSITION
# ============================================================

def random_device_position(
    lat_range_deg: Tuple[float, float] = (-60.0, 60.0),
    lon_range_deg: Tuple[float, float] = (-180.0, 180.0),
) -> Tuple[float, float, np.ndarray]:
    """Random device position on Earth's surface (ECEF km)."""
    lat = np.random.uniform(*lat_range_deg)
    lon = np.random.uniform(*lon_range_deg)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    r = C.EARTH_RADIUS_KM
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return float(lat), float(lon), np.array([x, y, z])


# ============================================================
# SATELLITE PASS GENERATOR
# ============================================================

def random_satellite_pass(
    device_lat_deg: float,
    device_lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a satellite pass near the device.

    The satellite travels along a great-circle ground track that is
    offset perpendicular to the device by 0-300 km. Arc length equals
    satellite velocity × pass duration (physically consistent).

    Returns:
        sat_positions: (T, 3) ECEF positions [km]
        sat_velocities: (T, 3) ECEF velocities [km/s]
        t_array: (T,) time stamps [s]
    """
    lat_rad = np.deg2rad(device_lat_deg)
    lon_rad = np.deg2rad(device_lon_deg)

    device_pos = C.EARTH_RADIUS_KM * np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])

    # Local horizontal frame at device
    up = device_pos / np.linalg.norm(device_pos)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    north = np.cross(up, east)

    # Ground track offset (0-300 km). Small values → near-overhead passes.
    offset_km = np.random.uniform(0.0, 800.0)    # wider range
    track_azim_rad = np.deg2rad(np.random.uniform(0.0, 360.0))

    track_dir = np.cos(track_azim_rad) * north + np.sin(track_azim_rad) * east
    perp_dir = -np.sin(track_azim_rad) * north + np.cos(track_azim_rad) * east

    # Closest approach point (ground track passes here at midpoint)
    closest_pt = device_pos + offset_km * perp_dir
    closest_pt = closest_pt / np.linalg.norm(closest_pt) * C.EARTH_RADIUS_KM

    # Physical arc length: velocity × time
    arc_total_km = C.SATELLITE_VELOCITY_KM_S * PASS_DURATION_S   # 7.6 × 480 = 3650 km
    arc_half_km = arc_total_km / 2.0

    # Rotation axis: perpendicular to (closest_pt × track_dir)
    axis = np.cross(closest_pt, track_dir)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = np.cross(closest_pt, np.array([0.0, 0.0, 1.0]))
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
            axis_norm = 1.0
    axis = axis / axis_norm

    # Ensure axis direction so that motion is "forward" (toward device first)
    # Test: rotate closest_pt by +small angle; check if it moves toward device
    test_angle = 0.01  # radians
    test_pt = (
        closest_pt * np.cos(-test_angle)
        + np.cross(axis, closest_pt) * np.sin(-test_angle)
        + axis * np.dot(axis, closest_pt) * (1 - np.cos(-test_angle))
    )
    # If dot(test_pt - closest_pt, device_pos - closest_pt) < 0, we're moving away → flip axis
    if np.dot(test_pt - closest_pt, device_pos - closest_pt) < 0:
        axis = -axis

    # Angular rate on sphere
    omega_rad_s = C.SATELLITE_VELOCITY_KM_S / C.EARTH_RADIUS_KM

    t_array = np.arange(0, PASS_DURATION_S, PASS_TIME_STEP_S)
    T = len(t_array)
    t_mid = PASS_DURATION_S / 2.0

    sat_positions = np.zeros((T, 3))
    sat_velocities = np.zeros((T, 3))

    for i, t in enumerate(t_array):
        # Rotation angle: at t = mid, satellite is at closest_pt.
        # At t < mid, satellite is "before" closest_pt (approaching).
        dt = t - t_mid
        theta = omega_rad_s * dt

        # Rodrigues rotation of closest_pt around axis
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        ground_pt = (
            closest_pt * cos_t
            + np.cross(axis, closest_pt) * sin_t
            + axis * np.dot(axis, closest_pt) * (1 - cos_t)
        )
        # Reproject to sphere
        ground_pt = ground_pt / np.linalg.norm(ground_pt) * C.EARTH_RADIUS_KM

        sat_positions[i] = ground_pt / C.EARTH_RADIUS_KM * C.SEMI_MAJOR_AXIS_KM

        # Instantaneous tangential velocity (perpendicular to position, along arc)
        tangent = np.cross(axis, ground_pt)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            sat_velocities[i] = np.zeros(3)
        else:
            sat_velocities[i] = C.SATELLITE_VELOCITY_KM_S * tangent / tangent_norm

    return sat_positions, sat_velocities, t_array


# ============================================================
# OBSERVABLES
# ============================================================

def compute_observables_at_time(
    device_pos_ecef: np.ndarray,
    sat_pos_ecef: np.ndarray,
    sat_vel_ecef: np.ndarray,
) -> Dict[str, float]:
    """Compute Doppler, delay, SNR, elevation at one time instant."""
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
# SAMPLE / DATASET
# ============================================================

def generate_temporal_sample() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one satellite pass with satellite reference position.

    Feature sequence (T, 7):
        [sat_x_ref, sat_y_ref, sat_z_ref,   ← satellite ECEF position at t=0
         doppler_hz, delay_ms, snr_db, elevation_deg]
    Target (2,):
        [lat_deg, lon_deg]
    """
    lat, lon, device_pos = random_device_position()
    sat_positions, sat_velocities, _ = random_satellite_pass(lat, lon)

    T = sat_positions.shape[0]
    sequence = np.zeros((T, 7))

    # Reference satellite position (at t=0) — same for every timestep
    sat_ref = sat_positions[0]

    for i in range(T):
        obs = compute_observables_at_time(
            device_pos, sat_positions[i], sat_velocities[i]
        )
        sequence[i, 0] = sat_ref[0]
        sequence[i, 1] = sat_ref[1]
        sequence[i, 2] = sat_ref[2]
        sequence[i, 3] = obs['doppler_hz']
        sequence[i, 4] = obs['delay_ms']
        sequence[i, 5] = obs['snr_db']
        sequence[i, 6] = obs['elevation_deg']

    target = np.array([lat, lon], dtype=np.float64)
    return sequence, target

def generate_temporal_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate temporal dataset.

    Returns:
        X: (n_samples, T, 4)
        y: (n_samples, 2)
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X = np.zeros((n_samples, PASS_NUM_STEPS, 7))
    y = np.zeros((n_samples, 2))

    for i in range(n_samples):
        X[i], y[i] = generate_temporal_sample()

    np.random.set_state(rng_state)
    return X, y


def save_temporal_dataset(X, y, out_dir: str = "data/training") -> None:
    """Save dataset."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "pnt_temporal_X.npy", X)
    np.save(out_path / "pnt_temporal_y.npy", y)
    print(f"Saved: {out_path / 'pnt_temporal_X.npy'}  (shape: {X.shape})")
    print(f"Saved: {out_path / 'pnt_temporal_y.npy'}  (shape: {y.shape})")


def load_temporal_dataset(in_dir: str = "data/training"):
    """Load dataset."""
    in_path = Path(in_dir)
    X = np.load(in_path / "pnt_temporal_X.npy")
    y = np.load(in_path / "pnt_temporal_y.npy")
    return X, y
