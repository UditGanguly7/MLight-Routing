"""
Temporal training data generator for PNT estimation.
"""

import numpy as np
from typing import Tuple, Dict
from pathlib import Path

from ..simulation import constants as C


PASS_DURATION_S = 480.0
PASS_TIME_STEP_S = 4.0
PASS_NUM_STEPS = int(PASS_DURATION_S / PASS_TIME_STEP_S)


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


def random_satellite_pass(
    device_lat_deg: float,
    device_lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a satellite pass near the device."""
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

    offset_km = np.random.uniform(0.0, 800.0)
    track_azim_rad = np.deg2rad(np.random.uniform(0.0, 360.0))

    track_dir = np.cos(track_azim_rad) * north + np.sin(track_azim_rad) * east
    perp_dir = -np.sin(track_azim_rad) * north + np.cos(track_azim_rad) * east

    closest_pt = device_pos + offset_km * perp_dir
    closest_pt = closest_pt / np.linalg.norm(closest_pt) * C.EARTH_RADIUS_KM

    axis = np.cross(closest_pt, track_dir)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = np.cross(closest_pt, np.array([0.0, 0.0, 1.0]))
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
            axis_norm = 1.0
    axis = axis / axis_norm

    omega_rad_s = C.SATELLITE_VELOCITY_KM_S / C.EARTH_RADIUS_KM

    t_array = np.arange(0, PASS_DURATION_S, PASS_TIME_STEP_S)
    T = len(t_array)
    t_mid = PASS_DURATION_S / 2.0

    sat_positions = np.zeros((T, 3))
    sat_velocities = np.zeros((T, 3))

    for i, t in enumerate(t_array):
        dt = t - t_mid
        theta = omega_rad_s * dt

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        ground_pt = (
            closest_pt * cos_t
            + np.cross(axis, closest_pt) * sin_t
            + axis * np.dot(axis, closest_pt) * (1 - cos_t)
        )
        ground_pt = ground_pt / np.linalg.norm(ground_pt) * C.EARTH_RADIUS_KM

        sat_positions[i] = ground_pt / C.EARTH_RADIUS_KM * C.SEMI_MAJOR_AXIS_KM

        tangent = np.cross(axis, ground_pt)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            sat_velocities[i] = np.zeros(3)
        else:
            sat_velocities[i] = C.SATELLITE_VELOCITY_KM_S * tangent / tangent_norm

    return sat_positions, sat_velocities, t_array


def compute_observables_at_time(
    device_pos_ecef: np.ndarray,
    sat_pos_ecef: np.ndarray,
    sat_vel_ecef: np.ndarray,
) -> Dict[str, float]:
    """Doppler, delay, SNR, elevation for satellite-to-device link."""
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


def generate_temporal_sample() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one satellite pass.

    Features per timestep (T, 8):
        0: sat_lat_0        (satellite subpoint latitude at t=0)
        1: sat_lon_0        (satellite subpoint longitude at t=0)
        2: doppler_hz       (instantaneous Doppler)
        3: delay_ms         (instantaneous delay)
        4: snr_db           (instantaneous SNR)
        5: elevation_deg    (instantaneous elevation)
        6: azimuth_deg      (instantaneous azimuth to satellite)
        7: elevation_rate   (d(elevation)/dt — geometry indicator)
    """
    lat, lon, device_pos = random_device_position()
    sat_positions, sat_velocities, _ = random_satellite_pass(lat, lon)

    T = sat_positions.shape[0]
    sequence = np.zeros((T, 8))

    sat_ref = sat_positions[0]
    r_sat = np.linalg.norm(sat_ref)
    sat_lat_0 = np.rad2deg(np.arcsin(sat_ref[2] / r_sat))
    sat_lon_0 = np.rad2deg(np.arctan2(sat_ref[1], sat_ref[0]))

    # First pass: compute observables and azimuth
    elevations = np.zeros(T)
    azimuths = np.zeros(T)
    dopplers = np.zeros(T)
    delays = np.zeros(T)
    snrs = np.zeros(T)

    for i in range(T):
        obs = compute_observables_at_time(
            device_pos, sat_positions[i], sat_velocities[i]
        )
        dopplers[i] = obs['doppler_hz']
        delays[i] = obs['delay_ms']
        snrs[i] = obs['snr_db']
        elevations[i] = obs['elevation_deg']

        # Azimuth from device to satellite (local frame)
        delta = sat_positions[i] - device_pos
        delta_norm = delta / np.linalg.norm(delta)

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])
        east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
        north = np.cross(up, east)

        az = np.arctan2(np.dot(delta_norm, east), np.dot(delta_norm, north))
        azimuths[i] = np.rad2deg(az) % 360.0

    # Elevation rate (finite difference)
    elev_rate = np.gradient(elevations, PASS_TIME_STEP_S)

    # Assemble sequence
    for i in range(T):
        sequence[i, 0] = sat_lat_0
        sequence[i, 1] = sat_lon_0
        sequence[i, 2] = dopplers[i]
        sequence[i, 3] = delays[i]
        sequence[i, 4] = snrs[i]
        sequence[i, 5] = elevations[i]
        sequence[i, 6] = azimuths[i]
        sequence[i, 7] = elev_rate[i]

    delta_lat = lat - sat_lat_0
    delta_lon = ((lon - sat_lon_0 + 180.0) % 360.0) - 180.0

    target = np.array([delta_lat, delta_lon], dtype=np.float64)
    reference = np.array([sat_lat_0, sat_lon_0], dtype=np.float64)

    return sequence, target, reference


def generate_temporal_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset. Returns X=(N,T,6), y=(N,2), R=(N,2)."""
    rng_state = np.random.get_state()
    np.random.seed(seed)

    X = np.zeros((n_samples, PASS_NUM_STEPS, 8))
    y = np.zeros((n_samples, 2))
    R = np.zeros((n_samples, 2))

    for i in range(n_samples):
        X[i], y[i], R[i] = generate_temporal_sample()

    np.random.set_state(rng_state)
    return X, y, R


def save_temporal_dataset(X, y, R, out_dir: str = "data/training") -> None:
    """Save dataset to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "pnt_temporal_X.npy", X)
    np.save(out_path / "pnt_temporal_y.npy", y)
    np.save(out_path / "pnt_temporal_R.npy", R)
    print(f"Saved: {out_path / 'pnt_temporal_X.npy'}  (shape: {X.shape})")
    print(f"Saved: {out_path / 'pnt_temporal_y.npy'}  (shape: {y.shape})")
    print(f"Saved: {out_path / 'pnt_temporal_R.npy'}  (shape: {R.shape})")


def load_temporal_dataset(in_dir: str = "data/training"):
    """Load dataset from disk."""
    in_path = Path(in_dir)
    X = np.load(in_path / "pnt_temporal_X.npy")
    y = np.load(in_path / "pnt_temporal_y.npy")
    R = np.load(in_path / "pnt_temporal_R.npy")
    return X, y, R
