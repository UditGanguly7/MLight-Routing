"""
LEO channel model: Doppler, propagation delay, SNR.

Computes link-layer observables used throughout MLight-Routing:
    - Doppler shift (from relative velocity)
    - Propagation delay (from distance)
    - Received SNR (from link budget)

Reference: 3GPP TR 38.821 (NTN channel models)
"""

import numpy as np
from typing import Tuple
from . import constants as C


# ============================================================
# DOPPLER SHIFT
# ============================================================

def compute_doppler_hz(
    pos_a_km: np.ndarray,
    vel_a_km_s: np.ndarray,
    pos_b_km: np.ndarray,
    vel_b_km_s: np.ndarray,
    carrier_freq_hz: float = C.CARRIER_FREQUENCY_HZ,
) -> float:
    """
    Compute Doppler shift between two moving objects.

    f_d = (f_c / c) * (v_rel · u_ab)

    where:
        v_rel = relative velocity (v_a - v_b)
        u_ab  = unit vector from A to B
        f_c   = carrier frequency

    Sign convention:
        Positive = closing (approaching)
        Negative = opening (moving apart)

    Args:
        pos_a_km, pos_b_km: (3,) positions [km]
        vel_a_km_s, vel_b_km_s: (3,) velocities [km/s]
        carrier_freq_hz: carrier frequency [Hz]

    Returns:
        Doppler shift [Hz]
    """
    # Unit vector from A to B
    delta_pos = pos_b_km - pos_a_km
    dist = np.linalg.norm(delta_pos)
    if dist < 1e-6:
        return 0.0
    u_ab = delta_pos / dist

    # Relative velocity (A's perspective)
    v_rel = vel_a_km_s - vel_b_km_s

    # Radial velocity component [km/s]
    v_radial_km_s = np.dot(v_rel, u_ab)

    # Convert to m/s and apply Doppler formula
    v_radial_m_s = v_radial_km_s * 1000.0
    f_d = (v_radial_m_s / C.SPEED_OF_LIGHT_M_S) * carrier_freq_hz

    return float(f_d)


def compute_doppler_satellite_to_ground_hz(
    sat_pos_km: np.ndarray,
    sat_vel_km_s: np.ndarray,
    device_pos_ecef_km: np.ndarray,
    carrier_freq_hz: float = C.CARRIER_FREQUENCY_HZ,
) -> float:
    """
    Doppler shift for satellite-to-ground link.

    Assumes device is stationary on Earth's surface.
    Device position is in ECEF; satellite is converted from ECI to ECEF.
    """
    # Device velocity is Earth rotation
    device_vel_ecef = np.cross(
        [0.0, 0.0, C.EARTH_ROTATION_RATE_RAD_S],
        device_pos_ecef_km,
    )

    # Satellite: convert ECI → ECEF requires knowing time.
    # For simplicity, assume caller provides satellite position already in ECEF.
    delta_pos = device_pos_ecef_km - sat_pos_km
    dist = np.linalg.norm(delta_pos)
    if dist < 1e-6:
        return 0.0
    u_ab = delta_pos / dist

    # Relative velocity
    v_rel = sat_vel_km_s - device_vel_ecef
    v_radial_km_s = np.dot(v_rel, u_ab)
    v_radial_m_s = v_radial_km_s * 1000.0
    f_d = (v_radial_m_s / C.SPEED_OF_LIGHT_M_S) * carrier_freq_hz

    return float(f_d)


# ============================================================
# PROPAGATION DELAY
# ============================================================

def compute_propagation_delay_ms(pos_a_km: np.ndarray, pos_b_km: np.ndarray) -> float:
    """
    One-way propagation delay [ms].

    τ = d / c
    """
    dist_km = np.linalg.norm(pos_a_km - pos_b_km)
    delay_s = dist_km / C.SPEED_OF_LIGHT_KM_S
    return delay_s * 1000.0


# ============================================================
# LINK BUDGET / SNR
# ============================================================

def free_space_path_loss_db(dist_km: float, freq_hz: float = C.CARRIER_FREQUENCY_HZ) -> float:
    """
    Free-space path loss [dB].

    FSPL = 20*log10(4πd/λ)
         = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
    """
    if dist_km <= 0:
        return 0.0
    d_m = dist_km * 1000.0
    fspl = 20 * np.log10(4 * np.pi * d_m * freq_hz / C.SPEED_OF_LIGHT_M_S)
    return float(fspl)


def compute_received_power_dbm(
    dist_km: float,
    tx_power_dbm: float = C.TX_POWER_DBM,
    tx_gain_dbi: float = C.TX_ANTENNA_GAIN_DBI,
    rx_gain_dbi: float = C.RX_ANTENNA_GAIN_DBI,
    freq_hz: float = C.CARRIER_FREQUENCY_HZ,
) -> float:
    """
    Received power via Friis equation [dBm].

    P_rx = P_tx + G_tx + G_rx - FSPL
    """
    fspl = free_space_path_loss_db(dist_km, freq_hz)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl


def compute_snr_db(
    dist_km: float,
    tx_power_dbm: float = C.TX_POWER_DBM,
    tx_gain_dbi: float = C.TX_ANTENNA_GAIN_DBI,
    rx_gain_dbi: float = C.RX_ANTENNA_GAIN_DBI,
    freq_hz: float = C.CARRIER_FREQUENCY_HZ,
) -> float:
    """
    Received SNR [dB].

    SNR = P_rx - N
    where N = k*T*B + NF
    """
    p_rx_dbm = compute_received_power_dbm(dist_km, tx_power_dbm, tx_gain_dbi, rx_gain_dbi, freq_hz)
    noise_floor_dbm = C.NOISE_POWER_DBM + C.NOISE_FIGURE_DB
    return p_rx_dbm - noise_floor_dbm


# ============================================================
# COMBINED CHANNEL METRICS
# ============================================================

def compute_link_metrics(
    pos_a_km: np.ndarray,
    vel_a_km_s: np.ndarray,
    pos_b_km: np.ndarray,
    vel_b_km_s: np.ndarray,
) -> dict:
    """
    Compute all channel metrics for a link.

    Returns:
        dict with:
            - distance_km
            - delay_ms
            - doppler_hz
            - snr_db
            - path_loss_db
    """
    dist_km = float(np.linalg.norm(pos_a_km - pos_b_km))
    delay_ms = compute_propagation_delay_ms(pos_a_km, pos_b_km)
    doppler_hz = compute_doppler_hz(pos_a_km, vel_a_km_s, pos_b_km, vel_b_km_s)
    snr_db = compute_snr_db(dist_km)
    fspl_db = free_space_path_loss_db(dist_km)

    return {
        'distance_km': dist_km,
        'delay_ms': delay_ms,
        'doppler_hz': doppler_hz,
        'snr_db': snr_db,
        'path_loss_db': fspl_db,
    }
