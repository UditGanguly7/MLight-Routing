"""
Physical and system constants for MLight-Routing simulation.

All values are based on:
- 3GPP TR 38.821 (NTN channel models)
- Standard orbital mechanics
- NB-IoT / S-band parameters
"""

import numpy as np

# ============================================================
# EARTH PARAMETERS
# ============================================================

EARTH_RADIUS_KM = 6371.0                    # Mean Earth radius [km]
EARTH_MU_KM3_S2 = 398600.4418               # Earth gravitational parameter [km^3/s^2]
EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5    # Earth rotation rate [rad/s]
SPEED_OF_LIGHT_KM_S = 299792.458            # Speed of light [km/s]
SPEED_OF_LIGHT_M_S = 299792458.0            # Speed of light [m/s]

# ============================================================
# CONSTELLATION PARAMETERS (Walker-Delta)
# ============================================================

ALTITUDE_KM = 500.0                          # Orbital altitude [km]
SEMI_MAJOR_AXIS_KM = EARTH_RADIUS_KM + ALTITUDE_KM  # = 6871 km
INCLINATION_DEG = 53.0                       # Orbital inclination [degrees]
INCLINATION_RAD = np.deg2rad(INCLINATION_DEG)

NUM_ORBITAL_PLANES = 6                       # Number of orbital planes
SATS_PER_PLANE = 11                          # Satellites per plane
TOTAL_SATELLITES = NUM_ORBITAL_PLANES * SATS_PER_PLANE  # = 66 (Iridium-like)

# Walker-Delta phasing factor (relative spacing between planes)
WALKER_PHASING_FACTOR = 1

# Derived: orbital period (Kepler's third law)
ORBITAL_PERIOD_S = 2 * np.pi * np.sqrt(SEMI_MAJOR_AXIS_KM**3 / EARTH_MU_KM3_S2)  # ~5677 s
ORBITAL_PERIOD_MIN = ORBITAL_PERIOD_S / 60.0  # ~94.6 min

# Satellite velocity (circular orbit)
SATELLITE_VELOCITY_KM_S = np.sqrt(EARTH_MU_KM3_S2 / SEMI_MAJOR_AXIS_KM)  # ~7.6 km/s

# ============================================================
# CHANNEL PARAMETERS (S-band NB-IoT)
# ============================================================

CARRIER_FREQUENCY_HZ = 2.0e9                 # 2 GHz S-band
CARRIER_FREQUENCY_GHZ = 2.0
WAVELENGTH_M = SPEED_OF_LIGHT_M_S / CARRIER_FREQUENCY_HZ  # ~0.15 m

BANDWIDTH_HZ = 180e3                         # NB-IoT: 180 kHz
SUBCARRIER_SPACING_HZ = 3.75e3               # NB-IoT: 3.75 kHz

TX_POWER_DBM = 23.0                          # UE transmit power [dBm]
TX_POWER_W = 10**(TX_POWER_DBM / 10.0) / 1000.0  # Convert to watts

NOISE_FIGURE_DB = 5.0                        # Receiver noise figure [dB]
NOISE_TEMPERATURE_K = 290.0                  # Reference noise temperature [K]
BOLTZMANN_CONSTANT = 1.380649e-23            # Boltzmann constant [J/K]

# Thermal noise power: N = k * T * B
NOISE_POWER_W = BOLTZMANN_CONSTANT * NOISE_TEMPERATURE_K * BANDWIDTH_HZ
NOISE_POWER_DBM = 10 * np.log10(NOISE_POWER_W * 1000)

# ============================================================
# ANTENNA PARAMETERS
# ============================================================

TX_ANTENNA_GAIN_DBI = 0.0                    # UE antenna gain [dBi] (omnidirectional)
RX_ANTENNA_GAIN_DBI = 30.0                   # Satellite antenna gain [dBi]
TX_ANTENNA_GAIN_LINEAR = 10**(TX_ANTENNA_GAIN_DBI / 10.0)
RX_ANTENNA_GAIN_LINEAR = 10**(RX_ANTENNA_GAIN_DBI / 10.0)

# ============================================================
# LINK PARAMETERS
# ============================================================

MAX_ISL_RANGE_KM = 5000.0                    # Maximum ISL range [km]
MIN_ELEVATION_DEG = 10.0                     # Minimum elevation angle for ground link
MIN_ELEVATION_RAD = np.deg2rad(MIN_ELEVATION_DEG)

# ============================================================
# DERIVED DOPPLER / DELAY BOUNDS
# ============================================================

MAX_DOPPLER_HZ = (SATELLITE_VELOCITY_KM_S * 1000 / SPEED_OF_LIGHT_M_S) * CARRIER_FREQUENCY_HZ
MAX_DOPPLER_KHZ = MAX_DOPPLER_HZ / 1000.0    # ~50.7 kHz

MIN_DELAY_MS = (ALTITUDE_KM / SPEED_OF_LIGHT_KM_S) * 1000.0  # ~1.67 ms
# Max delay at horizon: sqrt((R+h)^2 - R^2) / c
_max_slant_range_km = np.sqrt(SEMI_MAJOR_AXIS_KM**2 - EARTH_RADIUS_KM**2)
MAX_DELAY_MS = (_max_slant_range_km / SPEED_OF_LIGHT_KM_S) * 1000.0  # ~6.6 ms
