"""
Unit tests for MLight-Routing simulation modules.

Run: pytest tests/test_simulation.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.simulation import constants as C
from src.simulation.orbit import Satellite, compute_mean_motion
from src.simulation.constellation import Constellation, create_default_constellation
from src.simulation.link import (
    is_in_range, is_line_of_sight_clear, is_link_available,
    compute_adjacency_matrix, count_isl_links,
)
from src.simulation.channel import (
    compute_doppler_hz, compute_propagation_delay_ms, compute_snr_db,
    compute_link_metrics,
)


# ============================================================
# CONSTANTS TESTS
# ============================================================

class TestConstants:
    def test_earth_radius(self):
        assert 6370 < C.EARTH_RADIUS_KM < 6380

    def test_altitude(self):
        assert C.ALTITUDE_KM == 500.0

    def test_semi_major_axis(self):
        assert abs(C.SEMI_MAJOR_AXIS_KM - 6871.0) < 0.1

    def test_orbital_period(self):
        # LEO ~94 min
        assert 90 < C.ORBITAL_PERIOD_MIN < 100

    def test_velocity(self):
        # LEO ~7.6 km/s
        assert 7.4 < C.SATELLITE_VELOCITY_KM_S < 7.8

    def test_max_doppler(self):
        # ~50 kHz at 2 GHz
        assert 48 < C.MAX_DOPPLER_KHZ < 52


# ============================================================
# ORBIT TESTS
# ============================================================

class TestOrbit:
    def setup_method(self):
        n = compute_mean_motion(C.SEMI_MAJOR_AXIS_KM)
        self.sat = Satellite(
            sat_id=0,
            semi_major_axis_km=C.SEMI_MAJOR_AXIS_KM,
            inclination_rad=C.INCLINATION_RAD,
            raan_rad=0.0,
            phase_rad=0.0,
            mean_motion_rad_s=n,
        )

    def test_position_magnitude(self):
        """Position magnitude should equal semi-major axis for circular orbit."""
        pos = self.sat.get_position_eci(t_sec=0.0)
        assert abs(np.linalg.norm(pos) - C.SEMI_MAJOR_AXIS_KM) < 1.0

    def test_velocity_magnitude(self):
        """Velocity should match circular orbit velocity."""
        vel = self.sat.get_velocity_eci(t_sec=0.0)
        assert abs(np.linalg.norm(vel) - C.SATELLITE_VELOCITY_KM_S) < 0.01

    def test_periodicity(self):
        """After one orbital period, satellite returns to starting position."""
        pos_0 = self.sat.get_position_eci(0.0)
        pos_T = self.sat.get_position_eci(C.ORBITAL_PERIOD_S)
        assert np.allclose(pos_0, pos_T, atol=1.0)

    def test_position_changes_over_time(self):
        pos_0 = self.sat.get_position_eci(0.0)
        pos_60 = self.sat.get_position_eci(60.0)
        assert not np.allclose(pos_0, pos_60, atol=1.0)


# ============================================================
# CONSTELLATION TESTS
# ============================================================

class TestConstellation:
    def setup_method(self):
        self.const = create_default_constellation()

    def test_total_satellites(self):
        assert len(self.const) == 66

    def test_planes_and_sats(self):
        assert self.const.num_planes == 6
        assert self.const.sats_per_plane == 11

    def test_all_satellites_at_altitude(self):
        positions = self.const.get_positions(0.0)
        radii = np.linalg.norm(positions, axis=1)
        assert np.allclose(radii, C.SEMI_MAJOR_AXIS_KM, atol=1.0)

    def test_positions_shape(self):
        positions = self.const.get_positions(0.0)
        assert positions.shape == (66, 3)

    def test_velocities_shape(self):
        vel = self.const.get_velocities(0.0)
        assert vel.shape == (66, 3)


# ============================================================
# LINK TESTS
# ============================================================

class TestLink:
    def test_range_check_far(self):
        a = np.array([0.0, 0.0, 6871.0])
        b = np.array([0.0, 0.0, 6871.0 + C.MAX_ISL_RANGE_KM + 100])
        assert not is_in_range(a, b)

    def test_range_check_near(self):
        a = np.array([0.0, 0.0, 6871.0])
        b = np.array([0.0, 0.0, 6871.0 + 100.0])
        assert is_in_range(a, b)

    def test_los_clear_above_earth(self):
        """Two satellites on same side of Earth should have clear LOS."""
        a = np.array([6871.0, 0.0, 0.0])
        b = np.array([6871.0, 100.0, 0.0])
        assert is_line_of_sight_clear(a, b)

    def test_los_blocked_by_earth(self):
        """Opposite sides of Earth → LOS blocked."""
        a = np.array([6871.0, 0.0, 0.0])
        b = np.array([-6871.0, 0.0, 0.0])
        assert not is_line_of_sight_clear(a, b)

    def test_adjacency_symmetric(self):
        const = create_default_constellation()
        pos = const.get_positions(0.0)
        adj = compute_adjacency_matrix(pos)
        assert np.array_equal(adj, adj.T)

    def test_adjacency_no_self_loop(self):
        const = create_default_constellation()
        pos = const.get_positions(0.0)
        adj = compute_adjacency_matrix(pos)
        assert not np.any(np.diag(adj))

    def test_links_exist(self):
        const = create_default_constellation()
        pos = const.get_positions(0.0)
        adj = compute_adjacency_matrix(pos)
        num_links = count_isl_links(adj)
        assert num_links > 0


# ============================================================
# CHANNEL TESTS
# ============================================================

class TestChannel:
    def test_delay_overhead(self):
        """Directly overhead: distance = altitude, delay ~1.67 ms."""
        sat = np.array([0.0, 0.0, C.SEMI_MAJOR_AXIS_KM])
        ground = np.array([0.0, 0.0, C.EARTH_RADIUS_KM])
        delay = compute_propagation_delay_ms(sat, ground)
        assert abs(delay - C.MIN_DELAY_MS) < 0.1

    def test_delay_positive(self):
        a = np.array([6871.0, 0.0, 0.0])
        b = np.array([6871.0, 100.0, 0.0])
        assert compute_propagation_delay_ms(a, b) > 0

    def test_doppler_zero_for_parallel_motion(self):
        """Two satellites moving in same direction at same speed → zero Doppler."""
        pos_a = np.array([6871.0, 0.0, 0.0])
        pos_b = np.array([6871.0, 100.0, 0.0])
        vel_a = np.array([0.0, 7.6, 0.0])
        vel_b = np.array([0.0, 7.6, 0.0])
        f_d = compute_doppler_hz(pos_a, vel_a, pos_b, vel_b)
        assert abs(f_d) < 1.0

    def test_doppler_sign_approaching(self):
        """Satellite moving toward another → positive Doppler."""
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1000.0, 0.0, 0.0])
        vel_a = np.array([7.6, 0.0, 0.0])   # moving toward b
        vel_b = np.array([0.0, 0.0, 0.0])
        f_d = compute_doppler_hz(pos_a, vel_a, pos_b, vel_b)
        assert f_d > 0

    def test_doppler_bounded(self):
        """Doppler should never exceed theoretical max."""
        const = create_default_constellation()
        pos = const.get_positions(0.0)
        vel = const.get_velocities(0.0)
        for i in range(5):
            for j in range(i + 1, min(i + 5, len(const))):
                f_d = compute_doppler_hz(pos[i], vel[i], pos[j], vel[j])
                assert abs(f_d) <= C.MAX_DOPPLER_HZ * 2.1  # 2× safety margin for ISL

    def test_snr_decreases_with_distance(self):
        snr_near = compute_snr_db(500.0)
        snr_far = compute_snr_db(5000.0)
        assert snr_near > snr_far

    def test_link_metrics_keys(self):
        const = create_default_constellation()
        pos = const.get_positions(0.0)
        vel = const.get_velocities(0.0)
        metrics = compute_link_metrics(pos[0], vel[0], pos[1], vel[1])
        assert set(metrics.keys()) == {
            'distance_km', 'delay_ms', 'doppler_hz', 'snr_db', 'path_loss_db'
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
