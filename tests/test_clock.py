"""
Unit tests for SimulationClock.
"""

import pytest
from simulator.clock import SimulationClock


def test_clock_initialization():
    clock = SimulationClock(dt_sec=0.02)
    assert clock.dt == 0.02
    assert clock.time == 0.0
    assert clock.step_count == 0


def test_clock_invalid_dt():
    with pytest.raises(ValueError):
        SimulationClock(dt_sec=-0.01)
    with pytest.raises(ValueError):
        SimulationClock(dt_sec=0.0)


def test_clock_advancement():
    clock = SimulationClock(dt_sec=0.02)
    for i in range(1, 101):
        t = clock.tick()
        assert clock.step_count == i
        assert round(t, 4) == round(i * 0.02, 4)
    assert round(clock.time, 4) == 2.0


def test_clock_reset():
    clock = SimulationClock(dt_sec=0.02)
    for _ in range(50):
        clock.tick()
    assert clock.step_count == 50
    clock.reset(start_time_sec=10.0)
    assert clock.time == 10.0
    assert clock.step_count == 0

