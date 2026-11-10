"""
Simulation Clock for fixed-timestep discrete simulation.
Guarantees deterministic, reproducible stepping independent of wall-clock time.
"""

from typing import Optional


class SimulationClock:
    """
    Monotonically increasing fixed-timestep clock.
    """

    def __init__(self, dt_sec: float = 0.02, start_time_sec: float = 0.0):
        """
        dt_sec: Discrete timestep size in seconds (default 0.02s = 50 Hz)
        start_time_sec: Initial simulation timestamp in seconds
        """
        if dt_sec <= 0.0:
            raise ValueError(f"dt_sec must be positive, got {dt_sec}")
        self._dt: float = float(dt_sec)
        self._time: float = float(start_time_sec)
        self._step: int = 0

    @property
    def dt(self) -> float:
        """Fixed timestep size in seconds."""
        return self._dt

    @property
    def time(self) -> float:
        """Current simulation timestamp in seconds."""
        return self._time

    @property
    def step_count(self) -> int:
        """Total number of discrete steps elapsed."""
        return self._step

    def tick(self) -> float:
        """
        Advance clock by exactly one fixed timestep.
        Returns the updated simulation timestamp in seconds.
        """
        self._step += 1
        self._time = round(self._step * self._dt, 8)
        return self._time

    def reset(self, start_time_sec: float = 0.0) -> None:
        """Reset the clock back to initial conditions."""
        self._time = float(start_time_sec)
        self._step = 0

    def __repr__(self) -> str:
        return f"SimulationClock(time={self._time:.3f}s, step={self._step}, dt={self._dt:.3f}s)"

