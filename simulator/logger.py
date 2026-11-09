"""
Simulation Logger and Telemetry Recorder.
Records deterministic trajectory states, control inputs, and event markers for evaluation.
"""

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from autonomy.interfaces.types import VehicleState, ControlCommand


@dataclass
class LogEntry:
    step: int
    time: float
    x: float
    y: float
    yaw: float
    velocity: float
    steering_angle: float
    acceleration: float
    angular_velocity: float
    cmd_accel: float
    cmd_steering: float
    cmd_brake: float
    collision: bool
    goal_reached: bool
    emergency_stop: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimulationLogger:
    """
    In-memory and file-backed logger for simulation state transitions.
    """

    def __init__(self, max_buffer_size: int = 100000):
        self.max_buffer_size: int = max_buffer_size
        self.entries: List[LogEntry] = []

    def record(
        self,
        step: int,
        state: VehicleState,
        command: ControlCommand,
        collision: bool = False,
        goal_reached: bool = False,
    ) -> LogEntry:
        """Appends one simulation step log entry."""
        entry = LogEntry(
            step=step,
            time=round(state.timestamp, 4),
            x=round(state.x, 4),
            y=round(state.y, 4),
            yaw=round(state.yaw, 4),
            velocity=round(state.velocity, 4),
            steering_angle=round(state.steering_angle, 4),
            acceleration=round(state.acceleration, 4),
            angular_velocity=round(state.angular_velocity, 4),
            cmd_accel=round(command.acceleration, 4),
            cmd_steering=round(command.steering, 4),
            cmd_brake=round(command.brake, 4),
            collision=collision,
            goal_reached=goal_reached,
            emergency_stop=command.emergency_stop,
        )
        if len(self.entries) < self.max_buffer_size:
            self.entries.append(entry)
        return entry

    def clear(self) -> None:
        """Clears all logged entries."""
        self.entries.clear()

    def export_csv(self, file_path: str | Path) -> None:
        """Exports all recorded log entries to a CSV file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.entries:
            return

        keys = list(self.entries[0].to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for e in self.entries:
                writer.writerow(e.to_dict())

    def export_json(self, file_path: str | Path) -> None:
        """Exports all recorded log entries to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> Dict[str, Any]:
        """Returns high-level summary statistics of the recorded run."""
        if not self.entries:
            return {"steps": 0, "duration_sec": 0.0}

        speeds = [e.velocity for e in self.entries]
        collisions = any(e.collision for e in self.entries)
        goal = any(e.goal_reached for e in self.entries)

        return {
            "total_steps": len(self.entries),
            "duration_sec": round(self.entries[-1].time, 4),
            "max_speed_mps": round(max(speeds), 3) if speeds else 0.0,
            "mean_speed_mps": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
            "final_x": self.entries[-1].x,
            "final_y": self.entries[-1].y,
            "collision_occurred": collisions,
            "goal_reached": goal,
        }
