"""
Integration and Headless execution tests for Simulation.
"""

from pathlib import Path
from autonomy.interfaces.types import ControlCommand
from simulator.simulation import Simulation
from simulator.world.road import RoadMap


def test_headless_simulation_run():
    sim = Simulation(dt_sec=0.02, road=RoadMap(road_type="straight", road_length_m=80.0))
    summary = sim.run_headless(duration_sec=3.0)

    assert summary["total_steps"] == 150
    assert summary["duration_sec"] == 3.0
    assert summary["max_speed_mps"] > 0.0
    assert not summary["collision_occurred"]


def test_headless_logger_export(tmp_path: Path):
    sim = Simulation(dt_sec=0.02)
    sim.run_headless(duration_sec=1.0)

    csv_path = tmp_path / "trajectory.csv"
    json_path = tmp_path / "trajectory.json"

    sim.logger.export_csv(csv_path)
    sim.logger.export_json(json_path)

    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
    assert json_path.exists()
    assert json_path.stat().st_size > 0


def test_headless_deterministic_replay():
    def make_sim():
        road = RoadMap(road_type="curve", road_length_m=100.0)
        return Simulation(dt_sec=0.02, road=road)

    sim1 = make_sim()
    sim2 = make_sim()

    # Apply identical commands
    cmds = [
        ControlCommand(acceleration=2.0, steering=0.1),
        ControlCommand(acceleration=1.0, steering=-0.15),
        ControlCommand(acceleration=-0.5, steering=0.05),
    ]

    for cmd in cmds:
        for _ in range(50):
            st1, _ = sim1.step(cmd)
            st2, _ = sim2.step(cmd)
            assert st1.x == st2.x
            assert st1.y == st2.y
            assert st1.yaw == st2.yaw
            assert st1.velocity == st2.velocity

