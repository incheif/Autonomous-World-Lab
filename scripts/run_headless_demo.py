"""
Demonstration script running Autonomous-World-Lab headlessly without browser/graphics.
Verifies fixed-step kinematic integration, road boundary tracking, and telemetry export.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autonomy.interfaces.types import ControlCommand
from simulator.simulation import Simulation
from simulator.world.road import RoadMap


def main():
    print("=" * 70)
    print(" 🚗 AUTONOMOUS-WORLD-LAB · HEADLESS SIMULATION DEMO (PHASE 1)")
    print("=" * 70)

    # 1. Create S-curve road and simulation
    road = RoadMap(road_type="s_curve", road_length_m=120.0, lane_width_m=3.75, num_lanes=2)
    sim = Simulation(dt_sec=0.02, road=road)

    print(f"Road type: {road.road_type} (length: {road.road_length_m}m, lanes: {road.num_lanes})")
    print(f"Start Pose: x={road.start_pose[0]}m, y={road.start_pose[1]}m, yaw={road.start_pose[2]}rad")
    print(f"Goal Pose:  x={road.goal_pose[0]}m, y={road.goal_pose[1]}m, yaw={road.goal_pose[2]}rad")
    print("-" * 70)

    # 2. Run simulation with simple scripted controller (accelerate and steer)
    def simple_scripted_driver(state, road_map):
        # Accelerate up to 8 m/s (~30 km/h) then hold
        accel = 2.0 if state.velocity < 8.0 else 0.0
        # Gentle sinusoidal steering input
        steer = 0.12 if state.x < 30.0 else (-0.12 if state.x < 70.0 else 0.05)
        return ControlCommand(acceleration=accel, steering=steer)

    print("Running 5.0 seconds of simulation headlessly (250 steps @ 50 Hz)...")
    summary = sim.run_headless(duration_sec=5.0, controller_fn=simple_scripted_driver)

    print("\nSimulation Finished:")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    # 3. Export CSV log
    out_dir = PROJECT_ROOT / "runs"
    out_csv = out_dir / "headless_demo_trajectory.csv"
    sim.logger.export_csv(out_csv)
    print(f"\nSaved trajectory log: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
