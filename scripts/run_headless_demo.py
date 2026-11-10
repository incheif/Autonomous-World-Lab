"""
Demonstration script running Autonomous-World-Lab headlessly without browser/graphics.
Verifies fixed-step kinematic integration, 32-beam LiDAR raycasting, road obstacle sensing,
and scan serialization.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autonomy.interfaces.types import ControlCommand
from simulator.simulation import Simulation
from simulator.world.road import RoadMap


def main():
    print("=" * 70)
    print(" 🚗 AUTONOMOUS-WORLD-LAB · HEADLESS SIMULATION & LIDAR DEMO (PHASE 2)")
    print("=" * 70)

    # 1. Create S-curve road with static obstacle boxes and simulation
    road = RoadMap(road_type="s_curve", road_length_m=120.0, lane_width_m=3.75, num_lanes=2, with_obstacles=True)
    sim = Simulation(dt_sec=0.02, road=road)

    print(f"Road type: {road.road_type} (length: {road.road_length_m}m, lanes: {road.num_lanes})")
    print(f"Obstacles spawned: {len(road.obstacles)} static 3D boxes")
    for o in road.obstacles:
        print(f"  • {o.obstacle_id}: (x={o.x}m, y={o.y}m, size={o.length}x{o.width}x{o.height}m)")
    print(f"Start Pose: x={road.start_pose[0]}m, y={road.start_pose[1]}m, yaw={road.start_pose[2]}rad")
    print(f"Goal Pose:  x={road.goal_pose[0]}m, y={road.goal_pose[1]}m, yaw={road.goal_pose[2]}rad")
    print("-" * 70)

    # 2. Run simulation with simple scripted controller
    def simple_scripted_driver(state, road_map):
        accel = 2.0 if state.velocity < 8.0 else 0.0
        steer = 0.10 if state.x < 30.0 else (-0.10 if state.x < 70.0 else 0.03)
        return ControlCommand(acceleration=accel, steering=steer)

    print(f"LiDAR Configuration: {sim.lidar.params.num_channels} vertical rings, {sim.lidar.total_beams} total rays/sweep")
    print("Running 4.0 seconds of simulation headlessly (200 steps @ 50 Hz)...")
    summary = sim.run_headless(duration_sec=4.0, controller_fn=simple_scripted_driver)

    print("\nSimulation Finished:")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")

    # Inspect the final LiDAR scan
    if sim.last_observation:
        noisy = sim.last_observation.noisy_scan
        gt = sim.last_observation.ground_truth_scan
        print("\nFinal LiDAR Sweep Telemetry:")
        print(f"  Total Beams Cast     : {len(noisy.ranges)}")
        print(f"  Ground Truth Returns : {int(np.sum(gt.valid_mask))}")
        print(f"  Noisy Valid Returns  : {int(np.sum(noisy.valid_mask))} (reflects range noise & dropouts)")
        valid_ranges = noisy.ranges[noisy.valid_mask]
        if len(valid_ranges) > 0:
            print(f"  Min Observed Range   : {np.min(valid_ranges):.2f} m")
            print(f"  Max Observed Range   : {np.max(valid_ranges):.2f} m")

        # 3. Export sample scan NPZ
        out_dir = PROJECT_ROOT / "runs"
        out_scan = out_dir / "sample_lidar_scan.npz"
        sim.lidar.save_scan_npz(noisy, str(out_scan))
        print(f"\nSaved sample scan NPZ: {out_scan}")

    # 4. Export CSV log
    out_dir = PROJECT_ROOT / "runs"
    out_csv = out_dir / "headless_demo_trajectory.csv"
    sim.logger.export_csv(out_csv)
    print(f"Saved trajectory log : {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()

