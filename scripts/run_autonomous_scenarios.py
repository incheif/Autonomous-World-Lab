"""
Phase 4 Autonomous Navigation Scenario Runner.
Executes all deterministic scenarios headlessly through the complete closed loop:
Sense (32-beam LiDAR) -> Perceive -> Map -> Plan (A* + Local Avoidance) -> Control (Pure Pursuit) -> Update.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.simulation import Simulation
from simulator.scenarios import SCENARIOS


def main():
    print("=" * 84)
    print(" AUTONOMOUS-WORLD-LAB · PHASE 4 CLOSED-LOOP NAVIGATION SCENARIO SUITE")
    print("=" * 84)

    all_passed = True
    out_dir = PROJECT_ROOT / "runs"

    for scenario_id, spec in SCENARIOS.items():
        t0 = time.time()
        sim = Simulation(dt_sec=0.02, road=spec.builder(), autopilot=True)
        summary = sim.run_headless(duration_sec=spec.max_duration_sec)
        elapsed = time.time() - t0

        # Check acceptance expectations
        passed = (
            summary["collision_occurred"] is False
            and summary["goal_reached"] == spec.expect_goal_reached
        )
        if spec.expect_safe_stop:
            passed = passed and (summary["failure_reason"] == "NO_SAFE_ROUTE") and (abs(sim.vehicle.state.velocity) < 0.05)

        if not passed:
            all_passed = False

        status_tag = "PASS" if passed else "FAIL"
        print(
            f"[{status_tag}] {spec.scenario_id:25s} | "
            f"Time: {summary['duration_sec']:5.2f}s ({elapsed:4.2f}s real) | "
            f"Goal: {str(summary['goal_reached']):5s} | "
            f"Coll: {str(summary['collision_occurred']):5s} | "
            f"Reason: {str(summary['failure_reason']):14s} | "
            f"Pos: ({summary['final_x']:6.2f}, {summary['final_y']:5.2f})"
        )

        # Save trajectory log for inspection
        sim.logger.export_csv(out_dir / f"phase4_{scenario_id}.csv")

    print("-" * 84)
    print(f"Overall Phase 4 Suite Status: {'ALL SCENARIOS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 84)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

