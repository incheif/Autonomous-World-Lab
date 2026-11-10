"""
Core Simulation Coordinator for Autonomous-World-Lab.
Orchestrates fixed-step physics integration, road boundary collision checks,
and state logging in headless or interactive viewer modes.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from autonomy.interfaces.types import VehicleState, ControlCommand, LidarObservation
from simulator.clock import SimulationClock
from simulator.vehicle.kinematic_bicycle import KinematicBicycleModel, VehicleParameters
from simulator.world.road import RoadMap
from simulator.logger import SimulationLogger
from simulator.sensors.lidar import LidarSensor, LidarParameters


class Simulation:
    """
    Master simulation coordinator.
    Completely decoupled from visual rendering.
    """

    def __init__(
        self,
        dt_sec: float = 0.02,
        road: Optional[RoadMap] = None,
        vehicle_params: Optional[VehicleParameters] = None,
        lidar: Optional[LidarSensor] = None,
    ):
        self.clock: SimulationClock = SimulationClock(dt_sec=dt_sec)
        self.road: RoadMap = road or RoadMap(road_type="straight", road_length_m=120.0)
        self.vehicle: KinematicBicycleModel = KinematicBicycleModel(
            params=vehicle_params or VehicleParameters()
        )
        self.lidar: LidarSensor = lidar or LidarSensor(
            params=LidarParameters(num_channels=32, h_resolution_rad=np.radians(1.5))
        )
        self.logger: SimulationLogger = SimulationLogger()

        self.collision: bool = False
        self.goal_reached: bool = False
        self.last_command: ControlCommand = ControlCommand()
        self.last_observation: Optional[LidarObservation] = None

        # Initialize vehicle at road start pose
        sx, sy, syaw = self.road.start_pose
        self.vehicle.reset(x=sx, y=sy, yaw=syaw, velocity=0.0, timestamp=0.0)
        self.last_observation = self.lidar.scan(self.vehicle.state, self.road.obstacles)

    def reset(self) -> VehicleState:
        """Resets the simulation to initial conditions."""
        self.clock.reset(start_time_sec=0.0)
        sx, sy, syaw = self.road.start_pose
        self.collision = False
        self.goal_reached = False
        self.last_command = ControlCommand()
        self.logger.clear()
        state = self.vehicle.reset(x=sx, y=sy, yaw=syaw, velocity=0.0, timestamp=0.0)
        self.last_observation = self.lidar.scan(state, self.road.obstacles)
        return state

    def step(self, command: Optional[ControlCommand] = None) -> Tuple[VehicleState, Dict[str, Any]]:
        """
        Executes one discrete simulation step.
        Returns:
            state: Updated ground-truth VehicleState
            status: Dictionary with step, time, collision, goal_reached
        """
        cmd = command or ControlCommand()
        self.last_command = cmd

        # Advance physics integration by dt
        new_state = self.vehicle.step(cmd, self.clock.dt)
        self.clock.tick()

        # Perform 3D LiDAR observation sweep
        self.last_observation = self.lidar.scan(new_state, self.road.obstacles)

        # Boundary and obstacle collision checks
        corners = self.vehicle.get_corners()
        boundary_coll = self.road.check_boundary_collision(corners)
        obstacle_coll = self.road.check_obstacle_collision(corners)
        self.collision = boundary_coll or obstacle_coll
        self.goal_reached = self.road.check_goal_reached(new_state.x, new_state.y)

        # Log entry
        self.logger.record(
            step=self.clock.step_count,
            state=new_state,
            command=cmd,
            collision=self.collision,
            goal_reached=self.goal_reached,
        )

        status = {
            "step": self.clock.step_count,
            "time": self.clock.time,
            "collision": self.collision,
            "boundary_collision": boundary_coll,
            "obstacle_collision": obstacle_coll,
            "goal_reached": self.goal_reached,
            "emergency_stop": cmd.emergency_stop,
            "lidar_points": int(np.sum(self.last_observation.noisy_scan.valid_mask)) if self.last_observation else 0,
        }
        return new_state, status

    def run_headless(
        self,
        duration_sec: float = 10.0,
        controller_fn=None,
    ) -> Dict[str, Any]:
        """
        Executes the simulation headlessly for the specified duration.
        controller_fn: Optional callable(VehicleState, RoadMap) -> ControlCommand
        Returns summary dictionary.
        """
        total_steps = int(duration_sec / self.clock.dt)
        for _ in range(total_steps):
            if controller_fn:
                cmd = controller_fn(self.vehicle.state, self.road)
            else:
                cmd = ControlCommand(acceleration=1.0, steering=0.0)

            _, status = self.step(cmd)
            if status["collision"] or status["goal_reached"]:
                break

        return self.logger.summary()

    def snapshot(self) -> Dict[str, Any]:
        """
        Serializes current state snapshot for WebSocket / Three.js telemetry.
        """
        s = self.vehicle.state
        corners = self.vehicle.get_corners()

        return {
            "sim_time": round(self.clock.time, 4),
            "step": self.clock.step_count,
            "dt": self.clock.dt,
            "vehicle": {
                **s.to_dict(),
                "corners": [[round(cx, 3), round(cy, 3)] for cx, cy in corners],
                "wheelbase_m": self.vehicle.params.wheelbase_m,
                "length_m": self.vehicle.params.length_m,
                "width_m": self.vehicle.params.width_m,
                "height_m": self.vehicle.params.height_m,
                "track_width_m": self.vehicle.params.track_width_m,
            },
            "command": self.last_command.to_dict(),
            "status": {
                "collision": self.collision,
                "goal_reached": self.goal_reached,
                "emergency_stop": self.last_command.emergency_stop,
            },
            "road": self.road.to_dict(),
            "lidar": self.last_observation.noisy_scan.to_dict(max_points=1200) if self.last_observation else None,
            "lidar_ground_truth_returns": int(np.sum(self.last_observation.ground_truth_scan.valid_mask)) if self.last_observation else 0,
        }

