"""
Core Simulation Coordinator for Autonomous-World-Lab.
Orchestrates fixed-step physics integration, road boundary collision checks,
and state logging in headless or interactive viewer modes.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from autonomy.interfaces.types import VehicleState, ControlCommand, LidarObservation, DetectedObstacle, OccupancyGridMap
from autonomy.perception import PerceptionPipeline, PerceptionResult
from autonomy.mapping import OccupancyGrid
from autonomy.navigation import NavigationStack
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
        perception: Optional[PerceptionPipeline] = None,
        occupancy_grid: Optional[OccupancyGrid] = None,
        navigation: Optional[NavigationStack] = None,
        autopilot: bool = False,
        sensor_interval: int = 5,
    ):
        self.clock: SimulationClock = SimulationClock(dt_sec=dt_sec)
        self.road: RoadMap = road or RoadMap(road_type="straight", road_length_m=120.0)
        self.vehicle: KinematicBicycleModel = KinematicBicycleModel(
            params=vehicle_params or VehicleParameters()
        )
        self.lidar: LidarSensor = lidar or LidarSensor(
            params=LidarParameters(num_channels=32, h_resolution_rad=np.radians(1.5))
        )
        self.perception: PerceptionPipeline = perception or PerceptionPipeline()
        self.occupancy_grid: OccupancyGrid = occupancy_grid or OccupancyGrid()
        self.navigation: NavigationStack = navigation or NavigationStack()
        self.autopilot: bool = autopilot
        self.sensor_interval: int = max(1, int(sensor_interval))
        self.logger: SimulationLogger = SimulationLogger()

        self.collision: bool = False
        self.goal_reached: bool = False
        self.failure_reason: Optional[str] = None
        self.last_command: ControlCommand = ControlCommand()
        self.last_observation: Optional[LidarObservation] = None
        self.last_perception: Optional[PerceptionResult] = None
        self.last_map: Optional[OccupancyGridMap] = None

        # Initialize vehicle at road start pose
        sx, sy, syaw = self.road.start_pose
        self.vehicle.reset(x=sx, y=sy, yaw=syaw, velocity=0.0, timestamp=0.0)
        self.occupancy_grid.reset(center_x=sx, center_y=sy)
        self.navigation.reset(self.road, timestamp=0.0)
        self._update_perception_and_mapping()

    def reset(self) -> VehicleState:
        """Resets the simulation to initial conditions."""
        self.clock.reset(start_time_sec=0.0)
        sx, sy, syaw = self.road.start_pose
        self.collision = False
        self.goal_reached = False
        self.failure_reason = None
        self.last_command = ControlCommand()
        self.logger.clear()
        state = self.vehicle.reset(x=sx, y=sy, yaw=syaw, velocity=0.0, timestamp=0.0)
        self.occupancy_grid.reset(center_x=sx, center_y=sy)
        self.navigation.reset(self.road, timestamp=0.0)
        self._update_perception_and_mapping()
        return state

    def compute_autopilot_command(self, replan: bool = True) -> ControlCommand:
        """Runs NavigationStack plan + control step using latest perception & map."""
        return self.navigation.step(
            estimated_state=self.vehicle.state,
            perception=self.last_perception,
            cost_map=self.last_map,
            road_half_width=self.road.total_width_m / 2.0,
            goal_pose=self.road.goal_pose,
            replan=replan,
        )

    def _update_perception_and_mapping(self):
        """
        Runs perception and mapping pipelines on the latest sensor sweep.
        Perception only ingests noisy_scan (NO ground truth access).
        """
        self.last_observation = self.lidar.scan(self.vehicle.state, self.road.obstacles)
        self.last_perception = self.perception.process(
            self.last_observation.noisy_scan,
            self.vehicle.state,
        )
        self.occupancy_grid.update_from_scan(
            self.vehicle.state,
            self.last_perception.obstacle_points,
            self.last_perception.ground_points,
            recompute_cost=False,
        )
        self.occupancy_grid.update_from_detected_obstacles(
            self.last_perception.detected_obstacles,
            recompute_cost=True,
        )
        self.last_map = self.occupancy_grid.get_map_snapshot(timestamp=self.clock.time)

    def step(self, command: Optional[ControlCommand] = None) -> Tuple[VehicleState, Dict[str, Any]]:
        """
        Executes one discrete simulation step.
        If command is None and self.autopilot is True, invokes NavigationStack automatically.
        Returns:
            state: Updated ground-truth VehicleState
            status: Dictionary with step, time, collision, goal_reached, failure_reason
        """
        replan_due = (self.clock.step_count % self.sensor_interval == 0)
        if command is not None:
            cmd = command
        elif self.autopilot:
            cmd = self.compute_autopilot_command(replan=replan_due)
        else:
            cmd = ControlCommand()

        self.last_command = cmd

        # Advance physics integration by dt
        new_state = self.vehicle.step(cmd, self.clock.dt)
        self.clock.tick()

        # Update perception and mapping pipelines at sensor rate (plus step 1)
        if self.clock.step_count == 1 or (self.clock.step_count % self.sensor_interval == 0):
            self._update_perception_and_mapping()

        # Boundary and obstacle collision checks
        corners = self.vehicle.get_corners()
        boundary_coll = self.road.check_boundary_collision(corners)
        obstacle_coll = self.road.check_obstacle_collision(corners)
        self.collision = boundary_coll or obstacle_coll
        self.goal_reached = self.road.check_goal_reached(new_state.x, new_state.y)

        if obstacle_coll:
            self.failure_reason = "COLLISION_OBSTACLE"
        elif boundary_coll:
            self.failure_reason = "COLLISION_BOUNDARY"
        elif self.goal_reached:
            self.failure_reason = None
        elif self.autopilot:
            self.failure_reason = self.navigation.failure_reason
        else:
            self.failure_reason = None

        # Log entry
        self.logger.record(
            step=self.clock.step_count,
            state=new_state,
            command=cmd,
            collision=self.collision,
            goal_reached=self.goal_reached,
            failure_reason=self.failure_reason,
        )

        status = {
            "step": self.clock.step_count,
            "time": self.clock.time,
            "collision": self.collision,
            "boundary_collision": boundary_coll,
            "obstacle_collision": obstacle_coll,
            "goal_reached": self.goal_reached,
            "emergency_stop": cmd.emergency_stop,
            "failure_reason": self.failure_reason,
            "planner_status": self.navigation.planner_status if self.autopilot else "MANUAL",
            "lidar_points": int(np.sum(self.last_observation.noisy_scan.valid_mask)) if self.last_observation else 0,
            "detected_obstacles_count": len(self.last_perception.detected_obstacles) if self.last_perception else 0,
        }
        return new_state, status

    def run_headless(
        self,
        duration_sec: float = 10.0,
        controller_fn=None,
        use_autopilot: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes the simulation headlessly for the specified duration.
        controller_fn: Optional callable(VehicleState, RoadMap) -> ControlCommand
        use_autopilot: If True, runs the integrated NavigationStack closed-loop.
        Returns summary dictionary.
        """
        if use_autopilot is not None:
            self.autopilot = use_autopilot

        total_steps = int(duration_sec / self.clock.dt)
        for _ in range(total_steps):
            if controller_fn:
                cmd = controller_fn(self.vehicle.state, self.road)
            elif self.autopilot:
                cmd = None  # Trigger NavigationStack inside step()
            else:
                cmd = ControlCommand(acceleration=1.0, steering=0.0)

            _, status = self.step(cmd)
            if status["collision"] or status["goal_reached"]:
                break
            # Stop early if autopilot cleanly executed safe stop on blocked road
            if (
                self.autopilot
                and self.navigation.planner_status == "EMERGENCY_STOP"
                and abs(self.vehicle.state.velocity) < 0.02
                and self.clock.step_count > 10
            ):
                break

        return self.logger.summary()

    def snapshot(self) -> Dict[str, Any]:
        """
        Serializes current state snapshot for WebSocket / Three.js telemetry.
        """
        s = self.vehicle.state
        corners = self.vehicle.get_corners()

        global_pts = []
        if self.navigation.global_trajectory and self.navigation.global_trajectory.points:
            # Downsample slightly for lightweight WebSocket frames
            global_pts = [
                [round(p.x, 3), round(p.y, 3)]
                for p in self.navigation.global_trajectory.points[::2]
            ]
            last_g = self.navigation.global_trajectory.points[-1]
            global_pts.append([round(last_g.x, 3), round(last_g.y, 3)])

        local_pts = []
        if self.navigation.local_trajectory and self.navigation.local_trajectory.points:
            local_pts = [
                [round(p.x, 3), round(p.y, 3)]
                for p in self.navigation.local_trajectory.points
            ]

        lookahead = self.navigation.controller.last_lookahead_point

        return {
            "sim_time": round(self.clock.time, 4),
            "step": self.clock.step_count,
            "dt": self.clock.dt,
            "autopilot": self.autopilot,
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
                "failure_reason": self.failure_reason,
                "planner_status": self.navigation.planner_status if self.autopilot else "MANUAL",
                "detected_obstacles_count": len(self.last_perception.detected_obstacles) if self.last_perception else 0,
            },
            "road": self.road.to_dict(),
            "lidar": self.last_observation.noisy_scan.to_dict(max_points=1200) if self.last_observation else None,
            "lidar_ground_truth_returns": int(np.sum(self.last_observation.ground_truth_scan.valid_mask)) if self.last_observation else 0,
            "detected_obstacles": [obs.to_dict() for obs in self.last_perception.detected_obstacles] if self.last_perception else [],
            "cost_map": self.last_map.to_dict(downsample=3) if self.last_map else None,
            "planning": {
                "global_path": global_pts,
                "local_path": local_pts,
                "lookahead_point": [round(lookahead[0], 3), round(lookahead[1], 3)] if lookahead else None,
            },
        }

