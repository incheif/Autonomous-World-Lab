# Autonomous-World-Lab Architecture & Technical Specification

> **Working Title:** Autonomous-World-Lab (OpenDriveLab)  
> **System Architecture:** Modular, Headless-First 3D Autonomous Driving & LiDAR Simulation Testbed  
> **Initial Milestone:** Phase 1 (Minimal 3D World, Kinematic Vehicle, Fixed Timestep Clock, Headless Tests, and WebGL Three.js Visualizer)

---

## 1. System Philosophy & Boundaries

Autonomous-World-Lab enforces a strict separation between **Ground Truth Simulation State**, **Sensor Observations**, **Autonomous Stack (Perception/Planning/Control)**, and the **3D Viewer**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION GROUND TRUTH                         │
│  • World Engine (Road mesh, true obstacles, boundaries)                │
│  • Vehicle Kinematics (True pose: x, y, yaw, velocity, steering)       │
│  • Simulation Clock (Fixed dt, deterministic ticks)                    │
└───────────────────────────┬────────────────────────────────────────────┘
                            │  Generates raw signals
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          SENSOR SIMULATION                             │
│  • 32-Beam LiDAR (Raycasting, noise, dropout, occlusion)               │
│  • Wheel Odometry / IMU (Bias, Gaussian noise, drift)                  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │  Noisy observations
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AUTONOMOUS AGENT STACK                          │
│  • Perception & Clustering (Point cloud -> Obstacle bounding boxes)    │
│  • State Estimation (EKF: Odometry + GNSS -> Estimated Pose & Cov)     │
│  • World Representation (2D Occupancy Grid / Cost Map + Inflation)    │
│  • Global & Local Planners (A*, Hybrid A*, Collision-free Waypoints)   │
│  • Controller (Pure Pursuit, Stanley, Acceleration/Braking)            │
└───────────────────────────┬────────────────────────────────────────────┘
                            │  ControlCommand (accel, steering, brake)
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 SIMULATION KINEMATICS UPDATE (dt)                      │
└───────────────────────────┬────────────────────────────────────────────┘
                            │  Telemetry Stream (WebSocket / JSON)
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     THREE.JS 3D WEBGL VIEWER                           │
│  • Decoupled visual subscriber (renders state at 60 FPS)               │
│  • Interactive Camera Rigs (Chase, Cockpit, Top-Down Orthographic)     │
│  • Telemetry HUD & Real-time Keyboard Control Bridge                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Coordinate Conventions & Units

Strict adherence to international standards prevents transform confusion:

* **Units:** International System of Units (SI) throughout all Python code:
  - Distance: Metres ($\text{m}$)
  - Angles: Radians ($\text{rad}$) internally (degrees used only for UI display)
  - Time: Seconds ($\text{s}$)
  - Velocity: Metres per second ($\text{m/s}$)
  - Acceleration: Metres per second squared ($\text{m/s}^2$)
  - Angular Velocity: Radians per second ($\text{rad/s}$)

* **World Coordinate System (ISO 8855 / SAE Standard Ground Vehicle):**
  - $+X$: Forward / East
  - $+Y$: Lateral / North (Left hand relative to forward vehicle heading)
  - $+Z$: Upward (Normal to road plane)
  - $\psi$ (Yaw): Angle counter-clockwise from positive $X$-axis around $+Z$.

* **Graphics Three.js Mapping:**
  - Three.js uses $+Y$ up, $+X$ right, $+Z$ toward viewer.
  - Transformation adapter: `three_x = sim_x`, `three_y = sim_z (elevation)`, `three_z = -sim_y`.

---

## 3. Vehicle Kinematic Model (Rear-Axle Reference)

For low-speed autonomous navigation, the classic **Kinematic Bicycle Model** referenced to the center of the rear axle provides high fidelity without unnecessary tire-slip complexity:

$$\begin{aligned}
\dot{x} &= v \cos(\psi) \\
\dot{y} &= v \sin(\psi) \\
\dot{\psi} &= \frac{v}{L} \tan(\delta) \\
\dot{v} &= a - c_{\text{drag}} \cdot v
\end{aligned}$$

Where:
* $(x, y)$: Rear axle position ($\text{m}$)
* $\psi$: Vehicle heading ($\text{rad}$)
* $v$: Longitudinal velocity ($\text{m/s}$)
* $\delta$: Front wheel steering angle ($\text{rad}$)
* $a$: Commanded acceleration / deceleration ($\text{m/s}^2$)
* $L$: Wheelbase length ($\text{m}$)
* $c_{\text{drag}}$: Rolling resistance / aerodynamic drag damping factor ($1/\text{s}$)

### Physical Constraints Enforced:
1. **Steering Saturation:** $|\delta| \le \delta_{\max}$ (default: $\pm 35^\circ = \pm 0.6108\text{ rad}$)
2. **Steering Slew Rate:** $|\dot{\delta}| \le \omega_{\max}$ (default: $60^\circ/\text{s} = 1.047\text{ rad/s}$)
3. **Acceleration Envelope:** $a_{\min} \le a \le a_{\max}$ (default: $-5.0\text{ m/s}^2 \le a \le +3.0\text{ m/s}^2$)
4. **Velocity Bounds:** $v_{\min} \le v \le v_{\max}$ (default: $-3.0\text{ m/s} \le v \le +20.0\text{ m/s}$)

---

## 4. Simulation Clock & Determinism

The simulator executes in fixed discrete time increments:
$$\Delta t = 0.02\text{ s} \quad (50\text{ Hz})$$

* **Deterministic Replay:** Given an identical initial seed and identical `ControlCommand` array, the simulation output trajectories must match to within floating-point epsilon ($< 10^{-7}\text{ m}$).
* **Decoupled Viewer:** The simulation can run at $1000\times$ wall-clock speed headlessly for unit tests and benchmarks, or synchronized to $50\text{ Hz}$ when streaming to the Three.js viewer.

---

## 5. Telemetry & Streaming API

* **Protocol:** WebSocket (`ws://localhost:8080/ws`) + REST API (`http://localhost:8080/api`).
* **Frame Format (JSON payload at 30–50 Hz):**
  - `sim_time`: Current simulation timestamp ($\text{s}$).
  - `step`: Current step index.
  - `ground_truth`: `{x, y, z, yaw, velocity, steering, acceleration}`
  - `road`: Road centerlines, boundary polygons, start marker, goal marker.
  - `command`: Current active control command (`accel, steering, brake`).
  - `status`: `{collision: bool, emergency_stop: bool, goal_reached: bool}`
