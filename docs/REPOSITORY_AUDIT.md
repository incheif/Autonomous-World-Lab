# Repository Audit & Reuse Assessment

This report documents the local inspection of the four reference repositories located in `c:\Users\epost\OneDrive\Desktop\New folder\robotics\`, their branches, commit states, licenses, and specific components assessed for reuse in **Autonomous-World-Lab**.

---

## 1. Local Clones Inspected

| Repository | Current Branch | Head Commit SHA | License | Primary Domain |
| :--- | :--- | :--- | :--- | :--- |
| **`AutonomousVehicleControlBeginnersGuide`** | `main` | `2bddb87` | MIT | 2D AV Navigation, Control, Localization & Perception |
| **`3d-world-simulator`** | `main` | `9152640` | Unlicensed (Research) | Multimodal 3D Particle-Voxel World Dynamics |
| **`ReLift`** | `main` | `98b0425` | MIT | Dense 3D Point Tracking as Trajectory Repair |
| **`hyperdrone`** | `main` | `01eee73` | MIT | High-Throughput Aerial RL Simulation Engine |

---

## 2. Detailed Assessment by Repository

### 2.1 `AutonomousVehicleControlBeginnersGuide`
* **Local Path:** `c:\Users\epost\OneDrive\Desktop\New folder\robotics\AutonomousVehicleControlBeginnersGuide`
* **Implementation Stack:** Python 3.13, NumPy, SciPy, Matplotlib.
* **Component Evaluation:**
  - `src/components/vehicle/`: Implements `FourWheelsVehicle`, `VehicleSpecification`, body, tires, and axles. Contains solid geometrical relationships (wheelbase $L = f\_len + r\_len$, tread width, steering angles).
  - `src/components/state/state.py`: Formulates discrete-time state transitions for $[x, y, \psi, v]^T$.
  - `src/components/control/`: Pure Pursuit, Stanley, LQR, MPC. Clean reference for control math.
  - `src/components/perception/point_cloud_rectangle_fitting/`: Oriented Bounding Box (OBB) fitting algorithms.
* **Reuse Decision:**
  - **Verdict:** **Primary Algorithmic Reference.**
  - **Action:** Adapt the mathematical formulations of the kinematic bicycle model and baseline tracking controllers behind Autonomous-World-Lab's clean, typed interfaces. Do not import the Matplotlib-coupled `FourWheelsVehicle` class directly, as it tightly couples animation plotting objects (`axes`, `elems`) with physics updates.

### 2.2 `3d-world-simulator`
* **Local Path:** `c:\Users\epost\OneDrive\Desktop\New folder\robotics\3d-world-simulator`
* **Implementation Stack:** PyTorch, Conda (`environment.yml`), CUDA-dependent particle-voxel networks.
* **Component Evaluation:**
  - Designed for tabletop robotic manipulation (cloth, box, Push-T, plug insertion) using hybrid particle-voxel autoencoders and action-conditioned latent dynamics.
  - No autonomous driving vehicle models, road networks, or LiDAR raycasters exist.
* **Reuse Decision:**
  - **Verdict:** **Deferred to Future Research Phase (Phase 7).**
  - **Action:** Do not create a dependency on this heavy PyTorch stack for Phase 0/1. Study its latent dynamics formulation later when building uncertainty-aware world models.

### 2.3 `hyperdrone`
* **Local Path:** `c:\Users\epost\OneDrive\Desktop\New folder\robotics\hyperdrone`
* **Implementation Stack:** C++ / RL-Tools / Python / WebAssembly.
* **Component Evaluation:**
  - Drone dynamics simulator built for high-throughput reinforcement learning.
  - Highly effective architectural concept: strict separation of headless physical simulation clocks from asynchronous browser-based WebGL visualizers.
* **Reuse Decision:**
  - **Verdict:** **Architectural Pattern Reference.**
  - **Action:** Adopt the pattern of a headless fixed-timestep simulation clock (`simulator/clock.py`) communicating with a WebGL (Three.js) viewer over WebSockets, while implementing ground-vehicle kinematic bicycle dynamics rather than quadrotor physics.

### 2.4 `ReLift`
* **Local Path:** `c:\Users\epost\OneDrive\Desktop\New folder\robotics\ReLift`
* **Implementation Stack:** PyTorch, PyTorch3D, CoTracker, DINO, video transformers.
* **Component Evaluation:**
  - Dense monocular 3D point tracking from video sequences.
  - Requires large GPU models and extensive dependencies.
* **Reuse Decision:**
  - **Verdict:** **Not Reused in Initial Phases.**
  - **Action:** Exclude from the core simulation stack. Relevant only if future camera perception experiments require dense video-based tracking.

---

## 3. Reuse Policy Summary

1. **Zero Runtime Coupling:** Autonomous-World-Lab maintains zero dependencies on `AERO` or the four external repos.
2. **Attribution:** All mathematical equations and algorithmic inspirations adapted from Shisato Yano's work are cited in `THIRD_PARTY_NOTICES.md`.
3. **Clean Decoupling:** Simulation physics must be 100% headless, deterministic, and testable without Three.js or browser UI running.
