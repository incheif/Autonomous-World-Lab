/**
 * Autonomous-World-Lab 3D WebGL Simulation Viewer.
 * Powered by Three.js. Connects via WebSocket to the Python simulation engine.
 */

// Coordinate Transform:
// Sim: +X forward, +Y lateral (left), +Z up
// Three.js: +X right, +Y up, +Z toward camera
// Mapping: Three.x = Sim.x, Three.y = Sim.z, Three.z = -Sim.y
function simToThree(x, y, z = 0) {
  return new THREE.Vector3(x, z, -y);
}

class SimulationViewer {
  constructor() {
    this.container = document.getElementById("canvas-container");
    this.ws = null;
    this.cameraMode = "chase"; // 'chase', 'cockpit', 'drone', 'orbit'
    this.lastSnapshot = null;

    // Keyboard state
    this.keys = {
      up: false,
      down: false,
      left: false,
      right: false,
      space: false,
    };

    // Actuator command state
    this.accel = 0.0;
    this.steer = 0.0;
    this.brake = 0.0;
    this.estop = false;

    // Trajectory breadcrumbs
    this.maxTrailPoints = 600;
    this.trailPoints = [];
    this.trailLine = null;

    this.initThree();
    this.initControls();
    this.initWebSocket();
    this.initEventListeners();
    this.animate();
  }

  initThree() {
    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0c10);
    this.scene.fog = new THREE.FogExp2(0x0a0c10, 0.008);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(-15, 10, 15);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    this.orbitControls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.orbitControls.enableDamping = true;
    this.orbitControls.dampingFactor = 0.05;
    this.orbitControls.maxPolarAngle = Math.PI / 2 - 0.02;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
    this.scene.add(ambientLight);

    this.sunLight = new THREE.DirectionalLight(0xffffff, 0.85);
    this.sunLight.position.set(30, 50, 40);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 2048;
    this.sunLight.shadow.mapSize.height = 2048;
    this.sunLight.shadow.camera.near = 0.5;
    this.sunLight.shadow.camera.far = 250;
    const d = 60;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.scene.add(this.sunLight);

    // Ground Grid
    const groundGrid = new THREE.GridHelper(400, 80, 0x1e293b, 0x111827);
    groundGrid.position.y = -0.01;
    this.scene.add(groundGrid);

    // Groups
    this.roadGroup = new THREE.Group();
    this.scene.add(this.roadGroup);

    this.vehicleGroup = new THREE.Group();
    this.scene.add(this.vehicleGroup);

    this.buildVehicleMesh();
    this.initTrail();
  }

  buildVehicleMesh() {
    // Kinematic Vehicle Group
    this.carRoot = new THREE.Group();

    // Body dimensions (default standard sedan)
    const length = 4.5;
    const width = 1.8;
    const height = 1.45;
    const wheelbase = 2.8;
    const wheelRadius = 0.33;
    const wheelWidth = 0.24;

    // 1. Lower Chassis
    const chassisGeo = new THREE.BoxGeometry(length, height * 0.45, width);
    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7, // Metallic Cyan Blue
      metalness: 0.6,
      roughness: 0.3,
    });
    const chassis = new THREE.Mesh(chassisGeo, chassisMat);
    chassis.position.set(wheelbase / 2, wheelRadius + height * 0.22, 0);
    chassis.castShadow = true;
    this.carRoot.add(chassis);

    // 2. Cabin / Glass Roof
    const cabinGeo = new THREE.BoxGeometry(length * 0.55, height * 0.45, width * 0.85);
    const cabinMat = new THREE.MeshStandardMaterial({
      color: 0x111827,
      metalness: 0.9,
      roughness: 0.1,
    });
    const cabin = new THREE.Mesh(cabinGeo, cabinMat);
    cabin.position.set(wheelbase * 0.35, wheelRadius + height * 0.65, 0);
    cabin.castShadow = true;
    this.carRoot.add(cabin);

    // 3. Headlights & Taillights
    const lightGeo = new THREE.BoxGeometry(0.1, 0.12, 0.25);
    const headMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const tailMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });

    const flLight = new THREE.Mesh(lightGeo, headMat);
    flLight.position.set(length - 0.75, wheelRadius + 0.35, -width * 0.38);
    const frLight = new THREE.Mesh(lightGeo, headMat);
    frLight.position.set(length - 0.75, wheelRadius + 0.35, width * 0.38);
    this.carRoot.add(flLight, frLight);

    const rlLight = new THREE.Mesh(lightGeo, tailMat);
    rlLight.position.set(-0.75, wheelRadius + 0.35, -width * 0.38);
    const rrLight = new THREE.Mesh(lightGeo, tailMat);
    rrLight.position.set(-0.75, wheelRadius + 0.35, width * 0.38);
    this.carRoot.add(rlLight, rrLight);

    // 4. Heading Direction Arrow
    const arrowGeo = new THREE.ConeGeometry(0.25, 0.8, 16);
    arrowGeo.rotateZ(-Math.PI / 2);
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const arrow = new THREE.Mesh(arrowGeo, arrowMat);
    arrow.position.set(length * 0.9, wheelRadius + 0.45, 0);
    this.carRoot.add(arrow);

    // 5. Wheels
    this.wheels = {
      frontLeft: this.createWheel(wheelRadius, wheelWidth, true),
      frontRight: this.createWheel(wheelRadius, wheelWidth, true),
      rearLeft: this.createWheel(wheelRadius, wheelWidth, false),
      rearRight: this.createWheel(wheelRadius, wheelWidth, false),
    };

    // Position wheels relative to rear axle origin (0, 0, 0)
    const halfTrack = width / 2.0;

    this.wheels.rearLeft.root.position.set(0, wheelRadius, -halfTrack);
    this.wheels.rearRight.root.position.set(0, wheelRadius, halfTrack);
    this.wheels.frontLeft.root.position.set(wheelbase, wheelRadius, -halfTrack);
    this.wheels.frontRight.root.position.set(wheelbase, wheelRadius, halfTrack);

    this.carRoot.add(
      this.wheels.rearLeft.root,
      this.wheels.rearRight.root,
      this.wheels.frontLeft.root,
      this.wheels.frontRight.root
    );

    this.vehicleGroup.add(this.carRoot);
  }

  createWheel(radius, width, isSteering = false) {
    const root = new THREE.Group();
    const steerPivot = new THREE.Group();
    root.add(steerPivot);

    const wheelGeo = new THREE.CylinderGeometry(radius, radius, width, 24);
    wheelGeo.rotateX(Math.PI / 2);
    const wheelMat = new THREE.MeshStandardMaterial({
      color: 0x1f2937,
      roughness: 0.8,
    });
    const tire = new THREE.Mesh(wheelGeo, wheelMat);
    tire.castShadow = true;
    steerPivot.add(tire);

    // Hubcap
    const capGeo = new THREE.CylinderGeometry(radius * 0.5, radius * 0.5, width * 1.05, 12);
    capGeo.rotateX(Math.PI / 2);
    const capMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.8 });
    const cap = new THREE.Mesh(capGeo, capMat);
    steerPivot.add(cap);

    return { root, steerPivot, tire, isSteering, radius };
  }

  initTrail() {
    const maxPoints = this.maxTrailPoints;
    const positions = new Float32Array(maxPoints * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const material = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      linewidth: 2,
      transparent: true,
      opacity: 0.75,
    });
    this.trailLine = new THREE.Line(geometry, material);
    this.scene.add(this.trailLine);
  }

  updateTrail(pos) {
    this.trailPoints.push(pos.clone());
    if (this.trailPoints.length > this.maxTrailPoints) {
      this.trailPoints.shift();
    }
    const positions = this.trailLine.geometry.attributes.position.array;
    for (let i = 0; i < this.trailPoints.length; i++) {
      positions[i * 3] = this.trailPoints[i].x;
      positions[i * 3 + 1] = this.trailPoints[i].y + 0.05;
      positions[i * 3 + 2] = this.trailPoints[i].z;
    }
    this.trailLine.geometry.setDrawRange(0, this.trailPoints.length);
    this.trailLine.geometry.attributes.position.needsUpdate = true;
  }

  rebuildRoad(roadData) {
    while (this.roadGroup.children.length > 0) {
      const obj = this.roadGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      this.roadGroup.remove(obj);
    }

    if (!roadData || !roadData.centerline) return;

    const left = roadData.left_boundary;
    const right = roadData.right_boundary;
    const center = roadData.centerline;
    const n = center.length;

    // 1. Asphalt Road Mesh
    const vertices = [];
    const indices = [];

    for (let i = 0; i < n; i++) {
      const pL = simToThree(left[i][0], left[i][1], 0);
      const pR = simToThree(right[i][0], right[i][1], 0);
      vertices.push(pL.x, pL.y, pL.z);
      vertices.push(pR.x, pR.y, pR.z);

      if (i < n - 1) {
        const row = i * 2;
        indices.push(row, row + 1, row + 2);
        indices.push(row + 1, row + 3, row + 2);
      }
    }

    const roadGeo = new THREE.BufferGeometry();
    roadGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    roadGeo.setIndex(indices);
    roadGeo.computeVertexNormals();

    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x1e222b,
      roughness: 0.9,
      metalness: 0.1,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.receiveShadow = true;
    this.roadGroup.add(roadMesh);

    // 2. White Shoulder Lines
    const leftLinePts = left.map((p) => simToThree(p[0], p[1], 0.02));
    const rightLinePts = right.map((p) => simToThree(p[0], p[1], 0.02));

    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
    const lGeo = new THREE.BufferGeometry().setFromPoints(leftLinePts);
    const rGeo = new THREE.BufferGeometry().setFromPoints(rightLinePts);
    this.roadGroup.add(new THREE.Line(lGeo, lineMat));
    this.roadGroup.add(new THREE.Line(rGeo, lineMat));

    // 3. Center Dashed Line
    const centerPts = center.map((p) => simToThree(p[0], p[1], 0.02));
    const cGeo = new THREE.BufferGeometry().setFromPoints(centerPts);
    const cMat = new THREE.LineDashedMaterial({
      color: 0xfacc15, // Yellow dashes
      dashSize: 2.0,
      gapSize: 1.5,
    });
    const centerLine = new THREE.Line(cGeo, cMat);
    centerLine.computeLineDistances();
    this.roadGroup.add(centerLine);

    // 4. Start & Goal Markers
    if (roadData.start_pose) {
      const sp = simToThree(roadData.start_pose[0], roadData.start_pose[1], 0.03);
      const startGeo = new THREE.PlaneGeometry(roadData.total_width_m, 1.2);
      startGeo.rotateX(-Math.PI / 2);
      const startMat = new THREE.MeshBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.6 });
      const startMarker = new THREE.Mesh(startGeo, startMat);
      startMarker.position.copy(sp);
      this.roadGroup.add(startMarker);
    }

    if (roadData.goal_pose) {
      const gp = simToThree(roadData.goal_pose[0], roadData.goal_pose[1], 0.03);
      const goalGeo = new THREE.PlaneGeometry(roadData.total_width_m, 2.0);
      goalGeo.rotateX(-Math.PI / 2);
      const goalMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.7 });
      const goalMarker = new THREE.Mesh(goalGeo, goalMat);
      goalMarker.position.copy(gp);
      this.roadGroup.add(goalMarker);
    }
  }

  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const pill = document.getElementById("connection-pill");
    const pillText = document.getElementById("connection-text");

    const connect = () => {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        pill.className = "pill pill-online";
        pillText.textContent = "CONNECTED (50Hz)";
      };

      this.ws.onmessage = (event) => {
        try {
          const snapshot = jsonParseSafe(event.data);
          if (snapshot) this.handleSnapshot(snapshot);
        } catch (e) {
          console.error("WS Parse error", e);
        }
      };

      this.ws.onclose = () => {
        pill.className = "pill pill-offline";
        pillText.textContent = "RECONNECTING...";
        setTimeout(connect, 1500);
      };

      this.ws.onerror = () => {
        this.ws.close();
      };
    };

    connect();

    // 50Hz Client input stream loop
    setInterval(() => {
      this.sendControlCommand();
    }, 20);
  }

  sendControlCommand() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    // Process keys to actuator targets
    let targetAccel = 0.0;
    let targetSteer = 0.0;
    let targetBrake = 0.0;

    if (this.keys.up) targetAccel += 3.0;
    if (this.keys.down) {
      if (this.lastSnapshot && this.lastSnapshot.vehicle.velocity > 0.5) {
        targetBrake = 1.0;
      } else {
        targetAccel -= 2.0; // Reverse
      }
    }
    if (this.keys.left) targetSteer += 0.6108; // 35 deg
    if (this.keys.right) targetSteer -= 0.6108;

    const payload = {
      action: "control",
      accel: targetAccel,
      steer: targetSteer,
      brake: targetBrake,
      estop: this.keys.space,
    };
    this.ws.send(JSON.stringify(payload));
  }

  handleSnapshot(s) {
    this.lastSnapshot = s;

    // Rebuild road if not built or changed
    if (!this.currentRoadType || this.currentRoadType !== s.road.road_type) {
      this.currentRoadType = s.road.road_type;
      this.rebuildRoad(s.road);
    }

    const v = s.vehicle;
    // Update car position & heading
    // Sim yaw is counter-clockwise around +Z. Three.js rotation around Y is clockwise (-yaw).
    const threePos = simToThree(v.x, v.y, 0);
    this.carRoot.position.copy(threePos);
    this.carRoot.rotation.y = v.yaw;

    // Steer front wheels
    // Steering angle delta rotates front wheels around local Y
    this.wheels.frontLeft.steerPivot.rotation.y = v.steering_angle;
    this.wheels.frontRight.steerPivot.rotation.y = v.steering_angle;

    // Spin tires with velocity
    const rollDelta = (v.velocity * s.dt) / this.wheels.rearLeft.radius;
    this.wheels.rearLeft.tire.rotation.z -= rollDelta;
    this.wheels.rearRight.tire.rotation.z -= rollDelta;
    this.wheels.frontLeft.tire.rotation.z -= rollDelta;
    this.wheels.frontRight.tire.rotation.z -= rollDelta;

    // Update trail
    this.updateTrail(threePos);

    // Update camera rig
    this.updateCamera(v);

    // Update HUD metrics
    this.updateHUD(s);
  }

  updateCamera(v) {
    if (this.cameraMode === "orbit") return;

    const cosY = Math.cos(v.yaw);
    const sinY = Math.sin(v.yaw);

    if (this.cameraMode === "chase") {
      const dist = 8.5;
      const height = 3.6;
      const camX = v.x - dist * cosY;
      const camY = v.y - dist * sinY;
      const targetX = v.x + 3.0 * cosY;
      const targetY = v.y + 3.0 * sinY;

      const p = simToThree(camX, camY, height);
      const lookAt = simToThree(targetX, targetY, 0.8);

      this.camera.position.lerp(p, 0.12);
      this.camera.lookAt(lookAt);
    } else if (this.cameraMode === "cockpit") {
      const camX = v.x + 1.2 * cosY;
      const camY = v.y + 1.2 * sinY;
      const targetX = v.x + 15.0 * cosY;
      const targetY = v.y + 15.0 * sinY;

      const p = simToThree(camX, camY, 1.25);
      const lookAt = simToThree(targetX, targetY, 1.1);

      this.camera.position.copy(p);
      this.camera.lookAt(lookAt);
    } else if (this.cameraMode === "drone") {
      const p = simToThree(v.x, v.y, 45.0);
      const lookAt = simToThree(v.x, v.y, 0);
      this.camera.position.lerp(p, 0.1);
      this.camera.lookAt(lookAt);
    }
  }

  updateHUD(s) {
    const v = s.vehicle;
    const speedKmh = v.velocity * 3.6;
    const steerDeg = (v.steering_angle * 180) / Math.PI;
    const yawDeg = (v.yaw * 180) / Math.PI;

    document.getElementById("val-speed").innerHTML = `${speedKmh.toFixed(1)} <small>km/h</small>`;
    document.getElementById("val-steer").textContent = `${steerDeg.toFixed(1)}°`;
    document.getElementById("val-accel").innerHTML = `${v.acceleration.toFixed(2)} <small>m/s²</small>`;
    document.getElementById("val-yaw").textContent = `${yawDeg.toFixed(1)}°`;
    document.getElementById("val-pos").innerHTML = `${v.x.toFixed(2)}, ${v.y.toFixed(2)} <small>m</small>`;

    document.getElementById("val-time").innerHTML = `${s.sim_time.toFixed(2)} <small>s</small>`;
    document.getElementById("val-step").textContent = s.step;

    // Badges
    const collPill = document.getElementById("collision-pill");
    const goalPill = document.getElementById("goal-pill");

    collPill.style.display = s.status.collision ? "flex" : "none";
    goalPill.style.display = s.status.goal_reached ? "flex" : "none";
  }

  initControls() {
    // Camera toggle buttons
    document.querySelectorAll(".btn-cam").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-cam").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.cameraMode = e.target.dataset.cam;
        this.orbitControls.enabled = this.cameraMode === "orbit";
      });
    });

    // Road selector buttons
    document.querySelectorAll(".btn-road").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-road").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        const roadType = e.target.dataset.road;
        this.resetSimulation(roadType);
      });
    });

    // Reset button
    document.getElementById("btn-reset").addEventListener("click", () => {
      this.resetSimulation(this.currentRoadType || "s_curve");
    });

    // E-Stop button
    document.getElementById("btn-estop").addEventListener("click", () => {
      this.keys.space = true;
      setTimeout(() => (this.keys.space = false), 500);
    });
  }

  resetSimulation(roadType = "s_curve") {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: "reset", road_type: roadType }));
    }
    this.trailPoints = [];
    if (this.trailLine) this.trailLine.geometry.setDrawRange(0, 0);
  }

  initEventListeners() {
    window.addEventListener("resize", () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Keyboard controls
    window.addEventListener("keydown", (e) => {
      const code = e.code;
      if (code === "KeyW" || code === "ArrowUp") this.keys.up = true;
      if (code === "KeyS" || code === "ArrowDown") this.keys.down = true;
      if (code === "KeyA" || code === "ArrowLeft") this.keys.left = true;
      if (code === "KeyD" || code === "ArrowRight") this.keys.right = true;
      if (code === "Space") {
        this.keys.space = true;
        e.preventDefault();
      }
      if (code === "KeyR") this.resetSimulation(this.currentRoadType || "s_curve");
    });

    window.addEventListener("keyup", (e) => {
      const code = e.code;
      if (code === "KeyW" || code === "ArrowUp") this.keys.up = false;
      if (code === "KeyS" || code === "ArrowDown") this.keys.down = false;
      if (code === "KeyA" || code === "ArrowLeft") this.keys.left = false;
      if (code === "KeyD" || code === "ArrowRight") this.keys.right = false;
      if (code === "Space") this.keys.space = false;
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.cameraMode === "orbit") {
      this.orbitControls.update();
    }
    this.renderer.render(this.scene, this.camera);
  }
}

function jsonParseSafe(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new SimulationViewer();
});
