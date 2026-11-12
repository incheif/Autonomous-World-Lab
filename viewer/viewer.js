/**
 * Autonomous-World-Lab 3D WebGL Simulation Viewer.
 * Powered by Three.js. Connects via WebSocket to the Python simulation engine.
 * Features:
 *  - 60/120 FPS continuous interpolation (lerp & slerp) for buttery-smooth driving
 *  - Procedural 3D City Skyline (multi-tier skyscrapers, window lights, rooftop HVACs)
 *  - Dual-Lane Roadways with yellow dashed centerline, white shoulder boundaries
 *  - Elevated textured sidewalks/footpaths and raised concrete curbs
 *  - Roadside 3D Trees (foliage canopies & wood trunks) and Streetlights with warm glow
 *  - Intelligent Dynamic Traffic & NPC Vehicles with brake lights and headlights
 *  - Real-time 32-beam LiDAR point cloud with Turbo elevation colormap
 *  - 3D Oriented Bounding Boxes (OBB) from Euclidean clustering & L-shape fitting
 *  - Live Bird's-Eye-View (BEV) Cost Map minimap
 *  - Global A* planned route & Local Pure Pursuit lookahead marker
 */

// Coordinate Transform:
// Sim: +X forward, +Y lateral (left), +Z up
// Three.js: +X right, +Y up, +Z toward camera
// Mapping: Three.x = Sim.x, Three.y = Sim.z, Three.z = -Sim.y
function simToThree(x, y, z = 0) {
  return new THREE.Vector3(x, z, -y);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  let diff = (b - a) % (2 * Math.PI);
  if (diff < -Math.PI) diff += 2 * Math.PI;
  if (diff > Math.PI) diff -= 2 * Math.PI;
  return a + diff * t;
}

class SimulationViewer {
  constructor() {
    this.container = document.getElementById("canvas-container");
    this.ws = null;
    this.cameraMode = "chase"; // 'chase', 'cockpit', 'drone', 'orbit'
    this.lastSnapshot = null;

    // Smooth Interpolation State Buffers
    this.currState = null;
    this.targetState = null;
    this.stateLerpFactor = 0.25;

    // Traffic Mesh Cache
    this.trafficMeshes = new Map();

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

    // 32-Beam LiDAR Point Cloud
    this.showLidar = true;
    this.maxLidarPoints = 2500;
    this.lidarPointsMesh = null;

    // Perception & Mapping
    this.showBoundingBoxes = true;
    this.detectedBoxGroup = null;
    this.minimapCanvas = document.getElementById("minimap-canvas");
    this.minimapCtx = this.minimapCanvas ? this.minimapCanvas.getContext("2d") : null;

    // Navigation & Planning (Phase 4)
    this.autopilot = true;
    this.showPlannedPaths = true;
    this.roadSignature = null;
    this.globalPathLine = null;
    this.localPathLine = null;
    this.lookaheadMarker = null;

    this.initThree();
    this.initControls();
    this.initWebSocket();
    this.initEventListeners();
    this.animate();
  }

  initThree() {
    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090e);
    this.scene.fog = new THREE.FogExp2(0x07090e, 0.006);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.1,
      1200
    );
    this.camera.position.set(-15, 10, 15);

    // 3. WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    this.orbitControls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.orbitControls.enableDamping = true;
    this.orbitControls.dampingFactor = 0.06;
    this.orbitControls.maxPolarAngle = Math.PI / 2 - 0.02;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xdbeafe, 0.55);
    this.scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 0.35);
    this.scene.add(hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfff7ed, 1.1);
    this.sunLight.position.set(40, 70, 50);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 2048;
    this.sunLight.shadow.mapSize.height = 2048;
    this.sunLight.shadow.camera.near = 0.5;
    this.sunLight.shadow.camera.far = 300;
    const d = 80;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.scene.add(this.sunLight);

    // Ground Grid
    const groundGrid = new THREE.GridHelper(500, 100, 0x1e293b, 0x0f172a);
    groundGrid.position.y = -0.01;
    this.scene.add(groundGrid);

    // Dedicated Scene Groups
    this.sceneryGroup = new THREE.Group();
    this.scene.add(this.sceneryGroup);

    this.roadGroup = new THREE.Group();
    this.scene.add(this.roadGroup);

    this.obstacleGroup = new THREE.Group();
    this.scene.add(this.obstacleGroup);

    this.trafficGroup = new THREE.Group();
    this.scene.add(this.trafficGroup);

    this.detectedBoxGroup = new THREE.Group();
    this.scene.add(this.detectedBoxGroup);

    this.vehicleGroup = new THREE.Group();
    this.scene.add(this.vehicleGroup);

    this.buildVehicleMesh();
    this.initTrail();
    this.initLidarPointCloud();
    this.initPlanningVisuals();
  }

  initPlanningVisuals() {
    // Global A* Route (Indigo / Sky Ribbon)
    const gGeo = new THREE.BufferGeometry();
    const gMat = new THREE.LineBasicMaterial({ color: 0x6366f1, linewidth: 2, transparent: true, opacity: 0.75 });
    this.globalPathLine = new THREE.Line(gGeo, gMat);
    this.globalPathLine.frustumCulled = false;
    this.scene.add(this.globalPathLine);

    // Local Avoidance Trajectory (Bright Emerald Green)
    const lGeo = new THREE.BufferGeometry();
    const lMat = new THREE.LineBasicMaterial({ color: 0x22c55e, linewidth: 3 });
    this.localPathLine = new THREE.Line(lGeo, lMat);
    this.localPathLine.frustumCulled = false;
    this.scene.add(this.localPathLine);

    // Pure Pursuit Lookahead Marker (Magenta Sphere)
    const targetGeo = new THREE.SphereGeometry(0.35, 16, 16);
    const targetMat = new THREE.MeshBasicMaterial({ color: 0xec4899 });
    this.lookaheadMarker = new THREE.Mesh(targetGeo, targetMat);
    this.lookaheadMarker.visible = false;
    this.scene.add(this.lookaheadMarker);
  }

  initLidarPointCloud() {
    const maxPts = this.maxLidarPoints;
    const positions = new Float32Array(maxPts * 3);
    const colors = new Float32Array(maxPts * 3);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.32,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
    });

    this.lidarPointsMesh = new THREE.Points(geometry, material);
    this.lidarPointsMesh.frustumCulled = false;
    this.scene.add(this.lidarPointsMesh);
  }

  buildVehicleMesh() {
    this.carRoot = new THREE.Group();

    const length = 4.5;
    const width = 1.8;
    const height = 1.45;
    const wheelbase = 2.8;
    const wheelRadius = 0.33;
    const wheelWidth = 0.24;

    // 1. Lower Chassis (Sporty electric metallic blue)
    const chassisGeo = new THREE.BoxGeometry(length, height * 0.45, width);
    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      metalness: 0.7,
      roughness: 0.25,
    });
    const chassis = new THREE.Mesh(chassisGeo, chassisMat);
    chassis.position.set(wheelbase / 2, wheelRadius + height * 0.22, 0);
    chassis.castShadow = true;
    this.carRoot.add(chassis);

    // 2. Cabin / Glass Roof
    const cabinGeo = new THREE.BoxGeometry(length * 0.55, height * 0.45, width * 0.85);
    const cabinMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.9,
      roughness: 0.1,
    });
    const cabin = new THREE.Mesh(cabinGeo, cabinMat);
    cabin.position.set(wheelbase * 0.35, wheelRadius + height * 0.65, 0);
    cabin.castShadow = true;
    this.carRoot.add(cabin);

    // 3. Headlights (Bright Xenon Glow)
    const headGeo = new THREE.BoxGeometry(0.12, 0.14, 0.28);
    const headMat = new THREE.MeshBasicMaterial({ color: 0xecfeff });
    const flLight = new THREE.Mesh(headGeo, headMat);
    flLight.position.set(length - 0.75, wheelRadius + 0.35, -width * 0.38);
    const frLight = new THREE.Mesh(headGeo, headMat);
    frLight.position.set(length - 0.75, wheelRadius + 0.35, width * 0.38);
    this.carRoot.add(flLight, frLight);

    // 4. Taillights (Neon Red)
    const tailMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const rlLight = new THREE.Mesh(headGeo, tailMat);
    rlLight.position.set(-0.75, wheelRadius + 0.35, -width * 0.38);
    const rrLight = new THREE.Mesh(headGeo, tailMat);
    rrLight.position.set(-0.75, wheelRadius + 0.35, width * 0.38);
    this.carRoot.add(rlLight, rrLight);

    // 5. Roof LiDAR Sensor Dome
    const lidarDomeGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.16, 16);
    const lidarDomeMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8 });
    const lidarDome = new THREE.Mesh(lidarDomeGeo, lidarDomeMat);
    lidarDome.position.set(wheelbase * 0.4, wheelRadius + height + 0.08, 0);
    this.carRoot.add(lidarDome);

    // 6. Wheels
    this.wheels = {
      frontLeft: this.createWheel(wheelRadius, wheelWidth, true),
      frontRight: this.createWheel(wheelRadius, wheelWidth, true),
      rearLeft: this.createWheel(wheelRadius, wheelWidth, false),
      rearRight: this.createWheel(wheelRadius, wheelWidth, false),
    };

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
      color: 0x18181b,
      roughness: 0.85,
    });
    const tire = new THREE.Mesh(wheelGeo, wheelMat);
    tire.castShadow = true;
    steerPivot.add(tire);

    // Alloy Rim
    const capGeo = new THREE.CylinderGeometry(radius * 0.55, radius * 0.55, width * 1.05, 12);
    capGeo.rotateX(Math.PI / 2);
    const capMat = new THREE.MeshStandardMaterial({ color: 0xd4d4d8, metalness: 0.85 });
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
    // Clear road elements
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

    // 1. Asphalt Road Surface Mesh
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
      color: 0x181c24,
      roughness: 0.92,
      metalness: 0.08,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.receiveShadow = true;
    this.roadGroup.add(roadMesh);

    // 2. Solid White Shoulder Lines
    const leftLinePts = left.map((p) => simToThree(p[0], p[1], 0.02));
    const rightLinePts = right.map((p) => simToThree(p[0], p[1], 0.02));

    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
    const lGeo = new THREE.BufferGeometry().setFromPoints(leftLinePts);
    const rGeo = new THREE.BufferGeometry().setFromPoints(rightLinePts);
    this.roadGroup.add(new THREE.Line(lGeo, lineMat));
    this.roadGroup.add(new THREE.Line(rGeo, lineMat));

    // 3. Center Dashed Lane Divider (Yellow dashes)
    const centerPts = center.map((p) => simToThree(p[0], p[1], 0.025));
    const cGeo = new THREE.BufferGeometry().setFromPoints(centerPts);
    const cMat = new THREE.LineDashedMaterial({
      color: 0xfacc15,
      dashSize: 2.5,
      gapSize: 2.0,
    });
    const centerLine = new THREE.Line(cGeo, cMat);
    centerLine.computeLineDistances();
    this.roadGroup.add(centerLine);

    // 4. Start & Goal Markers
    if (roadData.start_pose) {
      const sp = simToThree(roadData.start_pose[0], roadData.start_pose[1], 0.03);
      const startGeo = new THREE.PlaneGeometry(roadData.total_width_m, 1.4);
      startGeo.rotateX(-Math.PI / 2);
      const startMat = new THREE.MeshBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.65 });
      const startMarker = new THREE.Mesh(startGeo, startMat);
      startMarker.position.copy(sp);
      this.roadGroup.add(startMarker);
    }

    if (roadData.goal_pose) {
      const gp = simToThree(roadData.goal_pose[0], roadData.goal_pose[1], 0.03);
      const goalGeo = new THREE.PlaneGeometry(roadData.total_width_m, 2.2);
      goalGeo.rotateX(-Math.PI / 2);
      const goalMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.75 });
      const goalMarker = new THREE.Mesh(goalGeo, goalMat);
      goalMarker.position.copy(gp);
      this.roadGroup.add(goalMarker);
    }

    // 5. Rebuild Scenery (Buildings, Sidewalks, Trees, Streetlights)
    this.rebuildScenery(roadData.scenery);

    // 6. Build 3D Static Obstacles
    this.rebuildObstacles(roadData.obstacles);
  }

  rebuildScenery(scenery) {
    while (this.sceneryGroup.children.length > 0) {
      const obj = this.sceneryGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      this.sceneryGroup.remove(obj);
    }
    if (!scenery) return;

    // A. Sidewalks & Raised Curbs
    if (scenery.sidewalks) {
      const curbMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.7 });
      const walkMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.85 });

      for (const sw of scenery.sidewalks) {
        if (!sw.points || sw.points.length < 2) continue;
        const pts = sw.points;
        const swWidth = sw.width_m || 2.4;
        const curbHeight = sw.curb_height_m || 0.18;

        const vertices = [];
        const indices = [];

        for (let i = 0; i < pts.length; i++) {
          const pt = pts[i];
          const nx = pt.nx || 0;
          const ny = pt.ny || 1;
          const side = sw.side === "left" ? 1.0 : -1.0;

          // Inner curb edge (roadside)
          const pInner = simToThree(pt.x, pt.y, curbHeight);
          // Outer edge (building side)
          const pOuter = simToThree(pt.x + side * nx * swWidth, pt.y + side * ny * swWidth, curbHeight);

          vertices.push(pInner.x, pInner.y, pInner.z);
          vertices.push(pOuter.x, pOuter.y, pOuter.z);

          if (i < pts.length - 1) {
            const row = i * 2;
            indices.push(row, row + 1, row + 2);
            indices.push(row + 1, row + 3, row + 2);
          }
        }

        const walkGeo = new THREE.BufferGeometry();
        walkGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
        walkGeo.setIndex(indices);
        walkGeo.computeVertexNormals();

        const walkMesh = new THREE.Mesh(walkGeo, walkMat);
        walkMesh.receiveShadow = true;
        this.sceneryGroup.add(walkMesh);
      }
    }

    // B. Procedural 3D Buildings
    if (scenery.buildings) {
      for (const b of scenery.buildings) {
        const bGroup = new THREE.Group();
        const baseGeo = new THREE.BoxGeometry(b.length, b.height, b.width);
        const baseMat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(b.color || "#1e293b"),
          roughness: 0.4,
          metalness: 0.3,
        });
        const baseMesh = new THREE.Mesh(baseGeo, baseMat);
        baseMesh.castShadow = true;
        baseMesh.receiveShadow = true;
        bGroup.add(baseMesh);

        // Glowing Windows Grid
        if (b.height > 8.0) {
          const winMat = new THREE.MeshBasicMaterial({
            color: new THREE.Color(b.accent_color || "#38bdf8"),
            transparent: true,
            opacity: 0.65,
          });
          const winGeo = new THREE.PlaneGeometry(b.length * 0.88, b.height * 0.85);
          const winMesh = new THREE.Mesh(winGeo, winMat);
          winMesh.position.set(0, 0, (b.width / 2.0) + 0.05);
          bGroup.add(winMesh);
        }

        // Rooftop HVAC unit
        if (b.has_hvac) {
          const hvacGeo = new THREE.BoxGeometry(b.length * 0.35, 1.8, b.width * 0.35);
          const hvacMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.5 });
          const hvac = new THREE.Mesh(hvacGeo, hvacMat);
          hvac.position.set(0, (b.height / 2.0) + 0.9, 0);
          bGroup.add(hvac);
        }

        const pos = simToThree(b.x, b.y, b.height / 2.0);
        bGroup.position.copy(pos);
        bGroup.rotation.y = b.yaw;
        this.sceneryGroup.add(bGroup);
      }
    }

    // C. 3D Roadside Trees
    if (scenery.trees) {
      for (const t of scenery.trees) {
        const treeGroup = new THREE.Group();
        const trunkH = t.trunk_height || 1.8;
        const trunkR = t.trunk_radius || 0.22;
        const canopyR = t.canopy_radius || 2.2;
        const canopyH = t.canopy_height || 3.8;

        // Trunk
        const trunkGeo = new THREE.CylinderGeometry(trunkR * 0.7, trunkR, trunkH, 8);
        const trunkMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(t.trunk_color || "#78350f"), roughness: 0.9 });
        const trunk = new THREE.Mesh(trunkGeo, trunkMat);
        trunk.position.y = trunkH / 2.0;
        trunk.castShadow = true;
        treeGroup.add(trunk);

        // Tiered Foliage Canopy
        const canopyMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(t.foliage_color || "#166534"), roughness: 0.75 });
        const c1 = new THREE.Mesh(new THREE.ConeGeometry(canopyR, canopyH * 0.6, 8), canopyMat);
        c1.position.y = trunkH + canopyH * 0.3;
        c1.castShadow = true;
        treeGroup.add(c1);

        const c2 = new THREE.Mesh(new THREE.ConeGeometry(canopyR * 0.75, canopyH * 0.5, 8), canopyMat);
        c2.position.y = trunkH + canopyH * 0.55;
        c2.castShadow = true;
        treeGroup.add(c2);

        const pos = simToThree(t.x, t.y, 0);
        treeGroup.position.copy(pos);
        this.sceneryGroup.add(treeGroup);
      }
    }

    // D. 3D Streetlights
    if (scenery.streetlights) {
      for (const sl of scenery.streetlights) {
        const slGroup = new THREE.Group();
        const poleH = sl.height || 6.5;

        // Vertical pole
        const poleGeo = new THREE.CylinderGeometry(0.08, 0.12, poleH, 8);
        const poleMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8 });
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.y = poleH / 2.0;
        slGroup.add(pole);

        // Light Fixture Head
        const headGeo = new THREE.BoxGeometry(0.35, 0.15, 0.8);
        const headMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(sl.light_color || "#fef08a") });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.set(0, poleH, 0.4);
        slGroup.add(head);

        const pos = simToThree(sl.x, sl.y, 0);
        slGroup.position.copy(pos);
        slGroup.rotation.y = sl.yaw;
        this.sceneryGroup.add(slGroup);
      }
    }
  }

  rebuildObstacles(obstacles) {
    while (this.obstacleGroup.children.length > 0) {
      const obj = this.obstacleGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      this.obstacleGroup.remove(obj);
    }
    if (!obstacles) return;

    for (const obs of obstacles) {
      const boxGeo = new THREE.BoxGeometry(obs.length, obs.height, obs.width);
      const boxMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(obs.color || "#f97316"),
        roughness: 0.35,
        metalness: 0.25,
      });
      const boxMesh = new THREE.Mesh(boxGeo, boxMat);
      boxMesh.castShadow = true;
      boxMesh.receiveShadow = true;

      const pos = simToThree(obs.x, obs.y, (obs.z || 0) + obs.height / 2.0);
      boxMesh.position.copy(pos);
      boxMesh.rotation.y = obs.yaw;

      const wireGeo = new THREE.EdgesGeometry(boxGeo);
      const wireMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
      const wireMesh = new THREE.LineSegments(wireGeo, wireMat);
      boxMesh.add(wireMesh);

      this.obstacleGroup.add(boxMesh);
    }
  }

  createNPCVehicleMesh(npc) {
    const root = new THREE.Group();
    const len = npc.length || 4.5;
    const wid = npc.width || 1.8;
    const hei = npc.height || 1.45;
    const color = new THREE.Color(npc.color || "#ef4444");

    // Chassis
    const chassisGeo = new THREE.BoxGeometry(len, hei * 0.5, wid);
    const chassisMat = new THREE.MeshStandardMaterial({
      color: color,
      metalness: 0.5,
      roughness: 0.35,
    });
    const chassis = new THREE.Mesh(chassisGeo, chassisMat);
    chassis.position.y = hei * 0.3;
    chassis.castShadow = true;
    root.add(chassis);

    // Dark Cabin
    const cabinGeo = new THREE.BoxGeometry(len * 0.55, hei * 0.45, wid * 0.85);
    const cabinMat = new THREE.MeshStandardMaterial({
      color: 0x111827,
      metalness: 0.9,
      roughness: 0.1,
    });
    const cabin = new THREE.Mesh(cabinGeo, cabinMat);
    cabin.position.set(-len * 0.05, hei * 0.7, 0);
    cabin.castShadow = true;
    root.add(cabin);

    // Headlights (Warm White)
    const hlGeo = new THREE.BoxGeometry(0.1, 0.12, 0.24);
    const hlMat = new THREE.MeshBasicMaterial({ color: 0xfffbeb });
    const hl1 = new THREE.Mesh(hlGeo, hlMat);
    hl1.position.set(len / 2.0, hei * 0.35, -wid * 0.36);
    const hl2 = new THREE.Mesh(hlGeo, hlMat);
    hl2.position.set(len / 2.0, hei * 0.35, wid * 0.36);
    root.add(hl1, hl2);

    // Tail / Brake Lights (Red)
    const tlMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const tl1 = new THREE.Mesh(hlGeo, tlMat);
    tl1.position.set(-len / 2.0, hei * 0.35, -wid * 0.36);
    const tl2 = new THREE.Mesh(hlGeo, tlMat);
    tl2.position.set(-len / 2.0, hei * 0.35, wid * 0.36);
    root.add(tl1, tl2);

    // 4 Wheels
    const wR = 0.32;
    const wW = 0.22;
    const wGeo = new THREE.CylinderGeometry(wR, wR, wW, 16);
    wGeo.rotateX(Math.PI / 2);
    const wMat = new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.8 });

    const w1 = new THREE.Mesh(wGeo, wMat);
    w1.position.set(len * 0.3, wR, -wid * 0.48);
    const w2 = new THREE.Mesh(wGeo, wMat);
    w2.position.set(len * 0.3, wR, wid * 0.48);
    const w3 = new THREE.Mesh(wGeo, wMat);
    w3.position.set(-len * 0.3, wR, -wid * 0.48);
    const w4 = new THREE.Mesh(wGeo, wMat);
    w4.position.set(-len * 0.3, wR, wid * 0.48);
    root.add(w1, w2, w3, w4);

    return root;
  }

  updateTrafficVisuals(trafficList) {
    if (!trafficList) return;

    const seenIds = new Set();

    for (const npc of trafficList) {
      seenIds.add(npc.id);
      let mesh = this.trafficMeshes.get(npc.id);
      if (!mesh) {
        mesh = this.createNPCVehicleMesh(npc);
        this.trafficGroup.add(mesh);
        this.trafficMeshes.set(npc.id, mesh);
      }

      // Smooth lerp NPC positions
      const targetPos = simToThree(npc.x, npc.y, 0);
      mesh.position.lerp(targetPos, 0.35);
      mesh.rotation.y = lerpAngle(mesh.rotation.y, npc.yaw, 0.35);
    }

    // Clean up despawned NPCs
    for (const [id, mesh] of this.trafficMeshes.entries()) {
      if (!seenIds.has(id)) {
        this.trafficGroup.remove(mesh);
        this.trafficMeshes.delete(id);
      }
    }
  }

  updateLidarPointCloud(s) {
    if (!this.lidarPointsMesh) return;
    if (!this.showLidar || !s.lidar || !s.lidar.points) {
      this.lidarPointsMesh.geometry.setDrawRange(0, 0);
      return;
    }

    const pts = s.lidar.points;
    const v = s.vehicle;
    const cosY = Math.cos(v.yaw);
    const sinY = Math.sin(v.yaw);
    const count = Math.min(pts.length, this.maxLidarPoints);

    const posArray = this.lidarPointsMesh.geometry.attributes.position.array;
    const colArray = this.lidarPointsMesh.geometry.attributes.color.array;

    for (let i = 0; i < count; i++) {
      const lx = pts[i][0];
      const ly = pts[i][1];
      const lz = pts[i][2];

      // Sensor mount offset: (1.4, 0.0, 1.6)
      const wx = v.x + (1.4 * cosY) + (lx * cosY - ly * sinY);
      const wy = v.y + (1.4 * sinY) + (lx * sinY + ly * cosY);
      const wz = 1.6 + lz;

      const threeP = simToThree(wx, wy, wz);
      posArray[i * 3 + 0] = threeP.x;
      posArray[i * 3 + 1] = threeP.y;
      posArray[i * 3 + 2] = threeP.z;

      // Turbo elevation color gradient
      if (wz <= 0.20) {
        colArray[i * 3 + 0] = 0.40;
        colArray[i * 3 + 1] = 0.15;
        colArray[i * 3 + 2] = 0.85;
      } else if (wz <= 0.95) {
        colArray[i * 3 + 0] = 0.08;
        colArray[i * 3 + 1] = 0.95;
        colArray[i * 3 + 2] = 0.88;
      } else {
        colArray[i * 3 + 0] = 1.00;
        colArray[i * 3 + 1] = 0.70;
        colArray[i * 3 + 2] = 0.12;
      }
    }

    this.lidarPointsMesh.geometry.setDrawRange(0, count);
    this.lidarPointsMesh.geometry.attributes.position.needsUpdate = true;
    this.lidarPointsMesh.geometry.attributes.color.needsUpdate = true;
  }

  updateDetectedObstacles(obstacles) {
    if (!this.detectedBoxGroup) return;

    while (this.detectedBoxGroup.children.length > 0) {
      const child = this.detectedBoxGroup.children[0];
      this.detectedBoxGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    if (!this.showBoundingBoxes || !obstacles || obstacles.length === 0) return;

    for (const obs of obstacles) {
      const boxGeo = new THREE.BoxGeometry(obs.length, obs.height, obs.width);
      const wireGeo = new THREE.EdgesGeometry(boxGeo);
      const wireMat = new THREE.LineBasicMaterial({
        color: 0x06b6d4, // Cyan glow
        linewidth: 2,
      });
      const boxMesh = new THREE.LineSegments(wireGeo, wireMat);

      const fillMat = new THREE.MeshBasicMaterial({
        color: 0x06b6d4,
        transparent: true,
        opacity: 0.14,
      });
      const fillMesh = new THREE.Mesh(boxGeo, fillMat);
      boxMesh.add(fillMesh);

      // Orientation tick mark along heading
      const arrowLen = Math.max(obs.length * 0.6, 1.0);
      const arrowGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(arrowLen, 0, 0),
      ]);
      const arrowMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2 });
      const arrowMesh = new THREE.Line(arrowGeo, arrowMat);
      boxMesh.add(arrowMesh);

      const threeP = simToThree(obs.x, obs.y, obs.z);
      boxMesh.position.copy(threeP);
      boxMesh.rotation.y = obs.yaw;

      this.detectedBoxGroup.add(boxMesh);
    }
  }

  renderMinimap(s) {
    if (!this.minimapCtx || !s.cost_map || !s.cost_map.costs) return;

    const ctx = this.minimapCtx;
    const cm = s.cost_map;
    const v = s.vehicle;
    const width = this.minimapCanvas.width;
    const height = this.minimapCanvas.height;

    ctx.clearRect(0, 0, width, height);

    const rows = cm.height;
    const cols = cm.width;
    const cellW = width / cols;
    const cellH = height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cost = cm.costs[r * cols + c];
        if (cost <= 0) {
          ctx.fillStyle = "#040d1a";
        } else if (cost >= 90) {
          ctx.fillStyle = "#ef4444";
        } else if (cost >= 50) {
          ctx.fillStyle = "#f59e0b";
        } else {
          ctx.fillStyle = "#0284c7";
        }
        ctx.fillRect(c * cellW, height - (r + 1) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Vehicle ego marker on minimap
    const egoCol = Math.round((v.x - cm.origin_x) / cm.resolution_m);
    const egoRow = Math.round((v.y - cm.origin_y) / cm.resolution_m);
    const egoPixX = egoCol * cellW;
    const egoPixY = height - egoRow * cellH;

    ctx.save();
    ctx.translate(egoPixX, egoPixY);
    ctx.rotate(-v.yaw);

    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.moveTo(7, 0);
    ctx.lineTo(-5, -4);
    ctx.lineTo(-3, 0);
    ctx.lineTo(-5, 4);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
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
    if (this.autopilot && !this.keys.space) return;

    let targetAccel = 0.0;
    let targetSteer = 0.0;
    let targetBrake = 0.0;

    if (this.keys.up) targetAccel += 3.2;
    if (this.keys.down) {
      if (this.lastSnapshot && this.lastSnapshot.vehicle.velocity > 0.4) {
        targetBrake = 1.0;
      } else {
        targetAccel -= 2.0;
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

  updatePlanningVisuals(planning) {
    if (!this.globalPathLine || !this.localPathLine || !this.lookaheadMarker) return;

    if (!this.showPlannedPaths || !planning) {
      this.globalPathLine.visible = false;
      this.localPathLine.visible = false;
      this.lookaheadMarker.visible = false;
      return;
    }

    if (planning.global_path && planning.global_path.length > 1) {
      const gPts = planning.global_path.map((p) => simToThree(p[0], p[1], 0.08));
      this.globalPathLine.geometry.setFromPoints(gPts);
      this.globalPathLine.visible = true;
    } else {
      this.globalPathLine.visible = false;
    }

    if (planning.local_path && planning.local_path.length > 1) {
      const lPts = planning.local_path.map((p) => simToThree(p[0], p[1], 0.14));
      this.localPathLine.geometry.setFromPoints(lPts);
      this.localPathLine.visible = true;
    } else {
      this.localPathLine.visible = false;
    }

    if (planning.lookahead_point) {
      const tp = simToThree(planning.lookahead_point[0], planning.lookahead_point[1], 0.25);
      this.lookaheadMarker.position.copy(tp);
      this.lookaheadMarker.visible = true;
    } else {
      this.lookaheadMarker.visible = false;
    }
  }

  handleSnapshot(s) {
    this.lastSnapshot = s;

    // Set target state for continuous 60/120 FPS lerp interpolation
    this.targetState = {
      x: s.vehicle.x,
      y: s.vehicle.y,
      yaw: s.vehicle.yaw,
      velocity: s.vehicle.velocity,
      steering_angle: s.vehicle.steering_angle,
      acceleration: s.vehicle.acceleration,
    };

    if (!this.currState) {
      this.currState = { ...this.targetState };
    }

    // Check if road needs rebuilding
    const obsKey = s.road.obstacles ? s.road.obstacles.map((o) => `${o.id}:${o.x}:${o.y}`).join("|") : "";
    const sig = `${s.road.road_type}_${s.road.road_length_m}_${obsKey}`;
    if (!this.roadSignature || this.roadSignature !== sig) {
      this.roadSignature = sig;
      this.currentRoadType = s.road.road_type;
      this.trailPoints = [];
      if (this.trailLine) this.trailLine.geometry.setDrawRange(0, 0);
      this.rebuildRoad(s.road);
    }

    // Dynamic Traffic NPCs
    this.updateTrafficVisuals(s.traffic);

    // LiDAR point cloud
    this.updateLidarPointCloud(s);

    // Detected obstacle bounding boxes
    this.updateDetectedObstacles(s.detected_obstacles);

    // Planning Visuals
    this.updatePlanningVisuals(s.planning);

    // Mini-map Cost Map
    this.renderMinimap(s);

    // Update HUD
    this.updateHUD(s);
  }

  updateCamera(v) {
    if (this.cameraMode === "orbit") return;

    const cosY = Math.cos(v.yaw);
    const sinY = Math.sin(v.yaw);

    if (this.cameraMode === "chase") {
      const dist = 9.0;
      const height = 3.8;
      const camX = v.x - dist * cosY;
      const camY = v.y - dist * sinY;
      const targetX = v.x + 3.5 * cosY;
      const targetY = v.y + 3.5 * sinY;

      const p = simToThree(camX, camY, height);
      const lookAt = simToThree(targetX, targetY, 0.85);

      this.camera.position.lerp(p, 0.15);
      this.camera.lookAt(lookAt);
    } else if (this.cameraMode === "cockpit") {
      const camX = v.x + 1.25 * cosY;
      const camY = v.y + 1.25 * sinY;
      const targetX = v.x + 18.0 * cosY;
      const targetY = v.y + 18.0 * sinY;

      const p = simToThree(camX, camY, 1.25);
      const lookAt = simToThree(targetX, targetY, 1.15);

      this.camera.position.copy(p);
      this.camera.lookAt(lookAt);
    } else if (this.cameraMode === "drone") {
      const p = simToThree(v.x, v.y, 48.0);
      const lookAt = simToThree(v.x, v.y, 0);
      this.camera.position.lerp(p, 0.12);
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

    const lidarHits = s.status.lidar_points !== undefined ? s.status.lidar_points : (s.lidar ? s.lidar.valid_returns : 0);
    const lidarHitsEl = document.getElementById("val-lidar-hits");
    if (lidarHitsEl) {
      lidarHitsEl.innerHTML = `${lidarHits} <small>pts</small>`;
    }

    const detObsEl = document.getElementById("val-detected-obs");
    if (detObsEl) {
      const count = s.status.detected_obstacles_count !== undefined
        ? s.status.detected_obstacles_count
        : (s.detected_obstacles ? s.detected_obstacles.length : 0);
      detObsEl.innerHTML = `${count} <small>objects</small>`;
    }

    this.autopilot = Boolean(s.autopilot);
    const autopilotBtn = document.getElementById("btn-autopilot");
    if (autopilotBtn) {
      if (this.autopilot) {
        autopilotBtn.classList.add("active");
        autopilotBtn.textContent = "AUTOPILOT: ON";
      } else {
        autopilotBtn.classList.remove("active");
        autopilotBtn.textContent = "AUTOPILOT: OFF";
      }
    }

    const plannerStateEl = document.getElementById("val-planner-status");
    const plannerState = s.status.planner_status || "MANUAL";
    if (plannerStateEl) {
      plannerStateEl.textContent = plannerState;
    }

    const plannerPill = document.getElementById("planner-pill");
    const plannerText = document.getElementById("planner-text");
    if (plannerPill && plannerText) {
      if (!this.autopilot) {
        plannerPill.className = "pill pill-offline";
        plannerText.textContent = "MANUAL MODE";
      } else if (plannerState === "EMERGENCY_STOP" || s.status.failure_reason) {
        plannerPill.className = "pill pill-danger";
        plannerText.textContent = `E-STOP (${s.status.failure_reason || "BLOCKED"})`;
      } else if (plannerState === "GOAL_REACHED" || s.status.goal_reached) {
        plannerPill.className = "pill pill-success";
        plannerText.textContent = "AUTOPILOT · GOAL";
      } else {
        plannerPill.className = "pill pill-online";
        plannerText.textContent = `AUTOPILOT · ${plannerState}`;
      }
    }

    const collPill = document.getElementById("collision-pill");
    const goalPill = document.getElementById("goal-pill");

    collPill.style.display = s.status.collision ? "flex" : "none";
    goalPill.style.display = s.status.goal_reached ? "flex" : "none";
  }

  initControls() {
    document.querySelectorAll(".btn-cam").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-cam").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.cameraMode = e.target.dataset.cam;
        this.orbitControls.enabled = this.cameraMode === "orbit";
      });
    });

    document.querySelectorAll(".btn-road").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-road, .btn-scenario").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        const roadType = e.target.dataset.road;
        this.resetSimulation(roadType);
      });
    });

    document.querySelectorAll(".btn-scenario").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-road, .btn-scenario").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        const scenarioId = e.target.dataset.scenario;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: "load_scenario", scenario_id: scenarioId }));
        }
      });
    });

    const autopilotBtn = document.getElementById("btn-autopilot");
    if (autopilotBtn) {
      autopilotBtn.addEventListener("click", () => {
        const nextState = !this.autopilot;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: "set_autopilot", enabled: nextState }));
        }
      });
    }

    const togglePathsBtn = document.getElementById("btn-toggle-paths");
    if (togglePathsBtn) {
      togglePathsBtn.addEventListener("click", () => {
        this.showPlannedPaths = !this.showPlannedPaths;
        if (this.showPlannedPaths) {
          togglePathsBtn.classList.add("active");
          togglePathsBtn.textContent = "VISIBLE";
        } else {
          togglePathsBtn.classList.remove("active");
          togglePathsBtn.textContent = "HIDDEN";
        }
      });
    }

    const toggleLidarBtn = document.getElementById("btn-toggle-lidar");
    if (toggleLidarBtn) {
      toggleLidarBtn.addEventListener("click", () => {
        this.showLidar = !this.showLidar;
        if (this.showLidar) {
          toggleLidarBtn.classList.add("active");
          toggleLidarBtn.textContent = "VISIBLE";
        } else {
          toggleLidarBtn.classList.remove("active");
          toggleLidarBtn.textContent = "HIDDEN";
          if (this.lidarPointsMesh) {
            this.lidarPointsMesh.geometry.setDrawRange(0, 0);
          }
        }
      });
    }

    const toggleBoxBtn = document.getElementById("btn-toggle-boxes");
    if (toggleBoxBtn) {
      toggleBoxBtn.addEventListener("click", () => {
        this.showBoundingBoxes = !this.showBoundingBoxes;
        if (this.showBoundingBoxes) {
          toggleBoxBtn.classList.add("active");
          toggleBoxBtn.textContent = "VISIBLE";
        } else {
          toggleBoxBtn.classList.remove("active");
          toggleBoxBtn.textContent = "HIDDEN";
          if (this.detectedBoxGroup) {
            while (this.detectedBoxGroup.children.length > 0) {
              const child = this.detectedBoxGroup.children[0];
              this.detectedBoxGroup.remove(child);
              if (child.geometry) child.geometry.dispose();
              if (child.material) child.material.dispose();
            }
          }
        }
      });
    }

    document.getElementById("btn-reset").addEventListener("click", () => {
      this.resetSimulation(this.currentRoadType || "s_curve");
    });

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

    // 60/120 FPS Continuous Physics Transform Interpolation
    if (this.currState && this.targetState) {
      this.currState.x = lerp(this.currState.x, this.targetState.x, 0.35);
      this.currState.y = lerp(this.currState.y, this.targetState.y, 0.35);
      this.currState.yaw = lerpAngle(this.currState.yaw, this.targetState.yaw, 0.35);
      this.currState.steering_angle = lerp(this.currState.steering_angle, this.targetState.steering_angle, 0.4);
      this.currState.velocity = lerp(this.currState.velocity, this.targetState.velocity, 0.35);

      const threePos = simToThree(this.currState.x, this.currState.y, 0);
      this.carRoot.position.copy(threePos);
      this.carRoot.rotation.y = this.currState.yaw;

      // Smooth Front Wheel Steer
      this.wheels.frontLeft.steerPivot.rotation.y = this.currState.steering_angle;
      this.wheels.frontRight.steerPivot.rotation.y = this.currState.steering_angle;

      // Spin tires
      const rollDelta = (this.currState.velocity * 0.016) / this.wheels.rearLeft.radius;
      this.wheels.rearLeft.tire.rotation.z -= rollDelta;
      this.wheels.rearRight.tire.rotation.z -= rollDelta;
      this.wheels.frontLeft.tire.rotation.z -= rollDelta;
      this.wheels.frontRight.tire.rotation.z -= rollDelta;

      // Update trail
      this.updateTrail(threePos);

      // Smooth Camera tracking
      this.updateCamera(this.currState);
    }

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
