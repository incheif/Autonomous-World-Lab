"""
FastAPI & WebSocket Server for Autonomous-World-Lab.
Streams real-time simulation telemetry to the Three.js 3D WebGL viewer
and accepts real-time keyboard control commands.
"""

import asyncio
import json
from pathlib import Path
from typing import Set

try:
    import orjson
    def dump_json_bytes(obj) -> bytes:
        return orjson.dumps(obj)
except ImportError:
    def dump_json_bytes(obj) -> bytes:
        return json.dumps(obj).encode("utf-8")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autonomy.interfaces.types import ControlCommand
from simulator.simulation import Simulation
from simulator.world.road import RoadMap
from simulator.world.traffic import TrafficManager
from simulator.scenarios import SCENARIOS, get_scenario

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIEWER_DIR = PROJECT_ROOT / "viewer"

app = FastAPI(title="Autonomous-World-Lab Simulation API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance - default to city_traffic scenario with dual lanes & NPC traffic
sim: Simulation = Simulation(
    dt_sec=0.02,
    road=get_scenario("city_traffic").builder(),
    traffic=TrafficManager(enabled=True, num_vehicles=6),
    autopilot=True,
)
active_connections: Set[WebSocket] = set()

# Current live control command from client
current_command: ControlCommand = ControlCommand()


class CommandRequest(BaseModel):
    acceleration: float = 0.0
    steering: float = 0.0
    brake: float = 0.0
    emergency_stop: bool = False


class ResetRequest(BaseModel):
    road_type: str = "s_curve"
    road_length_m: float = 140.0
    scenario_id: str | None = None
    autopilot: bool | None = None


class AutopilotRequest(BaseModel):
    enabled: bool


@app.get("/")
async def root():
    index_file = VIEWER_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Autonomous-World-Lab API</h1><p>Viewer index.html not found.</p>")


@app.get("/api/status")
async def get_status():
    return sim.snapshot()


@app.get("/api/scenarios")
async def list_scenarios():
    return {
        "scenarios": [
            {
                "id": s.scenario_id,
                "name": s.name,
                "description": s.description,
            }
            for s in SCENARIOS.values()
        ]
    }


@app.post("/api/autopilot")
async def post_autopilot(req: AutopilotRequest):
    sim.autopilot = req.enabled
    return {"status": "ok", "autopilot": sim.autopilot}


@app.post("/api/command")
async def post_command(req: CommandRequest):
    global current_command
    current_command = ControlCommand(
        acceleration=req.acceleration,
        steering=req.steering,
        brake=req.brake,
        emergency_stop=req.emergency_stop,
    )
    return {"status": "ok", "command": current_command.to_dict()}


@app.post("/api/reset")
async def post_reset(req: ResetRequest):
    global sim, current_command
    current_command = ControlCommand()
    use_autopilot = req.autopilot if req.autopilot is not None else sim.autopilot
    if req.scenario_id:
        road = get_scenario(req.scenario_id).builder()
    else:
        road = RoadMap(road_type=req.road_type, road_length_m=req.road_length_m)
    sim = Simulation(dt_sec=0.02, road=road, autopilot=use_autopilot)
    sim.reset()
    return {"status": "reset", "snapshot": sim.snapshot()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_command
    await websocket.accept()
    active_connections.add(websocket)

    # Initial frame sends full road data so client builds 3D meshes immediately
    sent_road_signature: str | None = None

    # Asynchronous receiver task so incoming keyboard/control commands are processed instantly with zero latency
    async def receiver():
        global current_command, sim
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                action = msg.get("action")
                if action == "control":
                    current_command = ControlCommand(
                        acceleration=float(msg.get("accel", 0.0)),
                        steering=float(msg.get("steer", 0.0)),
                        brake=float(msg.get("brake", 0.0)),
                        emergency_stop=bool(msg.get("estop", False)),
                    )
                elif action == "set_autopilot":
                    sim.autopilot = bool(msg.get("enabled", False))
                elif action == "load_scenario":
                    scenario_id = msg.get("scenario_id", "city_traffic")
                    spec = get_scenario(scenario_id)
                    sim.road = spec.builder()
                    num_npc = 6 if scenario_id == "city_traffic" else 0
                    sim.traffic = TrafficManager(num_vehicles=num_npc)
                    sim.traffic.initialize(sim.road)
                    sim.reset()
                    current_command = ControlCommand()
                elif action == "reset":
                    road_type = msg.get("road_type", "s_curve")
                    road = RoadMap(road_type=road_type, road_length_m=140.0)
                    sim.road = road
                    sim.reset()
                    current_command = ControlCommand()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            pass

    rx_task = asyncio.create_task(receiver())

    try:
        loop = asyncio.get_running_loop()
        while True:
            t_start = loop.time()

            # Step physics by dt (autopilot uses None to invoke NavigationStack)
            if sim.autopilot:
                if not sim.collision and not sim.goal_reached:
                    sim.step(None)
                else:
                    # Keep vehicle halted smoothly after goal/collision
                    sim.step(ControlCommand(brake=1.0, emergency_stop=True))
            else:
                sim.step(current_command)

            current_sig = f"{sim.road.road_type}_{sim.road.road_length_m}"
            include_road = (sent_road_signature != current_sig)
            if include_road:
                sent_road_signature = current_sig

            snap = sim.snapshot(include_static_road=include_road)
            payload_bytes = dump_json_bytes(snap)

            await websocket.send_bytes(payload_bytes)

            # Precise timing compensation: subtract actual step compute time from 0.02s
            elapsed = loop.time() - t_start
            sleep_time = max(0.001, 0.02 - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        active_connections.discard(websocket)
    except Exception:
        active_connections.discard(websocket)
    finally:
        rx_task.cancel()
        active_connections.discard(websocket)


# Mount static assets for Three.js viewer
if VIEWER_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(VIEWER_DIR)), name="static")

