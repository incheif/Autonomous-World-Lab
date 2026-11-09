"""
FastAPI & WebSocket Server for Autonomous-World-Lab.
Streams real-time simulation telemetry to the Three.js 3D WebGL viewer
and accepts real-time keyboard control commands.
"""

import asyncio
import json
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autonomy.interfaces.types import ControlCommand
from simulator.simulation import Simulation
from simulator.world.road import RoadMap

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

# Global simulation instance
sim: Simulation = Simulation(dt_sec=0.02, road=RoadMap(road_type="s_curve", road_length_m=140.0))
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


@app.get("/")
async def root():
    index_file = VIEWER_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Autonomous-World-Lab API</h1><p>Viewer index.html not found.</p>")


@app.get("/api/status")
async def get_status():
    return sim.snapshot()


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
    road = RoadMap(road_type=req.road_type, road_length_m=req.road_length_m)
    sim = Simulation(dt_sec=0.02, road=road)
    sim.reset()
    return {"status": "reset", "snapshot": sim.snapshot()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_command
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # Check for incoming client control messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.02)
                msg = json.loads(data)
                action = msg.get("action")
                if action == "control":
                    current_command = ControlCommand(
                        acceleration=float(msg.get("accel", 0.0)),
                        steering=float(msg.get("steer", 0.0)),
                        brake=float(msg.get("brake", 0.0)),
                        emergency_stop=bool(msg.get("estop", False)),
                    )
                elif action == "reset":
                    road_type = msg.get("road_type", "s_curve")
                    road = RoadMap(road_type=road_type, road_length_m=140.0)
                    sim.road = road
                    sim.reset()
                    current_command = ControlCommand()
            except asyncio.TimeoutError:
                pass

            # Step physics by dt
            sim.step(current_command)
            payload = json.dumps(sim.snapshot())

            await websocket.send_text(payload)
            await asyncio.sleep(0.02)  # 50 Hz loop

    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Mount static assets for Three.js viewer
if VIEWER_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(VIEWER_DIR)), name="static")
