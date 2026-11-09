"""
Launcher script for Autonomous-World-Lab 3D WebGL Viewer.
Starts the FastAPI + WebSocket backend and opens http://localhost:8080 in your browser.
"""

import sys
import webbrowser
import uvicorn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.api.server import app

def main():
    port = 8080
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 70)
    print(" ⚡ AUTONOMOUS-WORLD-LAB · 3D SIMULATION VIEWER (PHASE 1)")
    print(f" URL: {url}")
    print(" Press Ctrl+C in terminal to stop server.")
    print("=" * 70)

    # Open browser automatically after a short delay
    webbrowser.open(url)

    # Run Uvicorn ASGI server
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
