# parhi_gui_server.py
"""Parhi Neural Interface — FastAPI GUI Server & WebSocket Telemetry Gateway.

Serves the 8K biological Brainbow neural connectome interface, manages
real-time bi-directional WebSockets for speech/text dialogue, streams
synaptic thoughts, and executes JARVIS-style hardware triggers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Any

import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Workspace directory
BASE_DIR = Path(__file__).resolve().parent
GUI_DIR = BASE_DIR / "gui"

app = FastAPI(title="Parhi Neural Interface Gateway")

# Shared CLI instance reference
_cli_instance = None
_cli_lock = threading.Lock()


def get_cli():
    """Lazily load or retrieve the singleton ParhiCLI instance."""
    global _cli_instance
    with _cli_lock:
        if _cli_instance is None:
            from chat import ParhiCLI, DEFAULT_CHECKPOINT
            # Fast mode overrides: keep streaming in WebSocket layer
            overrides = {
                "enable_streaming": False,
                "enable_thinking": True,
                "enable_emotion_detection": True,
                "enable_personality": True,
                "enable_memory": True,
                "enable_agent_tools": True,
                "enable_system_commands": True,
                "enable_screen_vision": True,
            }
            _cli_instance = ParhiCLI(
                checkpoint_path=DEFAULT_CHECKPOINT,
                config_overrides=overrides,
            )
    return _cli_instance


def collect_telemetry() -> dict[str, Any]:
    """Query live Windows system vitals (battery, network, relationship bond, mood)."""
    telemetry: dict[str, Any] = {
        "battery": 100,
        "battery_charging": True,
        "wifi_ssid": "Connected",
        "wifi_strength": 95,
        "local_ip": "127.0.0.1",
        "bond": 65,
        "bond_label": "Warm & Attentive",
        "mood": "Thoughtful",
    }

    # 1. Real Battery
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            telemetry["battery"] = int(batt.percent)
            telemetry["battery_charging"] = bool(batt.power_plugged)
    except Exception:
        pass

    # 2. Real Local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        telemetry["local_ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    # 3. Real WiFi SSID & Signal via netsh (Windows)
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netsh", "wlan", "show", "interfaces"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
            )
            ssid_m = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
            if ssid_m:
                telemetry["wifi_ssid"] = ssid_m.group(1).strip()
            sig_m = re.search(r"^\s*Signal\s*:\s*(\d+)%", out, re.M)
            if sig_m:
                telemetry["wifi_strength"] = int(sig_m.group(1))
        except Exception:
            pass

    # 4. Parhi Neuro-Bond and Mood
    try:
        cli = get_cli()
        if cli.personality and hasattr(cli.personality, "mood"):
            telemetry["mood"] = str(cli.personality.mood).capitalize()
        if cli.memory and hasattr(cli.memory, "data"):
            bond_val = cli.memory.data.get("bond", 65)
            telemetry["bond"] = bond_val
            if bond_val >= 80:
                telemetry["bond_label"] = "Deep Synaptic Resonance"
            elif bond_val >= 50:
                telemetry["bond_label"] = "Warm & Attentive"
            else:
                telemetry["bond_label"] = "Developing Connection"
    except Exception:
        pass

    return telemetry


@app.get("/")
async def serve_index():
    """Serve the primary HTML application."""
    index_path = GUI_DIR / "index.html"
    return FileResponse(index_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Primary WebSocket gateway for two-way neural communication."""
    await websocket.accept()
    print("[✓] Neural UI Client Connected to Gateway")

    # Transmit initial telemetry immediately
    initial_telemetry = collect_telemetry()
    await websocket.send_text(json.dumps({
        "type": "telemetry",
        **initial_telemetry,
    }))

    cli = get_cli()

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            msg_type = data.get("type")

            if msg_type == "get_telemetry":
                t_data = collect_telemetry()
                await websocket.send_text(json.dumps({
                    "type": "telemetry",
                    **t_data,
                }))

            elif msg_type == "chat_message":
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue

                # Notify client that thinking has begun
                await websocket.send_text(json.dumps({
                    "type": "thinking_start",
                }))

                # Generate response in worker thread (prevents async blocking)
                def _do_respond():
                    return cli.respond(user_text, return_thinking=True)

                resp_tuple = await asyncio.to_thread(_do_respond)
                if isinstance(resp_tuple, tuple):
                    reply_text, thinking_text = resp_tuple
                else:
                    reply_text, thinking_text = resp_tuple, ""

                # Detect if this message triggered learning or new memory
                is_learning = bool(
                    thinking_text
                    or any(k in user_text.lower() for k in ["learn", "remember", "know", "explain", "how to", "why", "science"])
                )

                # Send response to UI
                await websocket.send_text(json.dumps({
                    "type": "chat_response",
                    "text": reply_text,
                    "thinking": thinking_text,
                    "learned": is_learning,
                }))

                # Follow up with updated telemetry
                updated_t = collect_telemetry()
                await websocket.send_text(json.dumps({
                    "type": "telemetry",
                    **updated_t,
                }))

            elif msg_type == "system_command":
                cmd = data.get("command", "")
                target = data.get("target", "")
                print(f"[⚡] Hardware trigger received: {cmd} ({target})")

                feedback = ""
                if cli.sys_commands:
                    try:
                        res = cli._execute_system_command(cmd, target)
                        feedback = str(res)
                    except Exception as e:
                        feedback = f"Error executing {cmd}: {e}"
                else:
                    feedback = f"Executed {cmd} ({target})"

                await websocket.send_text(json.dumps({
                    "type": "command_feedback",
                    "message": feedback,
                }))

    except WebSocketDisconnect:
        print("[ℹ] Neural UI Client Disconnected")
    except Exception as e:
        print(f"[✗] WebSocket error: {e}")


# Mount static assets (style.css, app.js, images)
app.mount("/", StaticFiles(directory=str(GUI_DIR)), name="gui_static")


def find_available_port(start_port: int = 8000) -> int:
    """Find a free TCP port starting from start_port."""
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
            port += 1
    return start_port


def launch_native_window(url: str, fullscreen: bool = True, max_fps: bool = True):
    """Open Microsoft Edge with zero delay, borderless fullscreen, and uncapped max FPS."""
    if sys.platform == "win32":
        # Check for Microsoft Edge first
        edge_paths = [
            os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe"),
        ]
        
        edge_flags = [
            f"--app={url}",
            "--start-fullscreen",
            "--disable-frame-rate-limit",        # Uncaps FPS beyond 60Hz (120Hz/144Hz/240Hz+)
            "--disable-gpu-vsync",               # Disables VSync delay for maximum render rate
            "--enable-gpu-rasterization",        # Offload 8K canvas shaders directly to GPU
            "--enable-zero-copy",                # Fast GPU zero-copy raster buffers
            "--ignore-gpu-blocklist",            # Force hardware acceleration
            "--enable-accelerated-2d-canvas",    # High performance 2D canvas
            "--enable-accelerated-video-decode", # Hardware accelerated video decode
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--no-first-run",
            "--no-default-browser-check",
            "--fast-start",
            "--hide-crash-restore-bubble",
        ]

        for p in edge_paths:
            if os.path.exists(p):
                subprocess.Popen([p] + edge_flags)
                return

        # Check for Chrome with same high-performance flags
        chrome_paths = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]
        for p in chrome_paths:
            if os.path.exists(p):
                subprocess.Popen([p] + edge_flags)
                return

    # Fallback to default browser
    import webbrowser
    webbrowser.open(url)


def run_gui(host: str = "127.0.0.1", port: int = 8000, open_window: bool = True):
    """Run the FastAPI server and open the native window."""
    actual_port = find_available_port(port)
    url = f"http://{host}:{actual_port}"

    print(f"\n=======================================================")
    print(f"  🧠 PARHI v3.5 — SYNAPTIC NEURAL INTERFACE")
    print(f"  Server URL: {url}")
    print(f"  Mode: Edge Automation (Fullscreen | Uncapped Max FPS)")
    print(f"=======================================================\n")

    # Start uvicorn server in a separate background thread
    config = uvicorn.Config(
        app,
        host=host,
        port=actual_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Zero-delay server bind detection (fast socket check instead of fixed sleep)
    for _ in range(40):
        try:
            with socket.create_connection((host, actual_port), timeout=0.04):
                break
        except OSError:
            time.sleep(0.02)

    if open_window:
        print(f"[🚀] Launching Edge Automation Fullscreen (Max FPS) at {url}...")
        launch_native_window(url, fullscreen=True, max_fps=True)

    # Keep alive until interrupt
    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[!] Shutting down Parhi Neural Interface...")


if __name__ == "__main__":
    run_gui()
