# parhi_mcp_server.py
"""Model Context Protocol (MCP) Server for Parhi-GPT.

Exposes Parhi's autonomous laptop controls, batch filesystem tools,
game automation, system commands, and hybrid intelligence to external
MCP clients (Antigravity IDE, Claude Desktop, Cursor, etc.).

Complies with the MCP Specification using JSON-RPC 2.0 over stdio.
"""
from __future__ import annotations

import json
import os
import sys

# Save real standard output for JSON-RPC protocol messages ONLY
_mcp_protocol_stdout = sys.stdout
# Redirect default print statements to stderr so module loading doesn't corrupt JSON-RPC
sys.stdout = sys.stderr

from typing import Any

from autonomous_controller import AutonomousController
from batch_executor import BatchExecutor
from game_automation import GameAutomationEngine
from system_commands import SystemCommands, detect_system_command
from memory import MemoryManager
from hybrid_brain import HybridBrain


# ---------------------------------------------------------------------------
# MCP Tool Definitions (JSON Schema)
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "parhi_take_control",
        "description": "Engage or disengage autonomous laptop control with Master Access verification ('With my access only').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["authorize", "status", "revoke", "take_control", "release_control"],
                    "description": "Action to perform on autonomous controller.",
                },
                "master_pin": {
                    "type": "string",
                    "description": "User's master authorization PIN or passphrase.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "parhi_batch_folders",
        "description": "Create batch folders in bulk (e.g. 100 folders with friend names) safely.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of folders to create (1 to 500).",
                    "default": 100,
                },
                "base_name": {
                    "type": "string",
                    "description": "Base prefix or name format (e.g. 'Friend').",
                    "default": "Friend",
                },
                "custom_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit list of friend names.",
                },
                "target_dir": {
                    "type": "string",
                    "description": "Optional destination folder path.",
                },
            },
            "required": ["count"],
        },
    },
    {
        "name": "parhi_game_automation",
        "description": "Control Free Fire or Android emulator game loop (anti-AFK survivor routine with WASD, loot, heal).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "status"],
                    "description": "Start, stop, or check status of game automation loop.",
                },
                "max_duration_minutes": {
                    "type": "number",
                    "description": "Maximum minutes to run before auto-stopping.",
                    "default": 30.0,
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "parhi_system_control",
        "description": "Execute JARVIS system commands (open apps, volume, screenshot, system info, power).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Natural language system command (e.g., 'open chrome', 'set volume to 50', 'take a screenshot', 'battery status').",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "parhi_query_memory",
        "description": "Retrieve stored facts, user preferences, and memories from Parhi's persistent database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or topic to search in memory.",
                },
            },
        },
    },
    {
        "name": "parhi_ask",
        "description": "Query Parhi's hybrid brain (Gemini 3.8 Flash / local Transformer) with emotional awareness.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "User question or prompt for Parhi.",
                },
            },
            "required": ["prompt"],
        },
    },
]


# ---------------------------------------------------------------------------
# Server Implementation
# ---------------------------------------------------------------------------

class ParhiMCPServer:
    """Standard Model Context Protocol (MCP) server running on stdio JSON-RPC 2.0."""

    def __init__(self) -> None:
        self.controller = AutonomousController()
        self.batch_executor = BatchExecutor()
        self.game_engine = GameAutomationEngine()
        self.sys_cmds = SystemCommands()
        self.memory = MemoryManager()
        self.brain = HybridBrain()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a JSON-RPC 2.0 request and return the response."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # Notifications (no id) do not expect a response
        if req_id is None and method == "notifications/initialized":
            return None

        # 1. Initialize
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": False,
                        },
                    },
                    "serverInfo": {
                        "name": "parhi_mcp",
                        "version": "1.0.0",
                    },
                },
            }

        # 2. List tools
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": MCP_TOOLS,
                },
            }

        # 3. Call tool
        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result_content = self.execute_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_content,
                        }
                    ],
                },
            }

        # Unknown method
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not supported.",
            },
        }

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute the designated Parhi tool."""
        try:
            if tool_name == "parhi_take_control":
                action = args.get("action", "status")
                pin = args.get("master_pin", "")
                if action == "authorize":
                    ok, msg = self.controller.authorize(pin)
                    return msg
                elif action == "revoke":
                    return self.controller.revoke_access()
                elif action == "take_control":
                    if pin:
                        self.controller.authorize(pin)
                    handled, msg = self.controller.process_command("take control")
                    return msg
                elif action == "release_control":
                    handled, msg = self.controller.process_command("release control")
                    return msg
                else:
                    auth = self.controller.is_authorized()
                    return f"Autonomous Controller status: {'Authorized' if auth else 'Unauthorized (PIN required)'}."

            elif tool_name == "parhi_batch_folders":
                count = int(args.get("count", 100))
                base_name = args.get("base_name", "Friend")
                custom_names = args.get("custom_names")
                target_dir = args.get("target_dir")
                res = self.batch_executor.create_batch_folders(
                    count=count,
                    base_name=base_name,
                    custom_names=custom_names,
                    target_dir=target_dir,
                )
                return res.display_text

            elif tool_name == "parhi_game_automation":
                action = args.get("action", "status")
                if action == "start":
                    duration = float(args.get("max_duration_minutes", 30.0))
                    ok, msg = self.game_engine.start_game_routine(max_duration_minutes=duration)
                    return msg
                elif action == "stop":
                    return self.game_engine.stop_game_routine()
                else:
                    st = self.game_engine.status
                    return (
                        f"Game Routine: {'Running' if st.is_running else 'Idle'}.\n"
                        f"Target: {st.emulator_name or 'None'}, Actions: {st.actions_performed}."
                    )

            elif tool_name == "parhi_system_control":
                cmd = args.get("command", "")
                detected = detect_system_command(cmd)
                if detected:
                    cmd_type, target = detected
                    if cmd_type == "open_app":
                        res = self.sys_cmds.open_app(target)
                    elif cmd_type == "close_app":
                        res = self.sys_cmds.close_app(target)
                    elif cmd_type == "open_website":
                        res = self.sys_cmds.open_website(target)
                    elif cmd_type == "take_photo":
                        res = self.sys_cmds.take_photo()
                    elif cmd_type == "screenshot":
                        res = self.sys_cmds.screenshot()
                    elif cmd_type == "volume_up":
                        res = self.sys_cmds.volume_control("up")
                    elif cmd_type == "volume_down":
                        res = self.sys_cmds.volume_control("down")
                    elif cmd_type == "volume_mute":
                        res = self.sys_cmds.volume_control("mute")
                    elif cmd_type == "battery":
                        res = self.sys_cmds.battery_status()
                    elif cmd_type == "wifi":
                        res = self.sys_cmds.wifi_status()
                    else:
                        res = None
                    return res.display_text if res else f"Processed system command: {cmd_type}"
                return f"Could not recognize system command: '{cmd}'"

            elif tool_name == "parhi_query_memory":
                q = args.get("query", "")
                summary = self.memory.get_context_summary()
                user_name = self.memory.get_user_name()
                prefix = f"User: {user_name}\n" if user_name else ""
                return f"{prefix}{summary}" if summary else "Memory store is currently empty."

            elif tool_name == "parhi_ask":
                prompt = args.get("prompt", "")
                ans = self.brain.think(prompt, user_mood="neutral")
                return ans or "I'm thinking about that."

            return f"Unknown tool: '{tool_name}'"

        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def run_stdio(self) -> None:
        """Run standard I/O JSON-RPC loop for MCP clients with isolated protocol stream."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response:
                    _mcp_protocol_stdout.write(json.dumps(response) + "\n")
                    _mcp_protocol_stdout.flush()
            except Exception as e:
                err = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                _mcp_protocol_stdout.write(json.dumps(err) + "\n")
                _mcp_protocol_stdout.flush()


if __name__ == "__main__":
    server = ParhiMCPServer()
    server.run_stdio()
