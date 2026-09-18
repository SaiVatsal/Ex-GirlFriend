# tests/test_mcp_server.py
"""Unit and protocol tests for Parhi MCP (Model Context Protocol) Server."""
import json
import pytest
from parhi_mcp_server import ParhiMCPServer, MCP_TOOLS


def test_mcp_tools_schema():
    assert len(MCP_TOOLS) >= 6
    names = [t["name"] for t in MCP_TOOLS]
    assert "parhi_take_control" in names
    assert "parhi_batch_folders" in names
    assert "parhi_game_automation" in names
    assert "parhi_system_control" in names
    assert "parhi_query_memory" in names
    assert "parhi_ask" in names


def test_mcp_initialize():
    server = ParhiMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test_client", "version": "1.0"},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "parhi_mcp"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_tools_list():
    server = ParhiMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    assert len(tools) >= 6


def test_mcp_call_take_control():
    server = ParhiMCPServer()
    # 1. Authorize with PIN
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "parhi_take_control",
            "arguments": {
                "action": "authorize",
                "master_pin": "1327",
            },
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 3
    text = resp["result"]["content"][0]["text"]
    assert "accepted" in text.lower() or "authorized" in text.lower()


def test_mcp_call_batch_folders(tmp_path):
    server = ParhiMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "parhi_batch_folders",
            "arguments": {
                "count": 5,
                "base_name": "FriendTest",
                "target_dir": str(tmp_path),
            },
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    text = resp["result"]["content"][0]["text"]
    assert "created 5 folders" in text.lower() or "successfully" in text.lower()
    # Verify folders exist
    created = list(tmp_path.iterdir())
    assert len(created) == 5
