# tests/test_game_automation.py
"""Unit tests for Free Fire and emulator GameAutomationEngine."""
import time
import pytest
from game_automation import GameAutomationEngine, VK_CODES, SCAN_CODES


def test_game_engine_init():
    engine = GameAutomationEngine()
    assert not engine.status.is_running
    assert engine.status.actions_performed == 0


def test_key_mappings():
    # Verify critical Free Fire control keys are properly mapped
    for key in ["w", "a", "s", "d", "space", "shift", "c", "f", "4", "esc"]:
        assert key in VK_CODES
        assert key in SCAN_CODES
        assert VK_CODES[key] > 0
        assert SCAN_CODES[key] > 0


def test_find_game_window_graceful():
    engine = GameAutomationEngine()
    # Should not throw exception, returns None or (hwnd, title)
    result = engine.find_game_window()
    if result is not None:
        hwnd, title = result
        assert isinstance(hwnd, int)
        assert isinstance(title, str)


def test_game_loop_start_and_stop():
    engine = GameAutomationEngine()
    # Start routine with tiny duration
    ok, msg = engine.start_game_routine(max_duration_minutes=0.02)
    assert ok
    assert engine.status.is_running
    assert "Active" in msg or "Autonomous" in msg

    # Allow a tick
    time.sleep(0.4)

    # Stop routine
    stop_msg = engine.stop_game_routine("Test cleanup")
    assert not engine.status.is_running
    assert "stopped" in stop_msg.lower()
