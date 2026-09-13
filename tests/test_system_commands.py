# tests/test_system_commands.py
"""Tests for system_commands.py and wake_word.py."""
import os
import pytest
from system_commands import detect_system_command, SystemCommands, APP_REGISTRY
from wake_word import WakeWordConfig, WakeWordDetector


def test_detect_system_command_camera():
    cmd, target = detect_system_command("Parhi open camera")
    assert cmd == "open_camera"

    cmd2, _ = detect_system_command("take a photo")
    assert cmd2 == "take_photo"


def test_detect_system_command_app():
    cmd, target = detect_system_command("open notepad")
    assert cmd == "open_app"
    assert target == "notepad"

    cmd, target = detect_system_command("close chrome")
    assert cmd == "close_app"
    assert "chrome" in target


def test_detect_system_command_system_controls():
    cmd, _ = detect_system_command("take screenshot")
    assert cmd == "screenshot"

    cmd, _ = detect_system_command("lock screen")
    assert cmd == "lock_screen"

    cmd, _ = detect_system_command("turn volume up")
    assert cmd == "volume_up"

    cmd, _ = detect_system_command("mute volume")
    assert cmd == "volume_mute"

    cmd, _ = detect_system_command("increase brightness")
    assert cmd == "brightness_up"

    cmd, _ = detect_system_command("battery status")
    assert cmd == "battery"

    cmd, _ = detect_system_command("wifi status")
    assert cmd == "wifi"


def test_detect_system_command_website():
    cmd, target = detect_system_command("open youtube.com")
    assert cmd == "open_website"
    assert target == "youtube.com"


def test_system_commands_confirmation():
    sc = SystemCommands()
    # shutdown needs confirmation
    res = sc.power_command("shutdown")
    assert res.success
    assert "confirm" in res.display_text.lower() or "sure" in res.display_text.lower()
    assert sc._confirmation_pending is not None

    # cancel confirmation
    cancel_res = sc.check_confirmation("no cancel that")
    assert cancel_res is not None
    assert "cancelled" in cancel_res.display_text.lower()
    assert sc._confirmation_pending is None


def test_system_commands_info():
    sc = SystemCommands()
    ip_res = sc.get_ip_address()
    assert ip_res.success
    assert len(ip_res.display_text) > 0

    battery_res = sc.battery_status()
    assert battery_res.success


def test_wake_word_config():
    cfg = WakeWordConfig(wake_word="parhi", voice_log_enabled=False)
    assert cfg.wake_word == "parhi"
    assert not cfg.voice_log_enabled
