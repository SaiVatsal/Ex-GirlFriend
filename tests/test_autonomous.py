# tests/test_autonomous.py
"""Unit tests for AutonomousController, Access Control, and BatchExecutor."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from autonomous_controller import AutonomousController
from batch_executor import BatchExecutor
from config import ParhiConfig


def test_access_control_pin():
    config = ParhiConfig(master_access_pin="1327")
    controller = AutonomousController(config)
    assert not controller.is_authorized()

    # Wrong PIN
    success, msg = controller.authorize("9999")
    assert not success
    assert not controller.is_authorized()
    assert "Access Denied" in msg

    # Correct PIN
    success, msg = controller.authorize("1327")
    assert success
    assert controller.is_authorized()
    assert "Master PIN accepted" in msg

    # Revoke
    controller.revoke_access()
    assert not controller.is_authorized()


def test_access_control_vocal_passphrase():
    config = ParhiConfig(master_vocal_passphrase="vatsal access granted")
    controller = AutonomousController(config)

    success, msg = controller.authorize("vatsal access granted")
    assert success
    assert controller.is_authorized()
    assert "verified" in msg.lower()


def test_batch_folder_parsing():
    executor = BatchExecutor()

    count, base, names = executor.parse_folder_request("create 100 folders with a particular name of my friends")
    assert count == 100
    assert base.lower() == "friend"

    count, base, names = executor.parse_folder_request("generate 25 folders named ProjectX")
    assert count == 25
    assert base == "ProjectX"

    count, base, names = executor.parse_folder_request("make 3 folders named Alice, Bob, Charlie")
    assert count == 3
    assert "Alice" in names
    assert "Bob" in names
    assert "Charlie" in names


def test_batch_folder_creation_100_folders():
    temp_dir = tempfile.mkdtemp(prefix="parhi_test_folders_")
    try:
        executor = BatchExecutor(default_base_dir=temp_dir)
        result = executor.create_batch_folders(
            count=100,
            base_name="Friend",
            target_dir=temp_dir,
        )

        assert result.success
        assert len(result.items_created) == 100

        # Verify all 100 folders physically exist on disk
        created_dirs = list(Path(temp_dir).iterdir())
        assert len(created_dirs) == 100

        # Verify undo removes them cleanly
        removed, errs = executor.undo_batch(result.undo_token)
        assert removed == 100
        assert len(list(Path(temp_dir).iterdir())) == 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_autonomous_take_control_flow():
    config = ParhiConfig(master_access_pin="1327")
    controller = AutonomousController(config)

    # Unauthorized attempt to take control
    handled, resp = controller.process_command("Parhi take control of my laptop")
    assert handled
    assert "Security Verification Required" in resp

    # Supply PIN
    handled, resp = controller.process_command("PIN 1327")
    assert handled
    assert "accepted" in resp.lower()
    assert controller.is_authorized()

    # Now take control succeeds
    handled, resp = controller.process_command("take control")
    assert handled
    assert "Autonomous Control Activated" in resp

    # Release control
    handled, resp = controller.process_command("release control")
    assert handled
    assert "relinquished" in resp.lower()
