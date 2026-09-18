# autonomous_controller.py
"""Autonomous Laptop Control Engine ("Take Control" Mode) with Access Control.

Ensures that laptop control commands, batch automation, and game loops
execute ONLY with the master user's verified permission ("With my access only").
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from batch_executor import BatchExecutor, BatchExecutionResult
from config import ParhiConfig


@dataclass
class AuthSession:
    """Represents an active authorization session."""
    is_authenticated: bool = False
    authenticated_at: float = 0.0
    expires_at: float = 0.0
    auth_method: str = ""  # "pin", "phrase", or "session"


class AutonomousController:
    """Central manager for autonomous laptop control and access authorization."""

    def __init__(self, config: ParhiConfig | None = None) -> None:
        self.config = config or ParhiConfig()
        
        # Master credentials from config or environment variables
        self.master_pin = os.environ.get("PARHI_MASTER_PIN", getattr(self.config, "master_access_pin", "1327"))
        self.master_phrase = os.environ.get(
            "PARHI_MASTER_PHRASE",
            getattr(self.config, "master_vocal_passphrase", "vatsal access granted")
        ).strip().lower()

        # Session validity duration in seconds (default: 45 minutes)
        self.session_timeout: float = 45 * 60

        self.session = AuthSession()
        self.batch_executor = BatchExecutor()
        self.active_mode: str = "standby"  # "standby", "autonomous_active", "game_mode"
        self._emergency_stop_triggered: bool = False

    def is_authorized(self) -> bool:
        """Check if there is a currently valid, unexpired authorization session."""
        if not self.session.is_authenticated:
            return False
        if time.time() > self.session.expires_at:
            self.session.is_authenticated = False
            return False
        return True

    def authorize(self, credential: str) -> tuple[bool, str]:
        """Authenticate user with Master PIN or vocal passphrase.

        Args:
            credential: PIN string or spoken authorization phrase.

        Returns:
            Tuple of (success_boolean, human_readable_message)
        """
        cleaned = credential.strip().lower()

        # 1. Match PIN
        if cleaned == self.master_pin.strip().lower():
            now = time.time()
            self.session = AuthSession(
                is_authenticated=True,
                authenticated_at=now,
                expires_at=now + self.session_timeout,
                auth_method="pin",
            )
            return True, "Master PIN accepted. Full laptop control authorized for 45 minutes."

        # 2. Match Vocal Passphrase (e.g., "vatsal access granted")
        if self.master_phrase in cleaned or cleaned == self.master_phrase:
            now = time.time()
            self.session = AuthSession(
                is_authenticated=True,
                authenticated_at=now,
                expires_at=now + self.session_timeout,
                auth_method="phrase",
            )
            return True, f"Master vocal passphrase verified. Welcome Vatsal, laptop control is now active."

        # 3. Check for inline authorization in commands (e.g., "take control pin 1327")
        if self.master_pin in cleaned:
            now = time.time()
            self.session = AuthSession(
                is_authenticated=True,
                authenticated_at=now,
                expires_at=now + self.session_timeout,
                auth_method="inline_pin",
            )
            return True, "Master PIN accepted. Access granted."

        return False, "Access Denied: Invalid master credential. Autonomous control rejected."

    def revoke_access(self) -> str:
        """Revoke current authorization session immediately."""
        self.session = AuthSession(is_authenticated=False)
        self.active_mode = "standby"
        return "Autonomous session terminated and access credentials revoked."

    def process_command(self, message: str) -> tuple[bool, str]:
        """Process natural language request for control or task execution.

        Handles:
            - "take control", "took control", "Parhi take control of my laptop"
            - Batch folder tasks: "create 100 folders with a particular name of my friends"
            - Authentication requests

        Returns:
            Tuple of (handled_boolean, response_text)
        """
        lower = message.lower().strip()

        # 1. Direct authorization attempts (e.g., "pin 1327" or "1327")
        if lower.startswith("pin ") or lower == self.master_pin or self.master_phrase in lower:
            success, msg = self.authorize(message)
            return True, msg

        # 2. Check for "take control" / "took control"
        is_take_control = any(
            phrase in lower
            for phrase in (
                "take control", "took control", "take over", "took over",
                "control my laptop", "control my pc", "autonomous mode",
                "autopilot on"
            )
        )

        # Check for inline credentials with take control (e.g. "took control with pin 1327")
        if is_take_control:
            if self.master_pin in lower or self.master_phrase in lower:
                self.authorize(message)

            if not self.is_authorized():
                return True, (
                    "Security Verification Required: You requested autonomous laptop control ('With my access only').\n"
                    "Please state your Master PIN (e.g. 'PIN 1327') or authorization phrase to confirm your identity."
                )

            self.active_mode = "autonomous_active"
            return True, (
                "Autonomous Control Activated. I have taken control of your laptop with Master Access.\n"
                "You can now instruct me to perform tasks (e.g., 'create 100 folders with friend names', "
                "'play Free Fire while I step out', or any system workflow). Say 'release control' to cancel anytime."
            )

        # 3. Check for "release control" / "stop control"
        if any(phrase in lower for phrase in ("release control", "stop control", "cancel control", "hand over control")):
            self.active_mode = "standby"
            return True, "Autonomous control relinquished. Returning manual input to user."

        # 4. Batch folder creation command: "create 100 folders with a particular name of my friends"
        if ("folder" in lower or "directory" in lower) and any(verb in lower for verb in ("create", "make", "generate")):
            # Check authorization if count is large (e.g., > 5)
            count, base_name, custom_names = self.batch_executor.parse_folder_request(message)
            
            if count > 5 and not self.is_authorized():
                # Check if pin was supplied in the same message
                if self.master_pin in lower:
                    self.authorize(message)
                else:
                    return True, (
                        f"Authorization Required: Creating {count} folders requires Master Access.\n"
                        f"Please provide your PIN to confirm this batch operation."
                    )

            result = self.batch_executor.create_batch_folders(
                count=count,
                base_name=base_name,
                custom_names=custom_names,
            )
            return True, result.display_text

        # Not an autonomous command
        return False, ""
