# system_commands.py
"""JARVIS-style system control commands for Parhi-GPT.

Provides voice/text-activated system commands:
  - Open/close applications
  - Camera control
  - Screenshots
  - Volume/brightness control
  - System power (lock, shutdown, restart, sleep)
  - System info (battery, wifi, IP, running processes)
  - Web browsing
  - Recycle bin management

All commands use Windows-native APIs and subprocess calls.
Destructive operations require confirmation.
"""
from __future__ import annotations

import datetime
import os
import platform
import re
import subprocess
import socket
import sys
import webbrowser
from dataclasses import dataclass

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

from agent_tools import ToolResult


# ---------------------------------------------------------------------------
# App registry — maps common app names to executable paths / commands
# ---------------------------------------------------------------------------

APP_REGISTRY: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "wordpad": ["wordpad.exe"],
    "snipping tool": ["SnippingTool.exe"],
    "task manager": ["taskmgr.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "terminal": ["wt.exe"],
    "file explorer": ["explorer.exe"],
    "control panel": ["control.exe"],
    "settings": ["start", "ms-settings:"],
    "camera": ["start", "microsoft.windows.camera:"],
    "photos": ["start", "ms-photos:"],
    "maps": ["start", "bingmaps:"],
    "clock": ["start", "ms-clock:"],
    "calendar": ["start", "outlookcal:"],
    "mail": ["start", "outlookmail:"],
    "store": ["start", "ms-windows-store:"],
    "xbox": ["start", "xbox:"],
    "spotify": ["start", "spotify:"],
    "whatsapp": ["start", "whatsapp:"],
    "telegram": ["start", "telegram:"],
    "discord": ["start", "discord:"],
    # Browsers
    "chrome": ["start", "chrome"],
    "google chrome": ["start", "chrome"],
    "firefox": ["start", "firefox"],
    "edge": ["start", "msedge:"],
    "microsoft edge": ["start", "msedge:"],
    "brave": ["start", "brave"],
    # Dev tools
    "vs code": ["code"],
    "vscode": ["code"],
    "visual studio code": ["code"],
    "sublime": ["subl"],
    "sublime text": ["subl"],
    # Media
    "vlc": ["start", "vlc:"],
    "media player": ["start", "mswindowsmusic:"],
    "music": ["start", "mswindowsmusic:"],
    "movies": ["start", "mswindowsvideo:"],
    "groove music": ["start", "mswindowsmusic:"],
}


# ---------------------------------------------------------------------------
# System Commands
# ---------------------------------------------------------------------------

class SystemCommands:
    """JARVIS-style system control for Parhi.

    Provides methods for launching apps, controlling system settings,
    and querying system status. All methods return ToolResult for
    consistent integration with the agent tools pipeline.
    """

    def __init__(self) -> None:
        self._confirmation_pending: str | None = None

    # ---- App Management ----

    def open_app(self, app_name: str) -> ToolResult:
        """Open an application by name.

        Args:
            app_name: Name of the app to open.

        Returns:
            ToolResult with success/failure.
        """
        app_lower = app_name.lower().strip()

        # Check registry first
        if app_lower in APP_REGISTRY:
            cmd = APP_REGISTRY[app_lower]
            try:
                if cmd[0] == "start":
                    # Use shell start for UWP / protocol apps
                    subprocess.Popen(
                        ["cmd", "/c"] + cmd,
                        shell=False,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    subprocess.Popen(
                        cmd,
                        shell=False,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                return ToolResult(
                    tool_name="open_app",
                    success=True,
                    display_text=f"Opening {app_name} for you! 🚀",
                )
            except FileNotFoundError:
                return ToolResult(
                    tool_name="open_app",
                    success=False,
                    display_text=f"I couldn't find {app_name} on your system. It might not be installed.",
                )
            except Exception as e:
                return ToolResult(
                    tool_name="open_app",
                    success=False,
                    error=str(e),
                    display_text=f"I tried to open {app_name} but hit an error: {e}",
                )

        # Try running it directly as a command
        try:
            subprocess.Popen(
                [app_lower],
                shell=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return ToolResult(
                tool_name="open_app",
                success=True,
                display_text=f"Opening {app_name}! 🚀",
            )
        except Exception:
            pass

        # Try searching in Start Menu
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", app_lower],
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return ToolResult(
                tool_name="open_app",
                success=True,
                display_text=f"Launching {app_name}! 🚀",
            )
        except Exception:
            return ToolResult(
                tool_name="open_app",
                success=False,
                display_text=f"I couldn't find \"{app_name}\" on your system. Make sure it's installed.",
            )

    def close_app(self, app_name: str) -> ToolResult:
        """Close/kill a running application.

        Args:
            app_name: Name of the app to close.

        Returns:
            ToolResult with success/failure.
        """
        app_lower = app_name.lower().strip()

        # Map friendly names to process names
        process_map = {
            "notepad": "notepad.exe",
            "calculator": "CalculatorApp.exe",
            "chrome": "chrome.exe",
            "google chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "microsoft edge": "msedge.exe",
            "brave": "brave.exe",
            "vs code": "Code.exe",
            "vscode": "Code.exe",
            "spotify": "Spotify.exe",
            "discord": "Discord.exe",
            "vlc": "vlc.exe",
            "paint": "mspaint.exe",
            "word": "WINWORD.EXE",
            "excel": "EXCEL.EXE",
            "powerpoint": "POWERPNT.EXE",
            "teams": "Teams.exe",
            "telegram": "Telegram.exe",
            "whatsapp": "WhatsApp.exe",
        }

        proc_name = process_map.get(app_lower, f"{app_lower}.exe")

        try:
            result = subprocess.run(
                ["taskkill", "/IM", proc_name, "/F"],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                return ToolResult(
                    tool_name="close_app",
                    success=True,
                    display_text=f"Closed {app_name} for you. ✅",
                )
            else:
                return ToolResult(
                    tool_name="close_app",
                    success=False,
                    display_text=f"{app_name} doesn't seem to be running right now.",
                )
        except Exception as e:
            return ToolResult(
                tool_name="close_app",
                success=False,
                error=str(e),
                display_text=f"Couldn't close {app_name}: {e}",
            )

    # ---- Camera ----

    def open_camera(self) -> ToolResult:
        """Open the Windows Camera app."""
        return self.open_app("camera")

    def take_screenshot(self) -> ToolResult:
        """Take a screenshot and save it to the Desktop.

        Returns:
            ToolResult with the saved file path.
        """
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(desktop, f"parhi_screenshot_{timestamp}.png")

            try:
                import mss  # type: ignore[import-untyped]
                from PIL import Image  # type: ignore[import-untyped]

                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes(
                        "RGB",
                        (screenshot.width, screenshot.height),
                        screenshot.rgb,
                    )
                    img.save(filepath)

                return ToolResult(
                    tool_name="screenshot",
                    success=True,
                    result=filepath,
                    display_text=f"Screenshot saved to your Desktop! 📸\n  → {filepath}",
                )
            except ImportError:
                # Fallback: use Windows Snipping Tool
                subprocess.Popen(["SnippingTool.exe", "/clip"])
                return ToolResult(
                    tool_name="screenshot",
                    success=True,
                    display_text="I've opened the Snipping Tool for you to capture a screenshot! 📸",
                )
        except Exception as e:
            return ToolResult(
                tool_name="screenshot",
                success=False,
                error=str(e),
                display_text=f"Couldn't take a screenshot: {e}",
            )

    # ---- System Controls ----

    def lock_screen(self) -> ToolResult:
        """Lock the Windows screen."""
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            return ToolResult(
                tool_name="lock_screen",
                success=True,
                display_text="Locking your screen now. Stay safe! 🔒",
            )
        except Exception as e:
            return ToolResult(
                tool_name="lock_screen",
                success=False,
                error=str(e),
                display_text=f"Couldn't lock the screen: {e}",
            )

    def volume_control(self, action: str) -> ToolResult:
        """Control system volume.

        Args:
            action: One of 'up', 'down', 'mute'.

        Returns:
            ToolResult.
        """
        try:
            import ctypes

            VK_VOLUME_MUTE = 0xAD
            VK_VOLUME_DOWN = 0xAE
            VK_VOLUME_UP = 0xAF
            KEYEVENTF_KEYUP = 0x0002

            if action == "mute":
                vk = VK_VOLUME_MUTE
                msg = "Volume muted! 🔇"
            elif action == "down":
                vk = VK_VOLUME_DOWN
                msg = "Volume down! 🔉"
            elif action == "up":
                vk = VK_VOLUME_UP
                msg = "Volume up! 🔊"
            else:
                return ToolResult(
                    tool_name="volume",
                    success=False,
                    display_text="Say 'volume up', 'volume down', or 'volume mute'.",
                )

            # Press and release the volume key (repeat a few times for noticeable change)
            for _ in range(5 if action != "mute" else 1):
                ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
                ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

            return ToolResult(
                tool_name="volume",
                success=True,
                display_text=msg,
            )
        except Exception as e:
            return ToolResult(
                tool_name="volume",
                success=False,
                error=str(e),
                display_text=f"Couldn't adjust volume: {e}",
            )

    def brightness_control(self, action: str) -> ToolResult:
        """Control screen brightness.

        Args:
            action: 'up' or 'down'.

        Returns:
            ToolResult.
        """
        try:
            try:
                import screen_brightness_control as sbc  # type: ignore[import-untyped]
                current = sbc.get_brightness()[0]
                if action == "up":
                    new_val = min(100, current + 20)
                    sbc.set_brightness(new_val)
                    msg = f"Brightness up to {new_val}%! ☀️"
                elif action == "down":
                    new_val = max(0, current - 20)
                    sbc.set_brightness(new_val)
                    msg = f"Brightness down to {new_val}%! 🌙"
                else:
                    return ToolResult(
                        tool_name="brightness",
                        success=False,
                        display_text="Say 'brightness up' or 'brightness down'.",
                    )
                return ToolResult(
                    tool_name="brightness",
                    success=True,
                    display_text=msg,
                )
            except ImportError:
                # Fallback: use PowerShell WMI
                if action == "up":
                    ps_cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,80)"
                else:
                    ps_cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,30)"
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return ToolResult(
                    tool_name="brightness",
                    success=True,
                    display_text=f"Brightness adjusted {'up ☀️' if action == 'up' else 'down 🌙'}!",
                )
        except Exception as e:
            return ToolResult(
                tool_name="brightness",
                success=False,
                error=str(e),
                display_text=f"Couldn't adjust brightness: {e}",
            )

    def power_command(self, action: str, confirmed: bool = False) -> ToolResult:
        """Execute system power commands.

        Args:
            action: 'shutdown', 'restart', or 'sleep'.
            confirmed: Whether the user confirmed the action.

        Returns:
            ToolResult. If not confirmed, asks for confirmation.
        """
        if not confirmed:
            self._confirmation_pending = action
            return ToolResult(
                tool_name="power",
                success=True,
                display_text=f"⚠️ Are you sure you want to **{action}** the system? Say 'yes' or 'confirm' to proceed.",
            )

        try:
            if action == "shutdown":
                subprocess.run(["shutdown", "/s", "/t", "5"], creationflags=subprocess.CREATE_NO_WINDOW)
                msg = "Shutting down in 5 seconds... Goodbye! 👋"
            elif action == "restart":
                subprocess.run(["shutdown", "/r", "/t", "5"], creationflags=subprocess.CREATE_NO_WINDOW)
                msg = "Restarting in 5 seconds... See you soon! 🔄"
            elif action == "sleep":
                subprocess.run(
                    ["powershell", "-Command", "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                msg = "Putting the system to sleep... Sweet dreams! 😴"
            else:
                return ToolResult(tool_name="power", success=False, display_text="Unknown power action.")

            self._confirmation_pending = None
            return ToolResult(tool_name="power", success=True, display_text=msg)
        except Exception as e:
            return ToolResult(tool_name="power", success=False, error=str(e), display_text=f"Power command failed: {e}")

    def check_confirmation(self, message: str) -> ToolResult | None:
        """Check if a pending confirmation is being answered.

        Args:
            message: User's message.

        Returns:
            ToolResult if confirmation handled, None otherwise.
        """
        if not self._confirmation_pending:
            return None

        lower = message.lower().strip()
        yes_words = ["yes", "confirm", "do it", "go ahead", "y", "sure", "ok", "yeah", "yep", "proceed"]
        no_words = ["no", "cancel", "stop", "nah", "n", "never mind", "don't", "abort"]

        if lower in yes_words or any(w in lower.split() for w in ["yes", "confirm", "proceed", "sure"]):
            return self.power_command(self._confirmation_pending, confirmed=True)
        elif lower in no_words or any(w in lower.split() for w in ["no", "cancel", "stop", "nah", "abort"]):
            action = self._confirmation_pending
            self._confirmation_pending = None
            return ToolResult(
                tool_name="power",
                success=True,
                display_text=f"Cancelled {action}. Your system is safe! ✅",
            )
        return None

    # ---- Open Website ----

    def open_website(self, url: str) -> ToolResult:
        """Open a URL in the default browser.

        Args:
            url: The URL to open.

        Returns:
            ToolResult.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            webbrowser.open(url)
            return ToolResult(
                tool_name="open_website",
                success=True,
                display_text=f"Opening {url} in your browser! 🌐",
            )
        except Exception as e:
            return ToolResult(
                tool_name="open_website",
                success=False,
                error=str(e),
                display_text=f"Couldn't open the website: {e}",
            )

    # ---- System Info ----

    def list_running_apps(self) -> ToolResult:
        """List currently running applications."""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | Select-Object -Property Name,MainWindowTitle | Format-Table -AutoSize"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            output = result.stdout.strip()
            if output:
                return ToolResult(
                    tool_name="running_apps",
                    success=True,
                    result=output,
                    display_text=f"Here's what's running on your system:\n\n```\n{output}\n```",
                )
            else:
                return ToolResult(
                    tool_name="running_apps",
                    success=True,
                    display_text="No visible applications are running right now.",
                )
        except Exception as e:
            return ToolResult(
                tool_name="running_apps",
                success=False,
                error=str(e),
                display_text=f"Couldn't list running apps: {e}",
            )

    def battery_status(self) -> ToolResult:
        """Get battery level and charging status."""
        try:
            try:
                import psutil  # type: ignore[import-untyped]
                battery = psutil.sensors_battery()
                if battery:
                    pct = battery.percent
                    plugged = "charging ⚡" if battery.power_plugged else "on battery 🔋"
                    if battery.secsleft > 0 and not battery.power_plugged:
                        hours = battery.secsleft // 3600
                        mins = (battery.secsleft % 3600) // 60
                        time_left = f" — about {hours}h {mins}m remaining"
                    else:
                        time_left = ""
                    return ToolResult(
                        tool_name="battery",
                        success=True,
                        display_text=f"Battery is at **{pct}%**, {plugged}{time_left}.",
                    )
                else:
                    return ToolResult(
                        tool_name="battery",
                        success=True,
                        display_text="No battery detected — looks like you're on a desktop! 🖥️",
                    )
            except ImportError:
                # Fallback: PowerShell
                result = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-WmiObject Win32_Battery).EstimatedChargeRemaining"],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                pct = result.stdout.strip()
                if pct:
                    return ToolResult(
                        tool_name="battery",
                        success=True,
                        display_text=f"Battery is at **{pct}%** 🔋",
                    )
                else:
                    return ToolResult(
                        tool_name="battery",
                        success=True,
                        display_text="Couldn't read battery status — you might be on a desktop.",
                    )
        except Exception as e:
            return ToolResult(
                tool_name="battery",
                success=False,
                error=str(e),
                display_text=f"Couldn't check battery: {e}",
            )

    def wifi_status(self) -> ToolResult:
        """Get WiFi connection information."""
        try:
            result = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            output = result.stdout.strip()
            if "disconnected" in output.lower() or not output:
                return ToolResult(
                    tool_name="wifi",
                    success=True,
                    display_text="You're not connected to any WiFi network. 📡",
                )

            # Extract key info
            ssid = ""
            signal = ""
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("SSID") and "BSSID" not in line:
                    ssid = line.split(":", 1)[-1].strip()
                elif "Signal" in line:
                    signal = line.split(":", 1)[-1].strip()

            if ssid:
                return ToolResult(
                    tool_name="wifi",
                    success=True,
                    display_text=f"Connected to **{ssid}** — Signal: {signal} 📶",
                )
            else:
                return ToolResult(
                    tool_name="wifi",
                    success=True,
                    display_text="WiFi is active but I couldn't parse the connection details.",
                )
        except Exception as e:
            return ToolResult(
                tool_name="wifi",
                success=False,
                error=str(e),
                display_text=f"Couldn't check WiFi: {e}",
            )

    def get_ip_address(self) -> ToolResult:
        """Get the machine's IP address."""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            # Try to get public IP
            public_ip = ""
            try:
                import urllib.request
                public_ip = urllib.request.urlopen(
                    "https://api.ipify.org", timeout=5
                ).read().decode("utf-8")
            except Exception:
                pass

            parts = [f"**Local IP**: {local_ip}"]
            if public_ip:
                parts.append(f"**Public IP**: {public_ip}")
            parts.append(f"**Hostname**: {hostname}")

            return ToolResult(
                tool_name="ip_address",
                success=True,
                display_text="Here's your network info:\n" + "\n".join(parts),
            )
        except Exception as e:
            return ToolResult(
                tool_name="ip_address",
                success=False,
                error=str(e),
                display_text=f"Couldn't get IP address: {e}",
            )

    def play_music(self) -> ToolResult:
        """Open the default music player."""
        return self.open_app("music")

    def empty_recycle_bin(self, confirmed: bool = False) -> ToolResult:
        """Empty the Windows recycle bin.

        Args:
            confirmed: Whether the user confirmed.

        Returns:
            ToolResult.
        """
        if not confirmed:
            self._confirmation_pending = "empty_recycle_bin"
            return ToolResult(
                tool_name="recycle_bin",
                success=True,
                display_text="⚠️ Are you sure you want to **empty the Recycle Bin**? This can't be undone. Say 'yes' to confirm.",
            )

        try:
            import ctypes
            # SHEmptyRecycleBin flags: SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND
            ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 0x7)
            self._confirmation_pending = None
            return ToolResult(
                tool_name="recycle_bin",
                success=True,
                display_text="Recycle Bin emptied! 🗑️✨",
            )
        except Exception as e:
            return ToolResult(
                tool_name="recycle_bin",
                success=False,
                error=str(e),
                display_text=f"Couldn't empty the Recycle Bin: {e}",
            )


# ---------------------------------------------------------------------------
# Command parser — extracts intent from natural language
# ---------------------------------------------------------------------------

# Trigger patterns for system commands
SYSTEM_TRIGGERS: dict[str, list[str]] = {
    "open_camera": [
        "open camera", "start camera", "launch camera",
        "open the camera", "turn on camera", "camera on",
    ],
    "take_photo": [
        "take a photo", "take a picture", "capture photo",
        "take photo", "take picture", "snap a photo",
    ],
    "screenshot": [
        "take a screenshot", "screenshot", "capture screen",
        "screen capture", "take screenshot", "grab screen",
        "snap the screen",
    ],
    "open_app": [
        "open ", "launch ", "start ", "run ",
    ],
    "close_app": [
        "close ", "kill ", "stop ", "exit ", "quit ",
        "shut down ", "end ",
    ],
    "lock_screen": [
        "lock screen", "lock the screen", "lock my screen",
        "lock the system", "lock computer", "lock my computer",
        "lock the pc", "lock pc",
    ],
    "volume_up": [
        "volume up", "increase volume", "turn up volume",
        "louder", "raise volume", "turn it up",
    ],
    "volume_down": [
        "volume down", "decrease volume", "turn down volume",
        "quieter", "lower volume", "turn it down",
    ],
    "volume_mute": [
        "mute", "mute volume", "silence", "volume mute",
        "turn off sound", "mute the sound",
    ],
    "brightness_up": [
        "brightness up", "increase brightness", "brighter",
        "screen brighter", "more brightness",
    ],
    "brightness_down": [
        "brightness down", "decrease brightness", "dimmer",
        "screen dimmer", "less brightness", "dim the screen",
    ],
    "shutdown": [
        "shut down", "shutdown", "turn off the computer",
        "turn off pc", "power off",
    ],
    "restart": [
        "restart", "reboot", "restart the computer",
        "restart pc", "reboot system",
    ],
    "sleep": [
        "sleep", "put to sleep", "sleep mode",
        "hibernate", "standby",
    ],
    "open_website": [
        "open website", "go to website", "browse to",
        "open site", "visit ",
    ],
    "running_apps": [
        "what's running", "running apps", "active apps",
        "list running", "show processes", "what apps are open",
        "what is running", "running processes",
    ],
    "battery": [
        "battery status", "battery level", "how much battery",
        "battery percentage", "check battery", "battery",
    ],
    "wifi": [
        "wifi status", "wifi info", "wifi connection",
        "am i connected", "wifi", "network status",
    ],
    "play_music": [
        "play music", "play some music", "start music",
        "open music player", "play songs",
    ],
    "ip_address": [
        "ip address", "my ip", "what's my ip",
        "show ip", "get ip", "network info",
    ],
    "empty_recycle_bin": [
        "empty recycle bin", "clear recycle bin", "empty trash",
        "clean recycle bin", "delete recycle bin",
    ],
}


def detect_system_command(message: str) -> tuple[str, str] | None:
    """Detect if a message is a system command.

    Args:
        message: The user's message.

    Returns:
        Tuple of (command_type, extracted_target) or None.
    """
    lower = message.lower().strip()

    # Remove wake word prefix if present
    for prefix in ["parhi ", "parhi, ", "hey parhi ", "hey parhi, "]:
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
            break

    # Check each command type
    for cmd_type, triggers in SYSTEM_TRIGGERS.items():
        for trigger in triggers:
            if trigger in lower:
                # Extract target for open/close commands
                target = lower
                for t in triggers:
                    target = target.replace(t, "").strip()
                # Clean up common filler words
                for filler in ["the", "my", "please", "for me", "now", "app", "application"]:
                    target = target.replace(filler, "").strip()

                # If open_app was matched but the target looks like a website / URL, route to open_website
                if cmd_type == "open_app":
                    url_exts = [".com", ".org", ".net", ".io", ".co", ".edu", ".gov", ".ai", ".app", ".dev"]
                    if any(ext in target for ext in url_exts) or target.startswith(("http://", "https://", "www.")):
                        cmd_type = "open_website"

                return (cmd_type, target)

    return None
