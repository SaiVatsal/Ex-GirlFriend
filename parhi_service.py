# parhi_service.py
"""Always-on background service for Parhi-GPT.

Runs silently in the background (use ``pythonw parhi_service.py``),
continuously listens for the wake word "Parhi", processes commands,
and speaks responses via TTS.

Features:
  - Auto-starts with Windows (via install_startup.py)
  - Startup greeting ("Good morning, Vatsal! Parhi is online.")
  - Wake word detection → command capture → response
  - Background voice logging (audio only, no camera)
  - System tray icon for status and quit

Usage:
    pythonw parhi_service.py          # Silent background mode
    python parhi_service.py           # Foreground with console output
    python parhi_service.py --no-tray # Skip system tray icon
"""
from __future__ import annotations

import argparse
import datetime
import os
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

# Ensure project root is on path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# .env loader (same as chat.py)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    env_file = os.path.join(_project_root, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_dotenv()


# ---------------------------------------------------------------------------
# Startup greeting
# ---------------------------------------------------------------------------

def get_greeting_text() -> str:
    """Generate a time-aware startup greeting.

    Returns:
        A greeting string like "Good morning, Vatsal! Parhi is online."
    """
    hour = datetime.datetime.now().hour

    if hour < 6:
        greeting = "Hey there, night owl"
    elif hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    elif hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Hey there"

    # Try to get user's name from memory
    name = ""
    try:
        memory_path = os.path.join(_project_root, "parhi_memory.json")
        if os.path.exists(memory_path):
            import json
            with open(memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = data.get("user_profile", {})
            name = profile.get("preferred_name") or profile.get("name", "")
    except Exception:
        pass

    if name:
        return f"{greeting}, {name.title()}! Parhi is online and ready to assist you."
    else:
        return f"{greeting}! Parhi is online and ready to assist you."


def speak_greeting() -> None:
    """Speak the startup greeting aloud using TTS."""
    text = get_greeting_text()
    print(f"  🔊 {text}")

    try:
        # Try pyttsx3 first (lightweight, no GPU needed)
        import pyttsx3  # type: ignore[import-untyped]
        engine = pyttsx3.init()
        # Set a pleasant voice
        voices = engine.getProperty("voices")
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass

    try:
        # Fallback: Windows SAPI via PowerShell
        import subprocess
        ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{text}")'
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=30,
        )
    except Exception as e:
        print(f"  [TTS error: {e}]")


# ---------------------------------------------------------------------------
# System tray icon
# ---------------------------------------------------------------------------

def create_tray_icon(stop_event: threading.Event) -> None:
    """Create a system tray icon for Parhi.

    Args:
        stop_event: Threading event to signal shutdown.
    """
    try:
        import pystray  # type: ignore[import-untyped]
        from PIL import Image, ImageDraw  # type: ignore[import-untyped]

        # Create a simple icon (blue circle with "P")
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, 60, 60], fill=(50, 120, 255, 255))
        # Draw a "P" in the center
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 12), "P", fill=(255, 255, 255, 255), font=font)

        def on_quit(icon, item):
            icon.stop()
            stop_event.set()

        def on_status(icon, item):
            pass  # Status shown in tooltip

        menu = pystray.Menu(
            pystray.MenuItem("Parhi — Online ✅", on_status, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Parhi", on_quit),
        )

        icon = pystray.Icon(
            "parhi",
            img,
            "Parhi AI — Listening 🎤",
            menu,
        )
        icon.run()

    except ImportError:
        print("[service] pystray not installed — no system tray icon")
        print("  pip install pystray Pillow")
        # Just wait for stop event
        stop_event.wait()
    except Exception as e:
        print(f"[service] Tray icon error: {e}")
        stop_event.wait()


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

class ParhiCommandHandler:
    """Handles voice commands by routing to chat engine or system commands.

    Bridges the wake word detector with the Parhi chat/tool infrastructure.
    """

    def __init__(self) -> None:
        self._cli = None
        self._sys_commands = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy-initialize the Parhi chat engine and system commands."""
        if self._initialized:
            return

        print("  [Loading Parhi brain...]")

        try:
            from system_commands import SystemCommands, detect_system_command
            self._sys_commands = SystemCommands()
            self._detect_system = detect_system_command
            print("  [✓] System commands loaded")
        except Exception as e:
            print(f"  [✗] System commands failed: {e}")

        try:
            from chat import ParhiCLI
            checkpoint = os.path.join(_project_root, "parhi_model.pt")
            if not os.path.exists(checkpoint):
                # Try legacy name
                checkpoint = os.path.join(_project_root, "prarthana_model.pt")
            if os.path.exists(checkpoint):
                self._cli = ParhiCLI(
                    checkpoint_path=checkpoint,
                    config_overrides={
                        "enable_streaming": False,
                        "show_mini_status": False,
                    },
                )
                print("  [✓] Parhi brain loaded")
            else:
                print("  [✗] No model checkpoint found — system commands only")
        except Exception as e:
            print(f"  [✗] Brain load error: {e}")

        self._initialized = True

    def handle_command(self, command: str) -> None:
        """Process a voice command and respond.

        Args:
            command: The transcribed command text.
        """
        self.initialize()

        if not command.strip():
            return

        # Clean wake word from command
        lower = command.lower().strip()
        for prefix in ["parhi ", "parhi, ", "hey parhi ", "hey parhi, "]:
            if lower.startswith(prefix):
                command = command[len(prefix):]
                lower = command.lower().strip()
                break

        # Check for system commands first
        if self._sys_commands and self._detect_system:
            result = self._detect_system(command)
            if result:
                cmd_type, target = result
                response = self._execute_system_command(cmd_type, target)
                if response:
                    print(f"\n  Parhi: {response}")
                    self._speak(response)
                    return

        # Fall back to chat engine
        if self._cli:
            try:
                res = self._cli.respond(command)
                response = res[0] if isinstance(res, tuple) else res
                print(f"\n  Parhi: {response}")
                self._speak(response)
            except Exception as e:
                print(f"  [Error: {e}]")
        else:
            print(f"  Parhi: I heard you say \"{command}\", but my brain isn't loaded yet.")

    def _execute_system_command(self, cmd_type: str, target: str) -> str:
        """Execute a system command and return the display text.

        Args:
            cmd_type: Type of system command.
            target: Target app/value.

        Returns:
            Response text.
        """
        sc = self._sys_commands

        cmd_map = {
            "open_camera": lambda: sc.open_camera(),
            "take_photo": lambda: sc.open_camera(),  # Opens camera for photo
            "screenshot": lambda: sc.take_screenshot(),
            "lock_screen": lambda: sc.lock_screen(),
            "volume_up": lambda: sc.volume_control("up"),
            "volume_down": lambda: sc.volume_control("down"),
            "volume_mute": lambda: sc.volume_control("mute"),
            "brightness_up": lambda: sc.brightness_control("up"),
            "brightness_down": lambda: sc.brightness_control("down"),
            "shutdown": lambda: sc.power_command("shutdown"),
            "restart": lambda: sc.power_command("restart"),
            "sleep": lambda: sc.power_command("sleep"),
            "running_apps": lambda: sc.list_running_apps(),
            "battery": lambda: sc.battery_status(),
            "wifi": lambda: sc.wifi_status(),
            "play_music": lambda: sc.play_music(),
            "ip_address": lambda: sc.get_ip_address(),
            "empty_recycle_bin": lambda: sc.empty_recycle_bin(),
        }

        if cmd_type == "open_app":
            result = sc.open_app(target) if target else sc.open_app("file explorer")
        elif cmd_type == "close_app":
            result = sc.close_app(target) if target else None
        elif cmd_type == "open_website":
            result = sc.open_website(target) if target else None
        elif cmd_type in cmd_map:
            result = cmd_map[cmd_type]()
        else:
            return ""

        return result.display_text if result else ""

    def _speak(self, text: str) -> None:
        """Speak text aloud using available TTS.

        Args:
            text: Text to speak.
        """
        # Strip markdown formatting for speech
        import re
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean = re.sub(r'```[^`]*```', '', clean)
        clean = re.sub(r'[🚀📸🔒🔇🔉🔊☀️🌙⚠️✅🗑️✨💻🖥️📡📶🌐🔋⚡👋🔄😴🎤📝]', '', clean)
        clean = clean.strip()

        if not clean:
            return

        try:
            import pyttsx3  # type: ignore[import-untyped]
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break
            engine.setProperty("rate", 170)
            engine.say(clean)
            engine.runAndWait()
            return
        except Exception:
            pass

        try:
            import subprocess
            # Escape quotes in text
            safe_text = clean.replace('"', "'").replace('\n', ' ')[:500]
            ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{safe_text}")'
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=30,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

def run_service(use_tray: bool = True) -> None:
    """Run the Parhi background service.

    Args:
        use_tray: Whether to show a system tray icon.
    """
    print()
    print("╔════════════════════════════════════════╗")
    print("║  🤖 Parhi Background Service          ║")
    print("║  Always-on AI Assistant                ║")
    print("╚════════════════════════════════════════╝")
    print()

    # Step 1: Speak startup greeting
    speak_greeting()

    # Step 2: Initialize command handler
    handler = ParhiCommandHandler()

    # Step 3: Start wake word detector
    try:
        from wake_word import WakeWordDetector, WakeWordConfig

        ww_config = WakeWordConfig(
            wake_word="parhi",
            voice_log_enabled=True,
            voice_log_dir=os.path.join(_project_root, "voice_logs"),
        )
        detector = WakeWordDetector(ww_config)
        detector.on_wake(handler.handle_command)
        detector.start()
        print("  [✓] Wake word detection active")
    except Exception as e:
        print(f"  [✗] Wake word detection failed: {e}")
        print("  Install: pip install sounddevice soundfile faster-whisper")

    # Step 4: Run system tray (blocks main thread) or wait
    stop_event = threading.Event()

    if use_tray:
        # Tray icon blocks on main thread
        create_tray_icon(stop_event)
    else:
        # Just wait forever
        print("\n  Press Ctrl+C to stop Parhi.\n")
        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass

    # Cleanup
    print("\n  Parhi is shutting down. Goodbye! 👋")
    try:
        detector.stop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parhi Background Service")
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Run without system tray icon.",
    )
    args = parser.parse_args()

    run_service(use_tray=not args.no_tray)


if __name__ == "__main__":
    main()
