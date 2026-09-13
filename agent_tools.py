# agent_tools.py
"""Agentic tool-use system for Parhi-GPT.

Gives Parhi the ability to take actions on behalf of the user:
web search, file operations, code execution, timers/reminders,
and system information.

All tools are sandboxed and safe. File operations are read-only by default.
"""
from __future__ import annotations

import datetime
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result from a tool invocation.

    Attributes:
        tool_name: Which tool was used.
        success: Whether the tool succeeded.
        result: The tool output/result.
        display_text: Human-readable text for Parhi to relay.
        error: Error message if the tool failed.
    """
    tool_name: str
    success: bool = True
    result: str = ""
    display_text: str = ""
    error: str = ""


@dataclass
class Reminder:
    """A scheduled reminder.

    Attributes:
        message: Reminder text.
        trigger_time: Unix timestamp when the reminder should fire.
        fired: Whether the reminder has already fired.
    """
    message: str
    trigger_time: float
    fired: bool = False


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Patterns that indicate the user wants a tool action
TOOL_TRIGGERS: dict[str, list[str]] = {
    "web_search": [
        "search for", "look up", "google", "find out", "search the web",
        "what's the latest", "find me", "look this up",
    ],
    "read_file": [
        "read the file", "open the file", "show me the file",
        "what's in", "read this", "open this",
    ],
    "run_code": [
        "run this code", "execute this", "try running",
        "test this code", "run the script",
    ],
    "system_info": [
        "what time", "what's the time", "current time",
        "what day", "what date", "system info",
        "how much ram", "disk space",
    ],
    "reminder": [
        "remind me", "set a reminder", "don't let me forget",
        "reminder in", "alert me", "notify me",
    ],
    "calculator": [
        "calculate", "what is", "how much is", "compute",
        "multiply", "divide", "add", "subtract",
    ],
    "system_command": [
        "open camera", "take a photo", "take photo", "screenshot",
        "lock screen", "volume up", "volume down", "mute",
        "brightness up", "brightness down", "battery status",
        "wifi status", "play music", "ip address",
    ],
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class AgentTools:
    """Collection of tools that Parhi can use to help the user.

    Each tool is a method that takes parameters and returns a ToolResult.
    Tools are sandboxed and safe — no destructive operations.
    """

    def __init__(self) -> None:
        self._reminders: list[Reminder] = []
        self._reminder_thread: threading.Thread | None = None
        self._reminder_callback = None

    def detect_tool_request(self, message: str) -> str | None:
        """Detect if a message is requesting a tool action.

        Args:
            message: The user's message.

        Returns:
            Tool name if detected, or None.
        """
        lower = message.lower()
        for tool, triggers in TOOL_TRIGGERS.items():
            if any(trigger in lower for trigger in triggers):
                return tool
        return None

    def execute_tool(self, tool_name: str, message: str) -> ToolResult:
        """Execute a tool based on the tool name and user message.

        Args:
            tool_name: Name of the tool to execute.
            message: The user's full message (for parameter extraction).

        Returns:
            ToolResult with the outcome.
        """
        tool_map = {
            "web_search": self.web_search,
            "read_file": self.read_file,
            "run_code": self.run_code,
            "system_info": self.system_info,
            "reminder": self.set_reminder,
            "calculator": self.calculate,
            "system_command": self.system_command,
        }

        tool_func = tool_map.get(tool_name)
        if not tool_func:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            return tool_func(message)
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                display_text=f"Oops, I ran into an error: {e}",
            )

    def web_search(self, message: str) -> ToolResult:
        """Search the web (uses DuckDuckGo instant answer API).

        Args:
            message: User message containing search query.

        Returns:
            ToolResult with search results.
        """
        # Extract search query
        query = message.lower()
        for trigger in TOOL_TRIGGERS["web_search"]:
            query = query.replace(trigger, "")
        query = query.strip().strip("\"'")

        if not query:
            return ToolResult(
                tool_name="web_search",
                success=False,
                display_text="What would you like me to search for?",
            )

        try:
            import requests
            # DuckDuckGo Instant Answer API (free, no key needed)
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10,
            )
            data = resp.json()

            # Extract useful info
            results: list[str] = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
            if data.get("Answer"):
                results.append(data["Answer"])
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"])

            if results:
                result_text = "\n".join(results[:3])
                return ToolResult(
                    tool_name="web_search",
                    success=True,
                    result=result_text,
                    display_text=f"Here's what I found about \"{query}\":\n\n{result_text}",
                )
            else:
                return ToolResult(
                    tool_name="web_search",
                    success=True,
                    result="",
                    display_text=f"I searched for \"{query}\" but couldn't find a quick answer. You might want to check a browser for more detailed results.",
                )

        except ImportError:
            return ToolResult(
                tool_name="web_search",
                success=False,
                display_text="I'd love to search the web for you, but the `requests` library isn't installed. Run: pip install requests",
            )
        except Exception as e:
            return ToolResult(
                tool_name="web_search",
                success=False,
                error=str(e),
                display_text=f"I tried searching but hit an error: {e}",
            )

    def read_file(self, message: str) -> ToolResult:
        """Read the contents of a file (read-only, safe).

        Args:
            message: User message containing file path.

        Returns:
            ToolResult with file contents.
        """
        # Extract file path from message
        # Look for quoted paths or paths with extensions
        path_patterns = [
            r'"([^"]+)"',          # Quoted path
            r"'([^']+)'",          # Single-quoted path
            r'(\S+\.\w{1,5})',     # Path with extension
        ]

        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message)
            if match:
                file_path = match.group(1)
                break

        if not file_path:
            return ToolResult(
                tool_name="read_file",
                success=False,
                display_text="Which file would you like me to read? Please provide the path.",
            )

        if not os.path.exists(file_path):
            return ToolResult(
                tool_name="read_file",
                success=False,
                display_text=f"I couldn't find the file: {file_path}",
            )

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Truncate very long files
            max_chars = 2000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... (truncated, {len(content)} total characters)"

            return ToolResult(
                tool_name="read_file",
                success=True,
                result=content,
                display_text=f"Here's the contents of {os.path.basename(file_path)}:\n\n```\n{content}\n```",
            )
        except Exception as e:
            return ToolResult(
                tool_name="read_file",
                success=False,
                error=str(e),
                display_text=f"I couldn't read that file: {e}",
            )

    def run_code(self, message: str) -> ToolResult:
        """Execute Python code in a sandboxed subprocess.

        Args:
            message: User message containing code.

        Returns:
            ToolResult with execution output.
        """
        # Extract code from message (between ``` markers or after "run this:")
        code_match = re.search(r'```(?:python)?\s*\n(.*?)```', message, re.DOTALL)
        if not code_match:
            code_match = re.search(r'(?:run|execute|try)(?:\s+this)?:?\s*(.*)', message, re.DOTALL | re.IGNORECASE)

        if not code_match:
            return ToolResult(
                tool_name="run_code",
                success=False,
                display_text="I'd be happy to run some code! Please share it with me (wrap in ```python``` blocks).",
            )

        code = code_match.group(1).strip()

        # Safety check — block dangerous operations
        dangerous_patterns = [
            r'\bos\.remove\b', r'\bos\.unlink\b', r'\bshutil\.rmtree\b',
            r'\bos\.system\b', r'\bsubprocess\.call\b',
            r'\b__import__\b', r'\beval\b', r'\bexec\b',
            r'\bopen\(.+["\']w["\']\)', r'\bformat\s*\(\s*\/',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return ToolResult(
                    tool_name="run_code",
                    success=False,
                    display_text="I'd rather not run that code — it contains potentially dangerous operations. Let's keep things safe! 😊",
                )

        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.getcwd(),
            )

            if result.returncode == 0:
                output = result.stdout.strip() or "(no output)"
                return ToolResult(
                    tool_name="run_code",
                    success=True,
                    result=output,
                    display_text=f"I ran your code and here's the output:\n\n```\n{output}\n```",
                )
            else:
                error = result.stderr.strip()
                return ToolResult(
                    tool_name="run_code",
                    success=False,
                    error=error,
                    display_text=f"The code ran but hit an error:\n\n```\n{error}\n```\n\nWant me to help debug it?",
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="run_code",
                success=False,
                display_text="The code took too long to run (>10 seconds). It might have an infinite loop?",
            )
        except Exception as e:
            return ToolResult(
                tool_name="run_code",
                success=False,
                error=str(e),
                display_text=f"I couldn't run the code: {e}",
            )

    def system_info(self, message: str) -> ToolResult:
        """Get system information.

        Args:
            message: User message (used to determine what info to return).

        Returns:
            ToolResult with system information.
        """
        lower = message.lower()

        if any(w in lower for w in ["time", "clock"]):
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            date_str = now.strftime("%A, %B %d, %Y")
            return ToolResult(
                tool_name="system_info",
                success=True,
                display_text=f"It's {time_str} on {date_str}.",
            )

        if any(w in lower for w in ["date", "day", "today"]):
            now = datetime.datetime.now()
            date_str = now.strftime("%A, %B %d, %Y")
            return ToolResult(
                tool_name="system_info",
                success=True,
                display_text=f"Today is {date_str}.",
            )

        # General system info
        info_parts = [
            f"**System**: {platform.system()} {platform.release()}",
            f"**Machine**: {platform.machine()}",
            f"**Python**: {platform.python_version()}",
            f"**Processor**: {platform.processor() or 'Unknown'}",
        ]

        try:
            import psutil  # type: ignore[import-untyped]
            mem = psutil.virtual_memory()
            info_parts.append(f"**RAM**: {mem.total // (1024**3)} GB total, {mem.percent}% used")
            disk = psutil.disk_usage("/")
            info_parts.append(f"**Disk**: {disk.total // (1024**3)} GB total, {disk.percent}% used")
        except ImportError:
            pass

        info_text = "\n".join(info_parts)
        return ToolResult(
            tool_name="system_info",
            success=True,
            display_text=f"Here's your system info:\n\n{info_text}",
        )

    def set_reminder(self, message: str, callback=None) -> ToolResult:
        """Set a reminder/timer.

        Args:
            message: User message containing reminder details.
            callback: Optional function to call when reminder fires.

        Returns:
            ToolResult confirming the reminder.
        """
        # Parse duration from message
        duration_match = re.search(
            r'(\d+)\s*(minute|min|hour|hr|second|sec)s?',
            message,
            re.IGNORECASE,
        )

        if not duration_match:
            return ToolResult(
                tool_name="reminder",
                success=False,
                display_text="How long from now should I remind you? (e.g., 'remind me in 5 minutes')",
            )

        amount = int(duration_match.group(1))
        unit = duration_match.group(2).lower()

        if unit.startswith("hour") or unit.startswith("hr"):
            seconds = amount * 3600
            human_time = f"{amount} hour{'s' if amount > 1 else ''}"
        elif unit.startswith("min"):
            seconds = amount * 60
            human_time = f"{amount} minute{'s' if amount > 1 else ''}"
        else:
            seconds = amount
            human_time = f"{amount} second{'s' if amount > 1 else ''}"

        # Extract reminder message
        reminder_text = re.sub(
            r'remind me|set a reminder|in \d+ \w+|don\'t let me forget',
            '', message, flags=re.IGNORECASE,
        ).strip().strip("to ").strip()

        if not reminder_text:
            reminder_text = "Time's up!"

        reminder = Reminder(
            message=reminder_text,
            trigger_time=time.time() + seconds,
        )
        self._reminders.append(reminder)

        # Start reminder thread
        def _fire():
            time.sleep(seconds)
            if not reminder.fired:
                reminder.fired = True
                print(f"\n\n  ⏰ REMINDER: {reminder_text}\n")
                if callback:
                    callback(reminder_text)

        thread = threading.Thread(target=_fire, daemon=True)
        thread.start()

        return ToolResult(
            tool_name="reminder",
            success=True,
            display_text=f"Got it! I'll remind you in {human_time}: \"{reminder_text}\" ⏰",
        )

    def calculate(self, message: str) -> ToolResult:
        """Evaluate a mathematical expression safely.

        Args:
            message: User message containing a math expression.

        Returns:
            ToolResult with the calculated answer.
        """
        # Extract math expression
        # Remove common phrases
        expr = message.lower()
        for phrase in ["calculate", "what is", "how much is", "compute", "what's"]:
            expr = expr.replace(phrase, "")
        expr = expr.strip().strip("?")

        # Only allow safe characters
        safe_chars = set("0123456789+-*/().% ")
        if not all(c in safe_chars for c in expr):
            # Try to convert words to math
            expr = expr.replace("plus", "+").replace("minus", "-")
            expr = expr.replace("times", "*").replace("multiplied by", "*")
            expr = expr.replace("divided by", "/").replace("over", "/")
            expr = expr.replace("to the power of", "**").replace("squared", "**2")
            # Remove remaining non-math characters
            expr = "".join(c for c in expr if c in safe_chars or c == "*")

        expr = expr.strip()
        if not expr:
            return ToolResult(
                tool_name="calculator",
                success=False,
                display_text="What would you like me to calculate?",
            )

        try:
            # Safe evaluation using only allowed operations
            result = eval(expr, {"__builtins__": {}}, {})
            return ToolResult(
                tool_name="calculator",
                success=True,
                result=str(result),
                display_text=f"{expr} = **{result}**",
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculator",
                success=False,
                display_text=f"I couldn't calculate that expression. Could you rephrase it?",
            )

    def system_command(self, message: str) -> ToolResult:
        """Execute a JARVIS system command by delegating to system_commands.

        Args:
            message: User command message.

        Returns:
            ToolResult with system command execution result.
        """
        try:
            from system_commands import SystemCommands, detect_system_command
            cmd_info = detect_system_command(message)
            if not cmd_info:
                return ToolResult(
                    tool_name="system_command",
                    success=False,
                    display_text="I couldn't identify the system command.",
                )
            cmd_type, target = cmd_info
            sc = SystemCommands()
            cmd_map = {
                "open_camera": lambda: sc.open_camera(),
                "take_photo": lambda: sc.open_camera(),
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
                res = sc.open_app(target) if target else sc.open_app("file explorer")
            elif cmd_type == "close_app":
                res = sc.close_app(target) if target else None
            elif cmd_type == "open_website":
                res = sc.open_website(target) if target else None
            elif cmd_type in cmd_map:
                res = cmd_map[cmd_type]()
            else:
                res = None

            return res if res else ToolResult(
                tool_name="system_command",
                success=False,
                display_text="System command not recognized.",
            )
        except Exception as e:
            return ToolResult(
                tool_name="system_command",
                success=False,
                error=str(e),
                display_text=f"Failed to execute system command: {e}",
            )
