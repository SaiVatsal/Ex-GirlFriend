# screen_vision.py
"""Real-time screen capture and analysis for Prarthana-GPT.

Captures the user's screen with minimal latency using ``mss``,
detects meaningful changes via pixel-diff thresholds, and sends
screenshots to a vision API (Gemini or OpenAI) for natural language
understanding.

Falls back gracefully if dependencies or API keys are missing.
"""
from __future__ import annotations

import base64
import io
import os
import time
import hashlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreenFrame:
    """A captured screen frame with metadata.

    Attributes:
        image_bytes: Raw PNG image bytes.
        timestamp: Capture time.
        hash: Content hash for change detection.
        description: Natural language description (filled by vision API).
    """
    image_bytes: bytes = b""
    timestamp: float = field(default_factory=time.time)
    hash: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Screen capture engine
# ---------------------------------------------------------------------------

class ScreenVision:
    """Real-time screen capture and AI-powered visual understanding.

    Uses ``mss`` for zero-lag screen grabs and routes screenshots through
    a vision API for natural language descriptions. Maintains a ring buffer
    of recent frames for temporal context.

    Gracefully degrades to no-op if:
    - ``mss`` / ``Pillow`` are not installed
    - No API key is configured
    """

    def __init__(
        self,
        api_provider: str = "gemini",
        api_key: str | None = None,
        buffer_size: int = 5,
        change_threshold: float = 0.05,
    ) -> None:
        """Initialize screen vision.

        Args:
            api_provider: Which vision API to use ("gemini" or "openai").
            api_key: API key for the vision service. Auto-detects from
                env vars if None: GEMINI_API_KEY or OPENAI_API_KEY.
            buffer_size: Number of frames to keep in ring buffer.
            change_threshold: Fraction of pixels that must change to
                trigger a new analysis (0.0-1.0).
        """
        self.api_provider = api_provider
        self.api_key = api_key or self._detect_api_key(api_provider)
        self._buffer: list[ScreenFrame] = []
        self._buffer_size = buffer_size
        self._change_threshold = change_threshold
        self._last_hash = ""
        self.active = False

        # Try to import dependencies
        try:
            import mss as _mss  # noqa: F401
            from PIL import Image as _Image  # noqa: F401
            self._mss_mod = _mss
            self._pil_mod = _Image
            self.active = True
        except ImportError:
            self._mss_mod = None
            self._pil_mod = None

        if self.active and not self.api_key:
            print("[screen_vision] No API key found — screen description disabled.")
            print("[screen_vision] Set GEMINI_API_KEY or OPENAI_API_KEY environment variable.")

    @staticmethod
    def _detect_api_key(provider: str) -> str | None:
        """Try to find an API key from environment variables.

        Args:
            provider: The API provider name.

        Returns:
            API key string or None.
        """
        if provider == "gemini":
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    def capture(self) -> ScreenFrame | None:
        """Capture the current screen.

        Returns:
            A ScreenFrame with image bytes and hash, or None if capture fails.
        """
        if not self.active or not self._mss_mod:
            return None

        try:
            with self._mss_mod.mss() as sct:
                # Capture the primary monitor
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)

                # Convert to PIL Image then to PNG bytes
                img = self._pil_mod.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb,
                )

                # Resize for API efficiency (max 1280px wide)
                max_width = 1280
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.resize(new_size, self._pil_mod.LANCZOS)

                # Convert to PNG bytes
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                image_bytes = buf.getvalue()

                # Content hash for change detection
                content_hash = hashlib.md5(image_bytes).hexdigest()

                frame = ScreenFrame(
                    image_bytes=image_bytes,
                    hash=content_hash,
                )

                # Add to ring buffer
                self._buffer.append(frame)
                if len(self._buffer) > self._buffer_size:
                    self._buffer = self._buffer[-self._buffer_size:]

                return frame

        except Exception as e:
            print(f"[screen_vision] capture error: {e}")
            return None

    def has_changed(self) -> bool:
        """Check if the screen content has meaningfully changed since last check.

        Returns:
            True if the screen has changed beyond the threshold.
        """
        frame = self.capture()
        if not frame:
            return False

        changed = frame.hash != self._last_hash
        self._last_hash = frame.hash
        return changed

    def describe(self, context: str = "") -> str:
        """Capture and describe the current screen content.

        Args:
            context: Optional context about what the user is doing.

        Returns:
            Natural language description of the screen, or error message.
        """
        if not self.active:
            return "[Screen vision not available — install mss and Pillow]"

        frame = self.capture()
        if not frame:
            return "[Could not capture screen]"

        if not self.api_key:
            return "[No API key configured for screen analysis. Set GEMINI_API_KEY or OPENAI_API_KEY]"

        # Send to vision API
        description = self._analyze_with_api(frame.image_bytes, context)
        frame.description = description

        return description

    def answer_about_screen(self, question: str) -> str:
        """Answer a specific question about the current screen content.

        Args:
            question: The user's question about what's on screen.

        Returns:
            Answer based on visual analysis.
        """
        context = f"The user asks: {question}"
        return self.describe(context)

    def _analyze_with_api(self, image_bytes: bytes, context: str = "") -> str:
        """Send image to vision API for analysis.

        Args:
            image_bytes: PNG image bytes.
            context: Optional context string.

        Returns:
            Natural language description from the API.
        """
        if self.api_provider == "gemini":
            return self._analyze_gemini(image_bytes, context)
        elif self.api_provider == "openai":
            return self._analyze_openai(image_bytes, context)
        return "[Unsupported vision API provider]"

    def _analyze_gemini(self, image_bytes: bytes, context: str = "") -> str:
        """Analyze screenshot using Google Gemini API.

        Args:
            image_bytes: PNG image bytes.
            context: Optional context.

        Returns:
            Description from Gemini.
        """
        try:
            import requests
        except ImportError:
            return "[requests library not installed]"

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "You are Prarthana, looking at the user's screen. "
            "Describe what you see in a natural, conversational way — "
            "as if you're a friend looking over their shoulder. "
            "Be specific about what apps are open, what text is visible, "
            "any errors or notifications, and what the user seems to be working on. "
            "Keep it concise but helpful."
        )
        if context:
            prompt += f"\n\nAdditional context: {context}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": b64_image,
                        }
                    },
                ]
            }],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 300,
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[Gemini vision error: {e}]"

    def _analyze_openai(self, image_bytes: bytes, context: str = "") -> str:
        """Analyze screenshot using OpenAI GPT-4o API.

        Args:
            image_bytes: PNG image bytes.
            context: Optional context.

        Returns:
            Description from GPT-4o.
        """
        try:
            import requests
        except ImportError:
            return "[requests library not installed]"

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "You are Prarthana, looking at the user's screen. "
            "Describe what you see naturally, like a friend looking over their shoulder. "
            "Be specific and helpful."
        )
        if context:
            prompt += f"\n\nContext: {context}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            "max_tokens": 300,
        }

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OpenAI vision error: {e}]"

    def get_recent_descriptions(self, n: int = 3) -> list[str]:
        """Get descriptions from recent frames.

        Args:
            n: Number of recent descriptions to return.

        Returns:
            List of description strings.
        """
        described = [f for f in self._buffer if f.description]
        return [f.description for f in described[-n:]]
