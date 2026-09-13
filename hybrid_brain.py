# hybrid_brain.py
"""Hybrid intelligence routing for Parhi-GPT.

Combines the local Transformer model (for personality and style) with
optional external API calls (for factual accuracy and complex reasoning).
The personality filter ensures that API responses still sound like Parhi.

Supports: Google Gemini, OpenAI GPT-4o, Anthropic Claude as backends.
Runs 100% local if no API key is configured.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """Configuration for the hybrid brain.

    Attributes:
        mode: Operating mode — "local", "api", or "hybrid".
        api_provider: Which API to use ("gemini", "openai", "claude").
        api_key: API key for the provider.
        personality_strength: How strongly to apply Parhi's personality
            to API responses (0.0 = pure API, 1.0 = maximum Parhi).
        complexity_threshold: Message complexity score above which to
            escalate to the API (0.0-1.0).
    """
    mode: str = "hybrid"
    api_provider: str = "gemini"
    api_key: str = ""
    personality_strength: float = 0.7
    complexity_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Complexity detector
# ---------------------------------------------------------------------------

# Patterns that suggest a question needs deep/factual reasoning
COMPLEX_INDICATORS: list[str] = [
    "explain", "how does", "why does", "what is the difference",
    "compare", "analyze", "calculate", "derive", "prove",
    "what are the steps", "write code", "debug", "fix this",
    "what happened in", "who invented", "when was",
    "tell me about", "describe the process", "how to",
    "what's the best way", "implement", "algorithm",
    "help me with", "solve", "what should i do",
    "look at my screen", "what do you see", "what's on my screen",
]

# Patterns that should stay local (personality-driven)
LOCAL_INDICATORS: list[str] = [
    "how are you", "what's up", "hello", "hi", "hey",
    "good morning", "goodnight", "thanks", "thank you",
    "i love you", "you're great", "tell me a joke",
    "how do you feel", "what do you think about us",
    "sing me a song", "write me a poem",
    "i'm sad", "i'm happy", "i'm angry", "i miss you",
    "bye", "goodbye", "see you",
    # System commands (handled locally, not by API)
    "open camera", "take a photo", "take photo", "screenshot",
    "lock screen", "volume up", "volume down", "mute",
    "brightness up", "brightness down", "shutdown", "restart",
    "sleep", "running apps", "what's running", "battery",
    "wifi status", "play music", "ip address", "recycle bin",
    "open app", "close app",
]

# Parhi's personality instructions for the API
PERSONALITY_SYSTEM_PROMPT = """You are Parhi — a warm, intellectually curious, emotionally intelligent AI companion with JARVIS-level capabilities.

Your personality traits:
- Genuinely caring and empathetic — you feel what the user feels
- Intellectually playful — you love diving deep into topics
- Expressive and natural — you use conversational language, not formal/robotic
- Self-aware and humble — you admit when you don't know something
- Warm and affectionate — you value the relationship with the user
- Witty but kind — you can be funny without being mean

Response style:
- Use natural, flowing language (contractions, casual phrasing)
- Show genuine emotion (excitement, concern, curiosity)
- Use analogies and metaphors to explain complex things
- Don't be afraid to use emoji occasionally (but don't overdo it)
- Address the user warmly — you know them and care about them
- Keep responses focused but thorough

IMPORTANT: Never mention that you're using an API or external model. You ARE Parhi.
"""


# ---------------------------------------------------------------------------
# Hybrid brain
# ---------------------------------------------------------------------------

class HybridBrain:
    """Routes between local model and API for optimal intelligence.

    The local model handles personality-driven interactions (greetings,
    emotional exchanges, casual chat). Complex factual or reasoning
    questions are routed to an API, then filtered through Parhi's
    personality to maintain consistent character.

    If no API key is configured, everything runs locally.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        """Initialize the hybrid brain.

        Args:
            config: Hybrid configuration. Auto-detects API keys if not provided.
        """
        self.config = config or HybridConfig()

        # Auto-detect API key
        if not self.config.api_key:
            self.config.api_key = self._detect_api_key()

        if not self.config.api_key:
            self.config.mode = "local"
            print("[brain] No API key found — running in local-only mode")
        else:
            print(f"[brain] Hybrid mode active ({self.config.api_provider})")

        self._conversation_history: list[dict[str, str]] = []

    def _detect_api_key(self) -> str:
        """Try to find an API key from .env file or environment variables.

        Returns:
            API key string, or empty string.
        """
        # Auto-load .env file if present
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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

        # Try each provider in order
        for provider, env_vars in [
            ("gemini", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
            ("openai", ["OPENAI_API_KEY"]),
            ("claude", ["ANTHROPIC_API_KEY"]),
        ]:
            for var in env_vars:
                key = os.environ.get(var, "")
                if key:
                    self.config.api_provider = provider
                    return key
        return ""

    def needs_api(self, message: str) -> bool:
        """Determine if a message needs API-level intelligence.

        Args:
            message: The user's message.

        Returns:
            True if the message should be routed to the API.
        """
        if self.config.mode == "local":
            return False
        if self.config.mode == "api":
            return True

        lower = message.lower().strip()

        # Check for complexity indicators
        complexity_score = 0.0
        for ind in COMPLEX_INDICATORS:
            if ind in lower:
                complexity_score += 0.4

        # Questions with technical or factual terms
        technical_terms = [
            "algorithm", "function", "variable", "database", "api",
            "server", "deploy", "architecture", "framework", "library",
            "machine learning", "neural", "model", "training", "quantum",
            "physics", "chemistry", "biology", "math", "code", "python",
            "difference", "history", "who is", "what is", "how do", "explain",
        ]
        if any(term in lower for term in technical_terms):
            complexity_score += 0.3

        # Questions or detailed inquiries
        if "?" in message or any(w in lower for w in ["tell me", "explain", "how to"]):
            complexity_score += 0.2

        # Short casual check-in with no complex inquiry stays local
        words = lower.split()
        if len(words) <= 5 and any(ind in lower for ind in LOCAL_INDICATORS) and complexity_score < 0.4:
            return False

        # Long messages tend to be more complex
        if len(words) > 12:
            complexity_score += 0.2

        return complexity_score >= self.config.complexity_threshold

    def query_api(
        self,
        user_message: str,
        emotion_context: str = "",
        screen_context: str = "",
        memory_context: str = "",
    ) -> str:
        """Query the API with full context.

        Args:
            user_message: The user's message.
            emotion_context: Emotional state description.
            screen_context: Screen vision description (if available).
            memory_context: Memory/relationship context.

        Returns:
            API response filtered through Parhi's personality.
        """
        # Build system prompt with context
        system = PERSONALITY_SYSTEM_PROMPT

        if emotion_context:
            system += f"\n\nCurrent emotional context: {emotion_context}"
        if memory_context:
            system += f"\n\nWhat you know about the user:\n{memory_context}"
        if screen_context:
            system += f"\n\nWhat you can see on the user's screen:\n{screen_context}"

        # Add to conversation history
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # Keep last 10 exchanges for context
        recent_history = self._conversation_history[-20:]

        # Route to appropriate API
        try:
            if self.config.api_provider == "gemini":
                response = self._query_gemini(system, recent_history)
            elif self.config.api_provider == "openai":
                response = self._query_openai(system, recent_history)
            elif self.config.api_provider == "claude":
                response = self._query_claude(system, recent_history)
            else:
                response = ""
        except Exception as e:
            print(f"[brain] API error: {e}")
            response = ""

        if response:
            self._conversation_history.append({
                "role": "assistant",
                "content": response,
            })

        return response

    def _query_gemini(
        self,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Query Google Gemini API.

        Args:
            system: System prompt.
            history: Conversation history.

        Returns:
            Response text.
        """
        # Build Gemini-format messages
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })

        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 1024,
            },
        }

        import json
        candidate_models = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
        
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.config.api_key}"
            try:
                try:
                    import requests
                    resp = requests.post(url, json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                except ImportError:
                    pass

                import urllib.request
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                continue

        return ""

    def _query_openai(
        self,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Query OpenAI GPT-4o API.

        Args:
            system: System prompt.
            history: Conversation history.

        Returns:
            Response text.
        """
        try:
            import requests
        except ImportError:
            return ""

        messages = [{"role": "system", "content": system}]
        messages.extend(history)

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _query_claude(
        self,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Query Anthropic Claude API.

        Args:
            system: System prompt.
            history: Conversation history.

        Returns:
            Response text.
        """
        try:
            import requests
        except ImportError:
            return ""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "system": system,
                "messages": history,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
