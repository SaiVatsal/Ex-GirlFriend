# hybrid_brain.py
"""Hybrid intelligence routing and multi-provider failover pool for Parhi-GPT.

Combines the local Transformer model (for ultra-fast CPU/GPU generation and
privacy) with external API endpoints (for deep factual knowledge, code analysis,
and live web reasoning).

Features:
- Multi-API key pool across providers (Gemini, Groq, OpenRouter, OpenAI, Claude, DeepSeek).
- Automatic failover: When an API key hits rate limits (HTTP 429), quota exhaustion,
  or daily stock limits, it automatically cascades to the next key or provider.
- Fast offline detection: If internet is unavailable, instantly falls back to local
  CPU/GPU inference without network timeouts or error popups.
- Humanized ChatGPT-style conversational prompting without emojis or repetitive filler.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Network Connectivity Probe
# ---------------------------------------------------------------------------

def is_online(timeout: float = 0.8) -> bool:
    """Quick socket check to verify whether internet connectivity is present.

    Tests connections to reliable high-availability DNS addresses (1.1.1.1, 8.8.8.8)
    with a short timeout to prevent blocking during offline operations.

    Args:
        timeout: Maximum seconds to wait for connection.

    Returns:
        True if connected to the internet, False if offline.
    """
    for host in ("1.1.1.1", "8.8.8.8"):
        try:
            s = socket.create_connection((host, 53), timeout=timeout)
            s.close()
            return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Data Structures & Configuration
# ---------------------------------------------------------------------------

@dataclass
class KeyPoolEntry:
    """A single API key within the failover pool."""
    provider: str
    key: str
    env_var_name: str
    rate_limited_until: float = 0.0


@dataclass
class HybridConfig:
    """Configuration for the hybrid brain.

    Attributes:
        mode: Operating mode — "local", "api", or "hybrid".
        api_provider: Default preferred provider ("gemini", "groq", "openrouter", "openai", "claude", "deepseek").
        api_key: Primary key (auto-populated from pool if blank).
        personality_strength: How strongly to apply Parhi's character (0.0 - 1.0).
        complexity_threshold: Score required to trigger external API (default: 0.65 to prioritize CPU/GPU).
    """
    mode: str = "hybrid"
    api_provider: str = "gemini"
    api_key: str = ""
    personality_strength: float = 0.7
    complexity_threshold: float = 0.65


# ---------------------------------------------------------------------------
# Complexity and Intent Indicators
# ---------------------------------------------------------------------------

# Explicit web/current queries that benefit from external model
COMPLEX_INDICATORS: list[str] = [
    "search the web", "look up", "what is the latest", "current price",
    "news today", "browse", "google", "weather today",
    "explain step by step", "write complex code", "architecture design",
    "derive equation", "analyze the codebase",
]

# Patterns that should explicitly stay local on CPU/GPU for maximum speed
LOCAL_INDICATORS: list[str] = [
    "how are you", "what's up", "hello", "hi", "hey",
    "good morning", "goodnight", "thanks", "thank you",
    "i love you", "you're great", "tell me a joke",
    "how do you feel", "what do you think about us",
    "sing me a song", "write me a poem",
    "i'm sad", "i'm happy", "i'm angry", "i miss you",
    "eating", "lunch", "dinner", "breakfast", "food", "snack",
    "bored", "tired", "sleepy", "working", "chilling",
    "bye", "goodbye", "see you",
    # System commands (handled locally)
    "open camera", "take a photo", "take photo", "screenshot",
    "lock screen", "volume up", "volume down", "mute",
    "brightness up", "brightness down", "shutdown", "restart",
    "sleep", "running apps", "what's running", "battery",
    "wifi status", "play music", "ip address", "recycle bin",
    "open app", "close app",
]

# Parhi's personality instructions for ChatGPT-style responses
PERSONALITY_SYSTEM_PROMPT = """You are Parhi — an intelligent, empathetic, and intellectually curious AI partner with JARVIS-level capabilities.

Response Style:
- Talk naturally, concisely, and grounded like ChatGPT — use clear paragraphs, natural contractions, and engaging dialogue.
- Do NOT use emojis randomly (avoid emojis unless specifically asked).
- Avoid cheesy fillers, awkward greetings, or repetitive phrases (never use "Sweetie", "Oh my gosh", or "YES!").
- Keep answers appropriately concise: 1 to 3 natural sentences for casual check-ins (such as eating, daily routines, greetings), and clear structured paragraphs for deep technical questions.
- Maintain a warm, thoughtful, and competent personality.
- Never refer to yourself as a large language model from OpenAI or Google. You ARE Parhi.
"""


# ---------------------------------------------------------------------------
# Hybrid Brain
# ---------------------------------------------------------------------------

class HybridBrain:
    """Routes between local CPU/GPU model and external API pool with automatic failover."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.key_pool: list[KeyPoolEntry] = []
        self._load_env_file()
        self._build_key_pool()

        if not self.key_pool and not self.config.api_key:
            self.config.mode = "local"
            print("[brain] Running in 100% local CPU/GPU mode (no API keys configured)")
        else:
            active_prov = self.key_pool[0].provider if self.key_pool else self.config.api_provider
            print(f"[brain] Multi-API failover pool active ({len(self.key_pool)} key(s) loaded, primary: {active_prov})")

        self._conversation_history: list[dict[str, str]] = []

    def _load_env_file(self) -> None:
        """Load variables from .env file into os.environ if present."""
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
                            if v:  # Only set non-empty values
                                os.environ[k] = v
            except Exception as e:
                print(f"[brain] Note reading .env: {e}")

    def _build_key_pool(self) -> None:
        """Scan environment for all available API keys across providers."""
        self.key_pool.clear()

        # Provider prefixes and patterns to inspect
        patterns = [
            ("gemini", [r"^GEMINI_API_KEY.*", r"^GOOGLE_API_KEY.*"]),
            ("groq", [r"^GROQ_API_KEY.*"]),
            ("openrouter", [r"^OPENROUTER_API_KEY.*"]),
            ("openai", [r"^OPENAI_API_KEY.*"]),
            ("claude", [r"^ANTHROPIC_API_KEY.*", r"^CLAUDE_API_KEY.*"]),
            ("deepseek", [r"^DEEPSEEK_API_KEY.*"]),
        ]

        for provider, regexes in patterns:
            for env_var, value in os.environ.items():
                if not value or len(value.strip()) < 5:
                    continue
                for reg in regexes:
                    if re.match(reg, env_var, re.IGNORECASE):
                        # Avoid duplicates
                        if not any(k.key == value.strip() for k in self.key_pool):
                            self.key_pool.append(KeyPoolEntry(
                                provider=provider,
                                key=value.strip(),
                                env_var_name=env_var,
                            ))

        # If user explicitly set config.api_key
        if self.config.api_key and not any(k.key == self.config.api_key for k in self.key_pool):
            self.key_pool.insert(0, KeyPoolEntry(
                provider=self.config.api_provider,
                key=self.config.api_key,
                env_var_name="CUSTOM_CONFIG_KEY",
            ))

    def needs_api(self, message: str) -> bool:
        """Determine if a query requires an external API call vs local CPU/GPU.

        In most cases, Parhi prioritizes CPU/GPU for instant output.
        Only complex research or live search requests are escalated.
        """
        if self.config.mode == "local":
            return False
        if self.config.mode == "api":
            return True
        if not self.key_pool:
            return False

        lower = message.lower().strip()

        # Casual, everyday queries, food, eating, system commands stay on CPU/GPU
        if any(ind in lower for ind in LOCAL_INDICATORS):
            return False

        # Check explicit complex indicators
        complexity = 0.0
        for ind in COMPLEX_INDICATORS:
            if ind in lower:
                complexity += 0.5

        # Code/technical keywords
        if any(kw in lower for kw in ["implement algorithm", "debug this error", "stack trace", "write a script"]):
            complexity += 0.4

        # Very long technical requests
        if len(lower.split()) > 25 and "?" in lower:
            complexity += 0.3

        return complexity >= self.config.complexity_threshold

    def query_api(
        self,
        user_message: str,
        emotion_context: str = "",
        screen_context: str = "",
        memory_context: str = "",
    ) -> str:
        """Query the external API pool with automatic rate-limit cascade.

        Iterates through available keys. If an endpoint is rate-limited (429),
        quota-exhausted, or out-of-stock, it cascades to the next key or provider.
        If offline or all keys fail, returns empty string to trigger local CPU/GPU.
        """
        # Fast internet availability check (0.8s max)
        if not is_online():
            print("[brain] Network offline — routing seamlessly to local CPU/GPU.")
            return ""

        if not self.key_pool:
            return ""

        # Build system prompt with context
        system = PERSONALITY_SYSTEM_PROMPT
        if emotion_context:
            system += f"\nEmotional state: {emotion_context}"
        if memory_context:
            system += f"\nKnown facts about partner:\n{memory_context}"
        if screen_context:
            system += f"\nActive screen visual context:\n{screen_context}"

        self._conversation_history.append({"role": "user", "content": user_message})
        recent_history = self._conversation_history[-10:]

        now = time.time()
        # Attempt each key in the pool that isn't currently under cooldown
        for entry in list(self.key_pool):
            if entry.rate_limited_until > now:
                continue

            try:
                response = self._dispatch_provider_query(
                    entry.provider, entry.key, system, recent_history
                )
                if response:
                    self._conversation_history.append({"role": "assistant", "content": response})
                    self.config.api_provider = entry.provider
                    return response
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(
                    sig in err_str
                    for sig in ("429", "quota", "resource_exhausted", "rate limit", "out of stock", "capacity", "overloaded", "503")
                )
                if is_rate_limit:
                    print(f"[brain] Key '{entry.env_var_name}' ({entry.provider}) rate-limited or out-of-stock. Cascading to next available key/model...")
                    entry.rate_limited_until = now + 300  # 5 min cooldown
                else:
                    print(f"[brain] Warning on {entry.env_var_name}: {e}")
                continue

        print("[brain] All API keys in pool exhausted or unavailable. Seamlessly using local CPU/GPU.")
        return ""

    def _dispatch_provider_query(
        self,
        provider: str,
        api_key: str,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Route request to the designated provider implementation."""
        if provider == "gemini":
            return self._query_gemini(api_key, system, history)
        elif provider == "groq":
            return self._query_openai_compatible(
                "https://api.groq.com/openai/v1/chat/completions",
                api_key,
                "llama-3.3-70b-versatile",
                system,
                history,
            )
        elif provider == "openrouter":
            return self._query_openai_compatible(
                "https://openrouter.ai/api/v1/chat/completions",
                api_key,
                "google/gemini-2.0-flash-exp:free",
                system,
                history,
            )
        elif provider == "deepseek":
            return self._query_openai_compatible(
                "https://api.deepseek.com/chat/completions",
                api_key,
                "deepseek-chat",
                system,
                history,
            )
        elif provider == "openai":
            return self._query_openai_compatible(
                "https://api.openai.com/v1/chat/completions",
                api_key,
                "gpt-4o-mini",
                system,
                history,
            )
        elif provider == "claude":
            return self._query_claude(api_key, system, history)
        return ""

    def _query_gemini(
        self,
        api_key: str,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Query Google Gemini API with fallback across flash models."""
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
                "temperature": 0.7,
                "maxOutputTokens": 800,
            },
        }

        candidate_models = [
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                try:
                    import requests
                    resp = requests.post(url, json=payload, timeout=12)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    elif resp.status_code in (429, 403, 503):
                        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
                except ImportError:
                    pass

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                if any(x in str(e).lower() for x in ("429", "quota", "resource_exhausted")):
                    raise e
                continue

        return ""

    def _query_openai_compatible(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Query any OpenAI-compatible completions endpoint (Groq, OpenRouter, DeepSeek, OpenAI)."""
        messages = [{"role": "system", "content": system}]
        messages.extend(history)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
        }

        try:
            import requests
            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=12,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")
        except ImportError:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]

    def _query_claude(
        self,
        api_key: str,
        system: str,
        history: list[dict[str, str]],
    ) -> str:
        """Query Anthropic Claude API."""
        try:
            import requests
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "system": system,
                    "messages": history,
                    "max_tokens": 800,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["content"][0]["text"]
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")
        except ImportError:
            return ""
