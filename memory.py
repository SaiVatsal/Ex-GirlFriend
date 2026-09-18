# memory.py
"""Persistent conversation memory for Parhi-GPT.

Provides short-term (in-session) and long-term (cross-session) memory
that tracks user preferences, emotional patterns, conversation facts,
and relationship history. Persists to a JSON file on disk.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationFact:
    """A fact extracted from conversation.

    Attributes:
        category: Type of fact (preference, personal, topic, opinion).
        key: Short label (e.g., "favorite_color").
        value: The fact content (e.g., "blue").
        confidence: How confident we are this is correct (0.0-1.0).
        timestamp: When this fact was recorded.
        source_message: The user message this was extracted from.
    """
    category: str
    key: str
    value: str
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    source_message: str = ""


@dataclass
class EmotionalSnapshot:
    """A snapshot of emotional state at a point in time.

    Attributes:
        emotion: Detected emotion category.
        intensity: Strength (0.0-1.0).
        timestamp: When this was recorded.
        message_preview: First 50 chars of the triggering message.
    """
    emotion: str
    intensity: float
    timestamp: float = field(default_factory=time.time)
    message_preview: str = ""


@dataclass
class SessionSummary:
    """Summary of a past conversation session.

    Attributes:
        date: ISO date string of the session.
        duration_minutes: How long the session lasted.
        message_count: Total messages exchanged.
        topics_discussed: List of topics covered.
        overall_mood: Dominant emotion of the session.
        highlights: Key moments or important exchanges.
    """
    date: str = ""
    duration_minutes: float = 0.0
    message_count: int = 0
    topics_discussed: list[str] = field(default_factory=list)
    overall_mood: str = "neutral"
    highlights: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fact extraction patterns
# ---------------------------------------------------------------------------

# Patterns to extract user facts from messages
FACT_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, category, key_template)
    (r"my name is (\w+)", "personal", "name"),
    (r"i'?m (\w+)", "personal", "name"),  # "I'm Vatsal"
    (r"call me (\w+)", "personal", "preferred_name"),
    (r"i (?:really )?(?:love|like|enjoy) (\w[\w\s]{1,30})", "preference", "likes"),
    (r"i (?:hate|dislike|don't like) (\w[\w\s]{1,30})", "preference", "dislikes"),
    (r"my favorite (\w+) is ([\w\s]+)", "preference", "favorite_{0}"),
    (r"i'?m (?:a |an )?(\w+(?:\s\w+)?)\s*(?:student|developer|engineer|designer|artist|teacher|doctor|scientist)", "personal", "occupation"),
    (r"i (?:study|am studying|am learning) ([\w\s]+)", "personal", "studying"),
    (r"i'?m (\d{1,2}) years old", "personal", "age"),
    (r"i (?:live|stay) in ([\w\s]+)", "personal", "location"),
    (r"i work (?:at|for|in) ([\w\s]+)", "personal", "workplace"),
    (r"i'?m eating ([\w\s]{1,30})", "activity", "current_food"),
    (r"i ate ([\w\s]{1,30})", "activity", "recent_food"),
    (r"my favorite food is ([\w\s]{1,30})", "preference", "favorite_food"),
    (r"i (?:like to eat|love eating) ([\w\s]{1,30})", "preference", "favorite_food"),
    (r"my dog'?s name is (\w+)", "personal", "dog_name"),
    (r"my cat'?s name is (\w+)", "personal", "cat_name"),
    (r"my pet'?s name is (\w+)", "personal", "pet_name"),
]


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Persistent memory system for Parhi-GPT.

    Manages three layers of memory:
    1. **Session memory**: Current conversation facts, emotions, exchanges
    2. **User profile**: Accumulated facts about the user across sessions
    3. **Relationship history**: Past session summaries and emotional patterns

    All data is periodically flushed to a JSON file for persistence.
    """

    def __init__(self, memory_path: str = "parhi_memory.json") -> None:
        """Initialize memory, loading from disk if available.

        Args:
            memory_path: Path to the JSON persistence file.
        """
        self._path = memory_path
        self._session_start = time.time()
        self._message_count = 0

        # Core memory stores
        self.user_profile: dict[str, Any] = {}
        self.facts: list[ConversationFact] = []
        self.emotional_history: list[EmotionalSnapshot] = []
        self.past_sessions: list[SessionSummary] = []
        self.topics_this_session: list[str] = []
        self.relationship_score: float = 50.0  # 0-100, starts neutral

        # Load existing memory
        self._load()

    def _load(self) -> None:
        """Load memory from disk if the file exists."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.user_profile = data.get("user_profile", {})
            self.facts = [
                ConversationFact(**f) for f in data.get("facts", [])
            ]
            self.past_sessions = [
                SessionSummary(**s) for s in data.get("past_sessions", [])
            ]
            self.relationship_score = data.get("relationship_score", 50.0)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Corrupted file — start fresh

    def save(self) -> None:
        """Persist memory to disk."""
        data = {
            "user_profile": self.user_profile,
            "facts": [asdict(f) for f in self.facts[-100:]],  # Keep last 100
            "past_sessions": [asdict(s) for s in self.past_sessions[-20:]],
            "relationship_score": self.relationship_score,
        }
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass  # Non-critical — memory is still in RAM

    def record_message(self, user_message: str, bot_response: str) -> None:
        """Record a message exchange and extract any facts.

        Args:
            user_message: The user's message.
            bot_response: Parhi's response.
        """
        self._message_count += 1
        self._extract_facts(user_message)
        self._extract_topics(user_message)

        # Immediate disk persistence so conversations & facts are never lost
        self.save()

    def record_emotion(self, emotion: str, intensity: float, message: str) -> None:
        """Record an emotional snapshot.

        Args:
            emotion: Detected emotion category.
            intensity: Emotion intensity (0.0-1.0).
            message: The message that triggered this emotion.
        """
        snapshot = EmotionalSnapshot(
            emotion=emotion,
            intensity=intensity,
            message_preview=message[:50],
        )
        self.emotional_history.append(snapshot)

        # Update relationship score based on emotions
        if emotion in ("happy", "loving"):
            self.relationship_score = min(100, self.relationship_score + intensity * 2)
        elif emotion in ("scolding", "angry"):
            self.relationship_score = max(0, self.relationship_score - intensity * 3)
        elif emotion == "sad":
            # Doesn't decrease score — she should be supportive
            pass

        # Keep last 200 snapshots
        if len(self.emotional_history) > 200:
            self.emotional_history = self.emotional_history[-200:]

    def _extract_facts(self, message: str) -> None:
        """Extract factual information from a user message.

        Args:
            message: The user's message to analyze.
        """
        lower = message.lower().strip()

        for pattern, category, key_template in FACT_PATTERNS:
            match = re.search(pattern, lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                value = groups[-1].strip() if groups else ""
                if not value:
                    continue

                # Build key from template
                key = key_template
                if "{0}" in key_template and len(groups) > 1:
                    key = key_template.format(groups[0])

                # Update user profile
                self.user_profile[key] = value

                # Record as a fact
                fact = ConversationFact(
                    category=category,
                    key=key,
                    value=value,
                    source_message=message[:100],
                )
                self.facts.append(fact)

    def _extract_topics(self, message: str) -> None:
        """Extract discussion topics from a message.

        Args:
            message: The user's message.
        """
        # Simple keyword-based topic extraction
        topic_keywords = {
            "neural networks": ["neural network", "nn", "deep learning"],
            "transformers": ["transformer", "attention", "self-attention"],
            "programming": ["code", "coding", "programming", "python", "javascript"],
            "philosophy": ["philosophy", "consciousness", "existence", "meaning"],
            "science": ["science", "physics", "chemistry", "biology"],
            "music": ["music", "song", "melody", "instrument"],
            "art": ["art", "painting", "drawing", "design"],
            "personal": ["my life", "my day", "how i feel", "i'm feeling"],
            "relationships": ["love", "relationship", "friend", "family"],
        }

        lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in lower for kw in keywords):
                if topic not in self.topics_this_session:
                    self.topics_this_session.append(topic)

    def get_user_name(self) -> str | None:
        """Get the user's name if known.

        Returns:
            The user's name or None.
        """
        return self.user_profile.get("preferred_name") or self.user_profile.get("name")

    def get_context_summary(self) -> str:
        """Generate a memory context string for injection into prompts.

        Returns:
            A formatted string summarizing known facts about the user.
        """
        parts: list[str] = []

        name = self.get_user_name()
        if name:
            parts.append(f"User's name: {name}")

        # Key profile facts
        for key, value in self.user_profile.items():
            if key not in ("name", "preferred_name"):
                parts.append(f"{key.replace('_', ' ').title()}: {value}")

        # Recent emotional trend
        if self.emotional_history:
            recent = self.emotional_history[-5:]
            dominant = max(set(s.emotion for s in recent),
                          key=lambda e: sum(1 for s in recent if s.emotion == e))
            if dominant != "neutral":
                parts.append(f"Recent mood trend: {dominant}")

        # Relationship health
        if self.relationship_score < 30:
            parts.append("Relationship status: needs repair (be extra kind and patient)")
        elif self.relationship_score > 80:
            parts.append("Relationship status: very close and warm")

        return "\n".join(parts) if parts else ""

    def end_session(self) -> None:
        """Finalize and save the current session summary."""
        duration = (time.time() - self._session_start) / 60.0

        # Determine overall mood
        if self.emotional_history:
            from collections import Counter
            mood_counts = Counter(s.emotion for s in self.emotional_history)
            overall_mood = mood_counts.most_common(1)[0][0]
        else:
            overall_mood = "neutral"

        summary = SessionSummary(
            date=time.strftime("%Y-%m-%d %H:%M"),
            duration_minutes=round(duration, 1),
            message_count=self._message_count,
            topics_discussed=self.topics_this_session[:10],
            overall_mood=overall_mood,
        )
        self.past_sessions.append(summary)
        self.save()

    def get_greeting_context(self) -> str:
        """Generate context for a personalized greeting.

        Returns:
            A string with info for crafting a personal greeting.
        """
        parts: list[str] = []

        name = self.get_user_name()
        if name:
            parts.append(f"The user's name is {name}.")

        if self.past_sessions:
            last = self.past_sessions[-1]
            parts.append(f"Last session: {last.date}, talked about {', '.join(last.topics_discussed[:3]) or 'various things'}.")
            parts.append(f"Last session mood: {last.overall_mood}.")

        if self.relationship_score > 70:
            parts.append("You have a warm, close relationship with this user.")
        elif self.relationship_score < 30:
            parts.append("The relationship has been strained recently. Be extra warm and patient.")

        return " ".join(parts)
