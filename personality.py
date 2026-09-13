# personality.py
"""Dynamic personality and mood system for Parhi-GPT.

Parhi has her own internal mood that evolves based on conversation
flow. This module modulates response generation parameters (temperature,
length, tone) to create a living, breathing personality that feels
genuinely human.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from emotion_engine import EmotionState


# ---------------------------------------------------------------------------
# Personality traits and mood definitions
# ---------------------------------------------------------------------------

@dataclass
class MoodState:
    """Parhi's internal mood at a given moment.

    Attributes:
        mood: Current mood label.
        energy: Energy level (0.0 = subdued, 1.0 = very energetic).
        warmth: How warm/affectionate she's being (0.0-1.0).
        confidence: How confident she is in her answers (0.0-1.0).
        playfulness: How playful/casual vs serious (0.0-1.0).
        updated_at: Last time the mood was updated.
    """
    mood: str = "cheerful"
    energy: float = 0.7
    warmth: float = 0.8
    confidence: float = 0.7
    playfulness: float = 0.5
    updated_at: float = field(default_factory=time.time)


# Mood transition rules: how user emotions affect Parhi's mood
MOOD_TRANSITIONS: dict[str, dict[str, float]] = {
    # user_emotion: {mood_dimension: delta}
    "happy": {"energy": 0.1, "warmth": 0.1, "playfulness": 0.15, "confidence": 0.05},
    "loving": {"energy": 0.05, "warmth": 0.2, "playfulness": 0.1, "confidence": 0.1},
    "curious": {"energy": 0.1, "warmth": 0.0, "playfulness": -0.05, "confidence": 0.1},
    "scolding": {"energy": -0.15, "warmth": 0.1, "playfulness": -0.3, "confidence": -0.2},
    "angry": {"energy": -0.1, "warmth": 0.15, "playfulness": -0.25, "confidence": -0.15},
    "frustrated": {"energy": -0.05, "warmth": 0.1, "playfulness": -0.2, "confidence": -0.1},
    "sad": {"energy": -0.1, "warmth": 0.2, "playfulness": -0.2, "confidence": 0.0},
    "neutral": {"energy": 0.0, "warmth": 0.0, "playfulness": 0.0, "confidence": 0.02},
}

# Mood labels derived from mood dimensions
MOOD_LABELS: list[tuple[str, dict[str, tuple[float, float]]]] = [
    ("excited", {"energy": (0.8, 1.0), "playfulness": (0.6, 1.0)}),
    ("playful", {"playfulness": (0.7, 1.0), "energy": (0.5, 1.0)}),
    ("cheerful", {"energy": (0.5, 0.8), "warmth": (0.6, 1.0)}),
    ("thoughtful", {"confidence": (0.6, 1.0), "playfulness": (0.0, 0.3)}),
    ("gentle", {"warmth": (0.7, 1.0), "energy": (0.2, 0.5)}),
    ("concerned", {"warmth": (0.7, 1.0), "confidence": (0.3, 0.6)}),
    ("apologetic", {"confidence": (0.0, 0.3), "warmth": (0.5, 1.0)}),
    ("subdued", {"energy": (0.0, 0.3), "confidence": (0.0, 0.4)}),
    ("calm", {"energy": (0.3, 0.6), "warmth": (0.4, 0.7)}),
]

# Tone markers by mood
TONE_MARKERS: dict[str, list[str]] = {
    "excited": ["!", " 🎉", " ✨", " 🚀", "!!"],
    "playful": [" 😄", " haha", " 😊", " ~"],
    "cheerful": [" 😊", "!", " 💫"],
    "thoughtful": ["...", " 🤔", " — "],
    "gentle": [" 💕", " ❤️", " 🌸"],
    "concerned": [" 💙", " 🤗"],
    "apologetic": [" 😔", " 🙏"],
    "subdued": [".", "..."],
    "calm": [".", " ☺️"],
}

# Conversational fillers by mood (make responses feel more natural)
CONVERSATIONAL_FILLERS: dict[str, list[str]] = {
    "excited": [
        "Oh my gosh, ", "Okay so, ", "Oh! ", "YES! ", "Ooh, ",
    ],
    "playful": [
        "Hehe, ", "Soo, ", "Well well well, ", "Hmm, ",
    ],
    "cheerful": [
        "Oh, ", "Hey! ", "So, ", "Alright! ",
    ],
    "thoughtful": [
        "Hmm, that's interesting... ", "Let me think about this... ",
        "You know, ", "That's a great point... ",
    ],
    "gentle": [
        "Hey, ", "You know, ", "Listen, ", "Sweetie, ",
    ],
    "concerned": [
        "Hey, I notice that... ", "I want you to know... ",
        "Listen, ", "I hear you... ",
    ],
    "apologetic": [
        "Look, I... ", "I have to be honest, ", "I want to say... ",
    ],
    "subdued": [
        "Yeah... ", "I understand... ", "Okay... ",
    ],
    "calm": [
        "Well, ", "So, ", "Alright, ",
    ],
}


# ---------------------------------------------------------------------------
# Personality engine
# ---------------------------------------------------------------------------

class PersonalityEngine:
    """Dynamic personality system that modulates Parhi's behavior.

    Maintains an internal mood state that evolves based on user emotions,
    conversation duration, and contextual factors. Adjusts generation
    parameters (temperature, top_k, max_tokens) and adds tone markers
    to make responses feel naturally human.
    """

    def __init__(self) -> None:
        self._mood = MoodState()
        self._exchanges_count = 0
        self._session_start = time.time()

    @property
    def mood(self) -> MoodState:
        """Current mood state."""
        return self._mood

    @property
    def mood_label(self) -> str:
        """Current mood as a human-readable label."""
        return self._mood.mood

    def update_mood(self, user_emotion: EmotionState) -> None:
        """Update Parhi's mood based on the user's detected emotion.

        Applies mood transition deltas and natural mood drift over time.

        Args:
            user_emotion: The detected emotion state of the user.
        """
        self._exchanges_count += 1

        # Apply emotion-based transitions
        deltas = MOOD_TRANSITIONS.get(user_emotion.emotion, {})
        scale = user_emotion.intensity  # stronger emotion → bigger mood shift

        self._mood.energy = self._clamp(
            self._mood.energy + deltas.get("energy", 0) * scale
        )
        self._mood.warmth = self._clamp(
            self._mood.warmth + deltas.get("warmth", 0) * scale
        )
        self._mood.confidence = self._clamp(
            self._mood.confidence + deltas.get("confidence", 0) * scale
        )
        self._mood.playfulness = self._clamp(
            self._mood.playfulness + deltas.get("playfulness", 0) * scale
        )

        # Natural drift: mood slowly returns to baseline over time
        drift = 0.02
        self._mood.energy += (0.7 - self._mood.energy) * drift
        self._mood.warmth += (0.8 - self._mood.warmth) * drift
        self._mood.confidence += (0.7 - self._mood.confidence) * drift
        self._mood.playfulness += (0.5 - self._mood.playfulness) * drift

        # Determine mood label
        self._mood.mood = self._classify_mood()
        self._mood.updated_at = time.time()

    def _classify_mood(self) -> str:
        """Classify current dimensions into a mood label.

        Returns:
            The best-matching mood label string.
        """
        best_label = "calm"
        best_score = -1.0

        for label, criteria in MOOD_LABELS:
            score = 0.0
            matches = 0
            for dim, (low, high) in criteria.items():
                val = getattr(self._mood, dim, 0.5)
                if low <= val <= high:
                    matches += 1
                    # How centered in the range
                    center = (low + high) / 2
                    score += 1.0 - abs(val - center) / max(high - low, 0.01)
            if matches == len(criteria) and score > best_score:
                best_score = score
                best_label = label

        return best_label

    def get_generation_params(self) -> dict[str, float | int]:
        """Get model generation parameters modulated by current mood.

        Returns:
            Dict with 'temperature', 'top_k', and 'max_tokens' keys.
        """
        # Base parameters
        temp = 0.8
        top_k = 30
        max_tokens = 500

        # Mood modulation
        mood = self._mood

        # Higher playfulness → higher temperature (more creative)
        temp += (mood.playfulness - 0.5) * 0.2

        # Lower confidence → lower temperature (more careful/focused)
        temp -= (0.7 - mood.confidence) * 0.15

        # High energy → allow more tokens
        max_tokens = int(max_tokens * (0.8 + mood.energy * 0.4))

        # Low confidence → more top_k options considered (hedging)
        if mood.confidence < 0.4:
            top_k = 50

        # Clamp
        temp = max(0.5, min(1.0, temp))
        top_k = max(10, min(100, top_k))
        max_tokens = max(100, min(800, max_tokens))

        return {
            "temperature": round(temp, 2),
            "top_k": top_k,
            "max_tokens": max_tokens,
        }

    def add_personality_touches(self, response: str) -> str:
        """Add subtle personality markers to a response.

        Occasionally adds tone markers, fillers, or adjusts phrasing
        based on current mood. Applied sparingly to feel natural.

        Args:
            response: The raw response text.

        Returns:
            The response with personality touches applied.
        """
        mood_label = self._mood.mood

        # 30% chance of adding a conversational filler at the start
        if random.random() < 0.3:
            fillers = CONVERSATIONAL_FILLERS.get(mood_label, [])
            if fillers and not any(response.startswith(f) for f in fillers):
                filler = random.choice(fillers)
                # Don't double-capitalize
                if response and response[0].isupper():
                    response = filler + response[0].lower() + response[1:]
                else:
                    response = filler + response

        # 20% chance of adding a tone marker at the end
        if random.random() < 0.2:
            markers = TONE_MARKERS.get(mood_label, [])
            if markers:
                marker = random.choice(markers)
                # Don't add emoji if response already ends with one
                if not (len(response) > 0 and ord(response[-1]) > 127):
                    response = response.rstrip(".!") + marker

        return response

    def get_mood_description(self) -> str:
        """Get a human-readable description of Parhi's current mood.

        Returns:
            A description string like "feeling cheerful and energetic".
        """
        m = self._mood
        parts = [f"feeling {m.mood}"]

        if m.energy > 0.8:
            parts.append("very energetic")
        elif m.energy < 0.3:
            parts.append("a bit tired")

        if m.warmth > 0.8:
            parts.append("very warm and caring")
        elif m.warmth < 0.3:
            parts.append("somewhat reserved")

        if m.confidence > 0.8:
            parts.append("confident")
        elif m.confidence < 0.3:
            parts.append("a little unsure")

        return ", ".join(parts)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        """Clamp a value to [low, high]."""
        return max(low, min(high, value))
