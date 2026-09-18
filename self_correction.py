# self_correction.py
"""Self-correction, apology, and mistake-tracking system for Parhi-GPT.

Maintains a ledger of recent exchanges, detects when the user indicates
a mistake was made, generates appropriately escalated apologies, and
persists past errors to disk so Parhi remembers corrections and learns
across sessions without repeating mistakes.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict

from emotion_engine import EmotionState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Exchange:
    """A single user-Parhi exchange with metadata.

    Attributes:
        user_message: What the user said.
        bot_response: What Parhi replied.
        timestamp: Unix timestamp of the exchange.
        emotion: Detected emotion state at this exchange.
        was_mistake: Whether the user indicated this was wrong.
        correction: What the correct answer was (if provided by user).
    """
    user_message: str
    bot_response: str
    timestamp: float = field(default_factory=time.time)
    emotion: EmotionState | None = None
    was_mistake: bool = False
    correction: str = ""


@dataclass
class MistakeRecord:
    """Record of a specific mistake Parhi made.

    Attributes:
        topic: Brief description of what the question was about.
        wrong_answer: The incorrect response given.
        user_reaction: How the user reacted (the scolding/correction message).
        correction: The correct answer if the user provided one.
        timestamp: When the mistake occurred.
        apology_given: The apology that was generated.
    """
    topic: str
    wrong_answer: str
    user_reaction: str
    correction: str = ""
    timestamp: float = field(default_factory=time.time)
    apology_given: str = ""


# ---------------------------------------------------------------------------
# Correction patterns that indicate the user is pointing out a mistake
# ---------------------------------------------------------------------------

CORRECTION_INDICATORS: list[str] = [
    "that's wrong", "that's not right", "incorrect", "no that's",
    "you're wrong", "actually it's", "actually, it's", "no, it's",
    "that's not what", "not what i asked", "i said", "i meant",
    "try again", "do it again", "redo", "fix it", "fix that",
    "the answer is", "it should be", "the correct", "wrong answer",
    "not even close", "way off", "completely wrong", "dead wrong",
    "nope", "no!", "that's incorrect", "that is wrong", "not right",
    "it is actually", "actually it is",
]

# Apology templates by severity (mature, sincere, without weird filler)
APOLOGY_TEMPLATES: dict[str, list[str]] = {
    "mild": [
        "My mistake. Thank you for correcting me. Here is the accurate answer:",
        "You're right, I got that wrong. Let me provide the correct answer:",
        "Thank you for catching that. Let me fix that for you:",
        "You're absolutely right. My apologies — here's the correct information:",
    ],
    "moderate": [
        "I apologize for the error. You're completely right, and I've noted that correction for the future. Here is the right answer:",
        "Thank you for setting me straight. That was an oversight on my part. Here is what's correct:",
        "I appreciate you correcting me. I've stored that in my memory so I don't get it wrong again. Here's the accurate answer:",
    ],
    "severe": [
        "I deeply apologize. I gave you incorrect information and you have every right to be frustrated. I have recorded this correction so I won't repeat it:",
        "I take full responsibility for that mistake. Thank you for your patience while I correct this:",
    ],
}

# Self-awareness phrases when Parhi recognizes a pattern of mistakes
PATTERN_AWARENESS: list[str] = [
    "I'm going to be extra careful with my answers going forward.",
    "I appreciate you bearing with me as I learn and refine my understanding.",
    "Thank you for teaching me — your feedback helps me get better.",
]


# ---------------------------------------------------------------------------
# Self-correction engine
# ---------------------------------------------------------------------------

class SelfCorrector:
    """Tracks mistakes, generates apologies, and prevents error repetition.

    Persists mistake records to disk in ``parhi_mistakes.json`` so corrections
    survive restarts and allow Parhi to genuinely learn from her mistakes.
    """

    def __init__(
        self,
        max_history: int = 20,
        max_mistakes: int = 100,
        persistence_path: str = "parhi_mistakes.json",
    ) -> None:
        self._history: list[Exchange] = []
        self._mistakes: list[MistakeRecord] = []
        self._max_history = max_history
        self._max_mistakes = max_mistakes
        self._consecutive_mistakes = 0
        self._persistence_path = persistence_path
        self._load()

    def _load(self) -> None:
        """Load past mistake records from disk."""
        if not os.path.exists(self._persistence_path):
            return
        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._mistakes = [
                MistakeRecord(**m) for m in data.get("mistakes", [])
            ]
        except Exception as e:
            print(f"[self_correction] Note loading past mistakes: {e}")

    def _save(self) -> None:
        """Save mistake records to disk."""
        try:
            data = {
                "mistakes": [asdict(m) for m in self._mistakes[-self._max_mistakes:]],
                "total_recorded": len(self._mistakes),
            }
            with open(self._persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[self_correction] Note saving mistakes: {e}")

    @property
    def mistake_count(self) -> int:
        """Total mistakes recorded."""
        return len(self._mistakes)

    @property
    def recent_mistakes(self) -> list[MistakeRecord]:
        """Last 5 mistakes for context."""
        return self._mistakes[-5:]

    def record_exchange(
        self,
        user_message: str,
        bot_response: str,
        emotion: EmotionState | None = None,
    ) -> None:
        """Record a user-bot exchange in the history ledger."""
        exchange = Exchange(
            user_message=user_message,
            bot_response=bot_response,
            emotion=emotion,
        )
        self._history.append(exchange)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def detect_mistake_indication(self, user_message: str) -> bool:
        """Check if the user's message indicates Parhi made a mistake."""
        lower = user_message.lower()
        return any(indicator in lower for indicator in CORRECTION_INDICATORS)

    def register_mistake(
        self,
        user_reaction: str,
        emotion: EmotionState | None = None,
    ) -> None:
        """Register that the last response was a mistake and persist it."""
        if not self._history:
            return

        last = self._history[-1]
        last.was_mistake = True

        # Extract what the user said was correct
        correction = ""
        lower = user_reaction.lower()
        markers = (
            "the correct answer is", "the right answer is", "the answer is",
            "it's actually", "it is actually", "actually it's", "actually, it's",
            "actually it is", "actually,", "actually ", "it should be",
            "no, it's", "no it's", "no, it is", "no it is",
            "i meant", "i said", "not that, it's", "real answer is",
        )
        for marker in markers:
            if marker in lower:
                idx = lower.index(marker) + len(marker)
                cand = user_reaction[idx:].strip().rstrip(".! ")
                if cand:
                    correction = cand
                    break

        mistake = MistakeRecord(
            topic=last.user_message[:100],
            wrong_answer=last.bot_response[:250],
            user_reaction=user_reaction[:250],
            correction=correction,
        )
        self._mistakes.append(mistake)
        if len(self._mistakes) > self._max_mistakes:
            self._mistakes = self._mistakes[-self._max_mistakes:]

        self._consecutive_mistakes += 1
        self._save()

    def generate_apology(self, emotion: EmotionState | None = None) -> str:
        """Generate a contextually appropriate apology."""
        intensity = emotion.intensity if emotion else 0.5

        if intensity > 0.7 or self._consecutive_mistakes >= 3:
            severity = "severe"
        elif intensity > 0.4 or self._consecutive_mistakes >= 2:
            severity = "moderate"
        else:
            severity = "mild"

        apology = random.choice(APOLOGY_TEMPLATES[severity])

        if self._consecutive_mistakes >= 3:
            apology += " " + random.choice(PATTERN_AWARENESS)

        if self._mistakes:
            last_mistake = self._mistakes[-1]
            if last_mistake.correction:
                apology += f" You're right that it is: {last_mistake.correction}."

        return apology

    def reset_streak(self) -> None:
        """Reset consecutive mistake counter upon successful exchange."""
        self._consecutive_mistakes = 0

    def find_relevant_correction(self, query: str) -> str | None:
        """Look up past mistakes to see if the user previously taught Parhi the correct answer.

        Returns:
            The learned correction string if a match is found, or None.
        """
        if not self._mistakes:
            return None

        q_lower = query.lower()
        q_words = set(re.findall(r"\w+", q_lower))

        for m in reversed(self._mistakes):
            if not m.correction:
                continue
            topic_words = set(re.findall(r"\w+", m.topic.lower()))
            # If significant keyword overlap exists between the current query and a past mistake
            overlap = q_words.intersection(topic_words)
            meaningful_overlap = [w for w in overlap if len(w) > 3 and w not in ("what", "when", "where", "which", "your", "this", "that")]
            if len(meaningful_overlap) >= 2 or (len(meaningful_overlap) == 1 and len(topic_words) <= 3):
                return m.correction
        return None

    def get_mistake_context(self) -> str:
        """Get a summary of past mistakes and learned corrections for prompt context."""
        if not self._mistakes:
            return ""

        corrected_records = [m for m in self._mistakes if m.correction]
        if not corrected_records:
            return ""

        lines = ["[Learned Corrections from User:]"]
        for m in corrected_records[-5:]:
            lines.append(f"- Query: \"{m.topic}\" → Correct fact: \"{m.correction}\"")
        return "\n".join(lines)
