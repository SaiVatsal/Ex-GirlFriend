# self_correction.py
"""Self-correction, apology, and mistake-tracking system for Parhi-GPT.

Maintains a ledger of recent exchanges, detects when the user indicates
a mistake was made, generates appropriately escalated apologies, and
tracks past errors to prevent repetition within a session.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

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
    "nope", "no!", "NO", "that's incorrect",
]

# Apology templates by severity
APOLOGY_TEMPLATES: dict[str, list[str]] = {
    "mild": [
        "Oops, my bad! Let me correct that.",
        "You're right, I got that mixed up. Here's what I meant to say:",
        "Sorry about that! Let me think more carefully...",
        "Ah, you're absolutely right. My mistake!",
        "Fair enough — that wasn't my best answer. Let me try again:",
    ],
    "moderate": [
        "I'm really sorry about that. You're completely right, and I should have been more careful. Let me give you a proper answer:",
        "I apologize — that was a careless mistake on my part. Thank you for correcting me. Here's the right answer:",
        "You're 100% right, and I feel bad for getting that wrong. I should know better. Let me fix this:",
        "I messed up there, and I'm genuinely sorry. Your patience means a lot. Here's what I should have said:",
        "That was wrong of me, and I take full responsibility. Thank you for pointing it out — here's the correct answer:",
    ],
    "severe": [
        "I am so, so sorry. That was a terrible mistake and you have every right to be upset with me. I'm going to think really carefully this time and give you the answer you deserve. Please bear with me...",
        "I deeply apologize. I've been giving you poor answers and that's not acceptable. You deserve much better from me. Let me start fresh and really think this through...",
        "I'm truly sorry — I feel awful about getting this wrong, especially when you trusted me with it. I promise I'll do better. Let me take a moment and give you a thorough, accurate answer...",
        "You're completely right to be frustrated with me. That was inexcusable, and I'm genuinely ashamed. I owe you a proper answer, and I'm going to give it everything I've got this time...",
        "I can hear how frustrated you are, and every bit of that frustration is justified. I failed you, and I'm sorry. I'm going to slow down, think carefully, and make sure I get this right for you...",
    ],
}

# Self-awareness phrases when Parhi recognizes a pattern of mistakes
PATTERN_AWARENESS: list[str] = [
    "I notice I've been making quite a few mistakes today. I'm going to be extra careful from now on.",
    "I realize I keep getting things wrong, and that's not fair to you. I'm going to slow down and think more carefully.",
    "I can see a pattern here — I've been too hasty with my answers. Let me take more time to think things through.",
    "You've been really patient with me despite my mistakes. I appreciate that, and I'm going to step up my game.",
]


# ---------------------------------------------------------------------------
# Self-correction engine
# ---------------------------------------------------------------------------

class SelfCorrector:
    """Tracks mistakes, generates apologies, and prevents error repetition.

    Maintains a rolling ledger of recent exchanges and a persistent
    list of mistakes made during the current session. Uses these to
    generate contextually appropriate apologies and to detect when
    Parhi is repeating past errors.
    """

    def __init__(self, max_history: int = 20, max_mistakes: int = 50) -> None:
        """Initialize the self-corrector.

        Args:
            max_history: Maximum exchanges to keep in the rolling ledger.
            max_mistakes: Maximum mistake records to retain.
        """
        self._history: list[Exchange] = []
        self._mistakes: list[MistakeRecord] = []
        self._max_history = max_history
        self._max_mistakes = max_mistakes
        self._consecutive_mistakes = 0

    @property
    def mistake_count(self) -> int:
        """Total mistakes recorded this session."""
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
        """Record a user-bot exchange in the history ledger.

        Args:
            user_message: What the user said.
            bot_response: What Parhi replied.
            emotion: Detected emotion state for this exchange.
        """
        exchange = Exchange(
            user_message=user_message,
            bot_response=bot_response,
            emotion=emotion,
        )
        self._history.append(exchange)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def detect_mistake_indication(self, user_message: str) -> bool:
        """Check if the user's message indicates Parhi made a mistake.

        Args:
            user_message: The user's latest message.

        Returns:
            True if the message contains correction indicators.
        """
        lower = user_message.lower()
        return any(indicator in lower for indicator in CORRECTION_INDICATORS)

    def register_mistake(
        self,
        user_reaction: str,
        emotion: EmotionState | None = None,
    ) -> None:
        """Register that the last response was a mistake.

        Args:
            user_reaction: The user's correction/scolding message.
            emotion: The detected emotion state.
        """
        if not self._history:
            return

        last = self._history[-1]
        last.was_mistake = True

        # Try to extract what the user said was correct
        correction = ""
        lower = user_reaction.lower()
        for marker in ("actually it's", "actually, it's", "it should be",
                        "the answer is", "it's actually", "the correct answer is"):
            if marker in lower:
                idx = lower.index(marker) + len(marker)
                correction = user_reaction[idx:].strip().rstrip(".")
                break

        mistake = MistakeRecord(
            topic=last.user_message[:80],
            wrong_answer=last.bot_response[:200],
            user_reaction=user_reaction[:200],
            correction=correction,
        )
        self._mistakes.append(mistake)
        if len(self._mistakes) > self._max_mistakes:
            self._mistakes = self._mistakes[-self._max_mistakes:]

        self._consecutive_mistakes += 1

    def generate_apology(self, emotion: EmotionState | None = None) -> str:
        """Generate a contextually appropriate apology.

        The apology escalates in depth and sincerity based on:
        - The intensity of the user's emotional reaction
        - How many consecutive mistakes have been made
        - The total number of mistakes this session

        Args:
            emotion: Current emotion state of the user.

        Returns:
            An apology string to prepend to Parhi's next response.
        """
        # Determine severity
        intensity = emotion.intensity if emotion else 0.5

        if intensity > 0.7 or self._consecutive_mistakes >= 3:
            severity = "severe"
        elif intensity > 0.4 or self._consecutive_mistakes >= 2:
            severity = "moderate"
        else:
            severity = "mild"

        apology = random.choice(APOLOGY_TEMPLATES[severity])

        # Add pattern awareness if too many mistakes
        if self._consecutive_mistakes >= 3:
            apology += "\n\n" + random.choice(PATTERN_AWARENESS)

        # Reference the specific mistake if we have context
        if self._mistakes:
            last_mistake = self._mistakes[-1]
            if last_mistake.correction:
                apology += f"\n\nYou're right that it should be: {last_mistake.correction}"

        return apology

    def reset_streak(self) -> None:
        """Reset the consecutive mistake counter (called on a successful exchange)."""
        self._consecutive_mistakes = 0

    def has_made_similar_mistake(self, topic: str) -> bool:
        """Check if Parhi has made a mistake on a similar topic before.

        Args:
            topic: Brief description of the current topic.

        Returns:
            True if a similar mistake exists in the ledger.
        """
        topic_lower = topic.lower()
        return any(
            topic_lower in m.topic.lower() or m.topic.lower() in topic_lower
            for m in self._mistakes
        )

    def get_mistake_context(self) -> str:
        """Get a summary of recent mistakes for context injection.

        Returns:
            A string describing past mistakes, or empty string if none.
        """
        if not self._mistakes:
            return ""

        lines = ["[Past mistakes to avoid repeating:]"]
        for m in self._mistakes[-3:]:
            lines.append(f"- Topic: {m.topic}")
            if m.correction:
                lines.append(f"  Correct answer: {m.correction}")
        return "\n".join(lines)
