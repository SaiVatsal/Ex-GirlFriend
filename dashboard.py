# dashboard.py
"""Conversation analytics dashboard for Parhi-GPT.

Provides a rich terminal-based dashboard showing real-time conversation
statistics, emotional trends, memory contents, and relationship health.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from emotion_engine import EmotionState
from memory import MemoryManager


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationStats:
    """Aggregated conversation statistics.

    Attributes:
        total_messages: Total user messages.
        total_responses: Total Parhi responses.
        session_duration_min: Session length in minutes.
        avg_response_length: Average response word count.
        topics_discussed: List of topics covered.
        emotion_distribution: Count of each emotion detected.
        mistakes_made: Number of mistakes/corrections.
        relationship_score: Current relationship health (0-100).
    """
    total_messages: int = 0
    total_responses: int = 0
    session_duration_min: float = 0.0
    avg_response_length: float = 0.0
    topics_discussed: list[str] = None  # type: ignore
    emotion_distribution: dict[str, int] = None  # type: ignore
    mistakes_made: int = 0
    relationship_score: float = 50.0

    def __post_init__(self):
        if self.topics_discussed is None:
            self.topics_discussed = []
        if self.emotion_distribution is None:
            self.emotion_distribution = {}


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

# Emotion emoji mapping
EMOTION_EMOJI: dict[str, str] = {
    "happy": "😊",
    "loving": "💕",
    "curious": "🤔",
    "neutral": "😐",
    "sad": "😢",
    "frustrated": "😤",
    "angry": "😠",
    "scolding": "😡",
}

# Relationship health labels
RELATIONSHIP_LABELS: list[tuple[float, str, str]] = [
    (90, "💖 Soulmates", "Your bond is incredibly strong!"),
    (75, "❤️ Very Close", "You two have a wonderful connection."),
    (60, "💛 Good Friends", "A solid and warm relationship."),
    (45, "🤝 Friendly", "Getting along well."),
    (30, "😕 Strained", "Things have been a bit rough lately."),
    (15, "💔 Troubled", "The relationship needs some care."),
    (0, "🥀 Critical", "Please be patient — healing takes time."),
]


class Dashboard:
    """Terminal-based conversation analytics dashboard.

    Renders a rich, formatted display of conversation metrics
    directly in the terminal.
    """

    def __init__(self, memory: MemoryManager | None = None) -> None:
        """Initialize the dashboard.

        Args:
            memory: Memory manager for accessing persistent data.
        """
        self._memory = memory
        self._session_start = time.time()
        self._message_count = 0
        self._response_lengths: list[int] = []
        self._emotions: list[EmotionState] = []
        self._emotion_counts: dict[str, int] = {}
        self._mistakes = 0

    def record_exchange(
        self,
        user_msg: str,
        response: str,
        emotion: EmotionState | None = None,
        was_mistake: bool = False,
    ) -> None:
        """Record an exchange for analytics.

        Args:
            user_msg: The user's message.
            response: Parhi's response.
            emotion: Detected emotion state.
            was_mistake: Whether this was a mistake/correction.
        """
        self._message_count += 1
        self._response_lengths.append(len(response.split()))

        if emotion:
            self._emotions.append(emotion)
            self._emotion_counts[emotion.emotion] = (
                self._emotion_counts.get(emotion.emotion, 0) + 1
            )

        if was_mistake:
            self._mistakes += 1

    def get_stats(self) -> ConversationStats:
        """Calculate current conversation statistics.

        Returns:
            ConversationStats with all current metrics.
        """
        duration = (time.time() - self._session_start) / 60.0
        avg_len = (
            sum(self._response_lengths) / len(self._response_lengths)
            if self._response_lengths else 0.0
        )

        topics = []
        if self._memory:
            topics = self._memory.topics_this_session
            rel_score = self._memory.relationship_score
        else:
            rel_score = 50.0

        return ConversationStats(
            total_messages=self._message_count,
            total_responses=self._message_count,
            session_duration_min=round(duration, 1),
            avg_response_length=round(avg_len, 1),
            topics_discussed=topics,
            emotion_distribution=dict(self._emotion_counts),
            mistakes_made=self._mistakes,
            relationship_score=rel_score,
        )

    def render(self) -> str:
        """Render the full dashboard as a formatted string.

        Returns:
            Multi-line string of the formatted dashboard.
        """
        stats = self.get_stats()
        width = 56

        lines: list[str] = []
        lines.append("")
        lines.append("╔" + "═" * width + "╗")
        lines.append("║" + "  📊 Parhi-GPT Dashboard  ".center(width) + "║")
        lines.append("╠" + "═" * width + "╣")

        # Session info
        lines.append("║" + "  📝 Session Stats".ljust(width) + "║")
        lines.append("║" + f"    Messages: {stats.total_messages}".ljust(width) + "║")
        lines.append("║" + f"    Duration: {stats.session_duration_min} min".ljust(width) + "║")
        lines.append("║" + f"    Avg Response: {stats.avg_response_length} words".ljust(width) + "║")
        lines.append("║" + f"    Corrections: {stats.mistakes_made}".ljust(width) + "║")
        lines.append("╟" + "─" * width + "╢")

        # Emotion distribution
        lines.append("║" + "  🎭 Emotion History".ljust(width) + "║")
        if stats.emotion_distribution:
            # Sort by count descending
            sorted_emotions = sorted(
                stats.emotion_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            total = sum(v for _, v in sorted_emotions)
            for emotion, count in sorted_emotions:
                emoji = EMOTION_EMOJI.get(emotion, "•")
                pct = (count / total * 100) if total > 0 else 0
                bar_len = int(pct / 5)  # 20-char max bar
                bar = "█" * bar_len + "░" * (20 - bar_len)
                line = f"    {emoji} {emotion:<12} {bar} {pct:.0f}%"
                lines.append("║" + line.ljust(width) + "║")
        else:
            lines.append("║" + "    (no emotions detected yet)".ljust(width) + "║")
        lines.append("╟" + "─" * width + "╢")

        # Emotion trend (last 10)
        lines.append("║" + "  📈 Recent Mood".ljust(width) + "║")
        if self._emotions:
            recent = self._emotions[-10:]
            trend = " → ".join(
                EMOTION_EMOJI.get(e.emotion, "•") for e in recent
            )
            # Wrap if too long
            if len(trend) > width - 6:
                trend = trend[:width - 9] + "..."
            lines.append("║" + f"    {trend}".ljust(width) + "║")
        else:
            lines.append("║" + "    (no data yet)".ljust(width) + "║")
        lines.append("╟" + "─" * width + "╢")

        # Topics
        lines.append("║" + "  💬 Topics Discussed".ljust(width) + "║")
        if stats.topics_discussed:
            topics_str = ", ".join(stats.topics_discussed[:6])
            if len(topics_str) > width - 6:
                topics_str = topics_str[:width - 9] + "..."
            lines.append("║" + f"    {topics_str}".ljust(width) + "║")
        else:
            lines.append("║" + "    (none yet)".ljust(width) + "║")
        lines.append("╟" + "─" * width + "╢")

        # Relationship health
        lines.append("║" + "  💗 Relationship Health".ljust(width) + "║")
        score = stats.relationship_score
        label, desc = self._get_relationship_label(score)
        bar_len = int(score / 5)  # 20-char max bar
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append("║" + f"    {label}".ljust(width) + "║")
        lines.append("║" + f"    {bar} {score:.0f}/100".ljust(width) + "║")
        lines.append("║" + f"    {desc}".ljust(width) + "║")

        lines.append("╚" + "═" * width + "╝")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _get_relationship_label(score: float) -> tuple[str, str]:
        """Get relationship label and description for a score.

        Args:
            score: Relationship score (0-100).

        Returns:
            (label, description) tuple.
        """
        for threshold, label, desc in RELATIONSHIP_LABELS:
            if score >= threshold:
                return label, desc
        return "🥀 Critical", "Please be patient."

    def print_dashboard(self) -> None:
        """Print the dashboard to the terminal."""
        print(self.render())

    def print_mini_status(self) -> None:
        """Print a compact one-line status bar."""
        stats = self.get_stats()

        # Dominant emotion
        if stats.emotion_distribution:
            top_emotion = max(stats.emotion_distribution, key=stats.emotion_distribution.get)
            emoji = EMOTION_EMOJI.get(top_emotion, "•")
        else:
            emoji = "😐"

        # Relationship indicator
        score = stats.relationship_score
        if score > 75:
            rel = "❤️"
        elif score > 50:
            rel = "💛"
        elif score > 25:
            rel = "😕"
        else:
            rel = "💔"

        status = (
            f"  [{emoji} mood] [{rel} bond:{score:.0f}] "
            f"[💬 {stats.total_messages} msgs] "
            f"[⏱ {stats.session_duration_min:.0f}m]"
        )
        print(status)
