# thinking_engine.py
"""Extended thinking and chain-of-thought reasoning for Prarthana-GPT.

For complex questions, Prarthana can "think out loud" — showing her
reasoning process step by step before delivering a final answer.
Inspired by Claude's extended thinking feature.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThoughtProcess:
    """Structured representation of a thinking process.

    Attributes:
        question: The original question.
        thinking_steps: List of reasoning steps.
        conclusion: Final synthesized answer.
        confidence: How confident Prarthana is in the answer (0.0-1.0).
        thinking_duration: Simulated thinking time in seconds.
        show_thinking: Whether thinking steps were shown to the user.
    """
    question: str = ""
    thinking_steps: list[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.7
    thinking_duration: float = 0.0
    show_thinking: bool = True


# ---------------------------------------------------------------------------
# Thinking patterns
# ---------------------------------------------------------------------------

# Question types that benefit from extended thinking
THINKING_TRIGGERS: list[str] = [
    "explain", "how does", "why does", "what's the difference",
    "compare", "analyze", "should i", "what would happen if",
    "how to", "what's the best", "pros and cons",
    "is it better to", "what do you think about",
    "help me decide", "which one", "what are the steps",
    "break down", "walk me through", "teach me",
]

# Thinking step templates by question type
THINKING_TEMPLATES: dict[str, list[str]] = {
    "analytical": [
        "Okay, let me break this down...",
        "First, I need to consider {aspect1}...",
        "Then there's the matter of {aspect2}...",
        "And I shouldn't forget about {aspect3}...",
        "Putting it all together...",
    ],
    "comparative": [
        "Hmm, let me think about both sides...",
        "On one hand, {option1}...",
        "On the other hand, {option2}...",
        "When I weigh the factors...",
        "I think the key distinction is...",
    ],
    "problem_solving": [
        "Let me think through this step by step...",
        "The core problem here is...",
        "One approach would be...",
        "But we also need to consider...",
        "So the best path forward is...",
    ],
    "creative": [
        "Ooh, let me think creatively here...",
        "What if we approach it from this angle...",
        "That reminds me of...",
        "Building on that idea...",
        "Here's what I came up with...",
    ],
    "emotional": [
        "Let me take a moment to really think about this...",
        "I want to be thoughtful here because this matters...",
        "Considering how you might be feeling...",
        "What's really important here is...",
        "Here's what I genuinely think...",
    ],
}

# Filler phrases that make thinking feel more natural
THINKING_FILLERS: list[str] = [
    "Hmm...",
    "Let me think...",
    "That's interesting because...",
    "Actually, wait —",
    "Oh, I just realized...",
    "You know what, that connects to...",
    "So if I think about it from that perspective...",
    "Right, and another thing to consider is...",
    "Actually, I want to reconsider that...",
    "Okay, so putting all of this together...",
]

# Confidence qualifiers
CONFIDENCE_PHRASES: dict[str, list[str]] = {
    "high": [
        "I'm quite confident that",
        "I'm pretty sure",
        "Based on what I know,",
        "I can say with confidence that",
    ],
    "medium": [
        "I think",
        "If I'm not mistaken,",
        "From what I understand,",
        "It seems to me that",
    ],
    "low": [
        "I'm not entirely sure, but",
        "This is my best guess, but",
        "I could be wrong, but I think",
        "Take this with a grain of salt, but",
    ],
}


# ---------------------------------------------------------------------------
# Thinking engine
# ---------------------------------------------------------------------------

class ThinkingEngine:
    """Extended thinking system for complex reasoning.

    Generates structured thinking processes that can be displayed to
    the user, making Prarthana feel like she's genuinely reasoning
    through problems rather than just pattern-matching.
    """

    def __init__(self, visible_thinking: bool = True) -> None:
        """Initialize the thinking engine.

        Args:
            visible_thinking: Whether to show thinking steps to the user.
        """
        self.visible_thinking = visible_thinking
        self._thinking_history: list[ThoughtProcess] = []

    def needs_thinking(self, message: str) -> bool:
        """Determine if a message warrants extended thinking.

        Args:
            message: The user's message.

        Returns:
            True if the message is complex enough for chain-of-thought.
        """
        lower = message.lower()

        # Check for thinking triggers
        trigger_score = sum(1 for t in THINKING_TRIGGERS if t in lower)

        # Long questions often need more thought
        word_count = len(message.split())
        if word_count > 20:
            trigger_score += 1

        # Multiple question marks suggest complexity
        if message.count("?") > 1:
            trigger_score += 1

        return trigger_score >= 1

    def classify_question(self, message: str) -> str:
        """Classify the type of question for thinking template selection.

        Args:
            message: The user's message.

        Returns:
            Question type: "analytical", "comparative", "problem_solving",
            "creative", or "emotional".
        """
        lower = message.lower()

        if any(w in lower for w in ["compare", "difference", "vs", "better", "pros and cons"]):
            return "comparative"
        elif any(w in lower for w in ["how to", "solve", "fix", "help me", "steps"]):
            return "problem_solving"
        elif any(w in lower for w in ["create", "imagine", "what if", "design", "invent"]):
            return "creative"
        elif any(w in lower for w in ["feel", "think about", "opinion", "believe", "sad", "happy"]):
            return "emotional"
        else:
            return "analytical"

    def generate_thinking(self, message: str) -> ThoughtProcess:
        """Generate a structured thinking process for a message.

        Creates natural-sounding reasoning steps that demonstrate
        genuine cognitive engagement with the question.

        Args:
            message: The user's message.

        Returns:
            A ThoughtProcess with reasoning steps.
        """
        question_type = self.classify_question(message)
        templates = THINKING_TEMPLATES.get(question_type, THINKING_TEMPLATES["analytical"])

        # Generate thinking steps
        steps: list[str] = []

        # Opening thought
        steps.append(templates[0])

        # Middle reasoning steps (2-3 steps)
        num_middle = random.randint(2, min(3, len(templates) - 2))
        for i in range(1, num_middle + 1):
            if i < len(templates):
                step = templates[i]
                # Replace placeholders with contextual content
                step = self._fill_placeholders(step, message)
                steps.append(step)

            # Occasionally add a natural filler
            if random.random() < 0.3:
                steps.append(random.choice(THINKING_FILLERS))

        # Closing thought
        if len(templates) > 1:
            steps.append(templates[-1])

        # Determine confidence
        confidence = self._estimate_confidence(message)

        process = ThoughtProcess(
            question=message,
            thinking_steps=steps,
            confidence=confidence,
            show_thinking=self.visible_thinking,
        )

        self._thinking_history.append(process)
        return process

    def format_thinking_display(self, process: ThoughtProcess) -> str:
        """Format thinking steps for terminal display.

        Args:
            process: The thought process to display.

        Returns:
            Formatted string with thinking steps.
        """
        if not process.show_thinking or not process.thinking_steps:
            return ""

        lines: list[str] = []
        lines.append("  💭 Thinking...")
        lines.append("  ┌─────────────────────────────────")

        for step in process.thinking_steps:
            # Wrap long steps
            wrapped = self._wrap_text(step, width=50)
            for line in wrapped:
                lines.append(f"  │ {line}")

        lines.append("  └─────────────────────────────────")
        lines.append("")

        return "\n".join(lines)

    def get_confidence_prefix(self, confidence: float) -> str:
        """Get a confidence qualifier to prepend to the response.

        Args:
            confidence: Confidence level (0.0-1.0).

        Returns:
            A natural confidence qualifier phrase.
        """
        if confidence > 0.8:
            return random.choice(CONFIDENCE_PHRASES["high"])
        elif confidence > 0.5:
            return random.choice(CONFIDENCE_PHRASES["medium"])
        else:
            return random.choice(CONFIDENCE_PHRASES["low"])

    def _fill_placeholders(self, template: str, message: str) -> str:
        """Replace placeholder tokens with context from the message.

        Args:
            template: Template string with {placeholders}.
            message: The user's message for context extraction.

        Returns:
            Template with placeholders filled.
        """
        # Extract key nouns/concepts from the message
        words = message.split()
        # Filter to meaningful words (>3 chars, not common words)
        common = {"the", "and", "but", "how", "what", "why", "does",
                  "can", "you", "tell", "about", "that", "this", "with"}
        concepts = [w.strip("?.,!") for w in words
                    if len(w) > 3 and w.lower() not in common]

        # Fill placeholders with concepts
        replacements = {
            "{aspect1}": concepts[0] if len(concepts) > 0 else "the basics",
            "{aspect2}": concepts[1] if len(concepts) > 1 else "the details",
            "{aspect3}": concepts[2] if len(concepts) > 2 else "the implications",
            "{option1}": concepts[0] if len(concepts) > 0 else "the first option",
            "{option2}": concepts[1] if len(concepts) > 1 else "the second option",
        }

        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)

        return template

    def _estimate_confidence(self, message: str) -> float:
        """Estimate how confident Prarthana should be about this topic.

        Args:
            message: The user's message.

        Returns:
            Confidence score (0.0-1.0).
        """
        lower = message.lower()

        # Technical/factual questions → lower confidence (needs care)
        if any(w in lower for w in ["calculate", "exact", "precisely", "specifically"]):
            return 0.5

        # Opinion questions → moderate confidence
        if any(w in lower for w in ["think", "opinion", "feel", "believe"]):
            return 0.7

        # General knowledge → higher confidence
        if any(w in lower for w in ["explain", "what is", "how does"]):
            return 0.8

        return 0.7

    @staticmethod
    def _wrap_text(text: str, width: int = 50) -> list[str]:
        """Wrap text to specified width.

        Args:
            text: Text to wrap.
            width: Maximum line width.

        Returns:
            List of wrapped lines.
        """
        words = text.split()
        lines: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > width and current:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1

        if current:
            lines.append(" ".join(current))

        return lines or [""]
