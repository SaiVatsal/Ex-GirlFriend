# streaming.py
"""Natural typing animation for Prarthana-GPT.

Instead of dumping full responses instantly, this module simulates
human-like typing with variable speed, pauses at punctuation, and
optional self-correction animations.
"""
from __future__ import annotations

import random
import sys
import time


# ---------------------------------------------------------------------------
# Timing constants (in seconds)
# ---------------------------------------------------------------------------

# Base character delay ranges by category
CHAR_DELAYS: dict[str, tuple[float, float]] = {
    "normal": (0.015, 0.045),      # Regular characters
    "space": (0.02, 0.06),         # Spaces (slight pause between words)
    "comma": (0.08, 0.15),         # Commas — brief thinking pause
    "period": (0.12, 0.25),        # Periods — sentence boundary pause
    "exclamation": (0.05, 0.1),    # Exclamation — quicker (excitement)
    "question": (0.1, 0.2),        # Question marks — slight reflection
    "ellipsis_char": (0.15, 0.3),  # Each dot in "..." — dramatic pause
    "newline": (0.2, 0.4),         # Newlines — paragraph pause
    "emoji": (0.05, 0.1),          # Emoji — quick (spontaneous)
    "dash": (0.06, 0.12),          # Dashes — mid-thought pause
}

# Speed multipliers by mood
MOOD_SPEED: dict[str, float] = {
    "excited": 0.6,      # Types fast when excited
    "playful": 0.7,      # Pretty quick
    "cheerful": 0.8,     # Normal-ish
    "thoughtful": 1.3,   # Slower, more deliberate
    "gentle": 1.1,       # Slightly slower, careful
    "concerned": 1.2,    # Measured pace
    "apologetic": 1.4,   # Slower, more careful
    "subdued": 1.5,      # Slowest
    "calm": 1.0,         # Normal baseline
}


# ---------------------------------------------------------------------------
# Typing indicator animation
# ---------------------------------------------------------------------------

TYPING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def show_typing_indicator(duration: float = 1.0, label: str = "Prarthana is thinking") -> None:
    """Display a brief animated typing indicator.

    Args:
        duration: How long to show the indicator (seconds).
        label: Text label next to the animation.
    """
    start = time.time()
    frame_idx = 0
    while time.time() - start < duration:
        frame = TYPING_FRAMES[frame_idx % len(TYPING_FRAMES)]
        sys.stdout.write(f"\r  {frame} {label}...")
        sys.stdout.flush()
        time.sleep(0.08)
        frame_idx += 1
    # Clear the indicator line
    sys.stdout.write("\r" + " " * (len(label) + 10) + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Streaming printer
# ---------------------------------------------------------------------------

class StreamingPrinter:
    """Prints text with human-like typing animation.

    Supports variable character-level delays, mood-based speed modulation,
    and optional self-correction effects.
    """

    def __init__(self, mood: str = "calm", speed_multiplier: float = 1.0) -> None:
        """Initialize the streaming printer.

        Args:
            mood: Current mood label (affects typing speed).
            speed_multiplier: Global speed multiplier (lower = faster).
        """
        self.mood = mood
        self.speed_multiplier = speed_multiplier
        self._mood_factor = MOOD_SPEED.get(mood, 1.0)

    def stream(
        self,
        text: str,
        prefix: str = "Prarthana: ",
        show_indicator: bool = True,
        thinking_duration: float = 0.8,
    ) -> None:
        """Stream text to stdout with typing animation.

        Args:
            text: The text to stream.
            prefix: Label printed before the text (not animated).
            show_indicator: Whether to show "thinking" animation first.
            thinking_duration: How long to show the thinking indicator.
        """
        if not text:
            return

        # Show thinking indicator
        if show_indicator:
            show_typing_indicator(
                duration=thinking_duration,
                label="Prarthana is thinking",
            )

        # Print prefix instantly
        sys.stdout.write(f"\n{prefix}")
        sys.stdout.flush()

        # Stream each character with variable delay
        in_ellipsis = False
        for i, char in enumerate(text):
            sys.stdout.write(char)
            sys.stdout.flush()

            delay = self._get_delay(char, i, text, in_ellipsis)
            in_ellipsis = char == "." and i + 1 < len(text) and text[i + 1] == "."

            if delay > 0:
                time.sleep(delay)

        sys.stdout.write("\n")
        sys.stdout.flush()

    def _get_delay(
        self,
        char: str,
        index: int,
        full_text: str,
        in_ellipsis: bool,
    ) -> float:
        """Calculate the delay for a specific character.

        Args:
            char: The character being printed.
            index: Position in the text.
            full_text: The full text being printed.
            in_ellipsis: Whether we're inside a "..." sequence.

        Returns:
            Delay in seconds.
        """
        # Determine character category
        if char == "\n":
            category = "newline"
        elif char == " ":
            category = "space"
        elif char == ",":
            category = "comma"
        elif char == ".":
            if in_ellipsis:
                category = "ellipsis_char"
            else:
                category = "period"
        elif char == "!":
            category = "exclamation"
        elif char == "?":
            category = "question"
        elif char in ("-", "—"):
            category = "dash"
        elif ord(char) > 127:
            category = "emoji"
        else:
            category = "normal"

        low, high = CHAR_DELAYS[category]
        base_delay = random.uniform(low, high)

        # Apply mood factor
        delay = base_delay * self._mood_factor * self.speed_multiplier

        # Add micro-variation for naturalness (±20%)
        delay *= random.uniform(0.8, 1.2)

        return max(0.005, delay)

    def stream_with_self_correction(
        self,
        text: str,
        correction_chance: float = 0.03,
        prefix: str = "Prarthana: ",
    ) -> None:
        """Stream text with occasional simulated self-corrections.

        Sometimes Prarthana "types" a wrong word, backspaces, and retypes
        the correct one — a deeply human touch.

        Args:
            text: The text to stream.
            correction_chance: Probability of a self-correction at each word.
            prefix: Label prefix.
        """
        if not text:
            return

        show_typing_indicator(0.8, "Prarthana is thinking")
        sys.stdout.write(f"\n{prefix}")
        sys.stdout.flush()

        words = text.split(" ")
        for w_idx, word in enumerate(words):
            # Decide whether to simulate a typo+correction
            if (random.random() < correction_chance and
                    len(word) > 4 and word.isalpha()):
                self._simulate_typo_correction(word)
            else:
                # Normal streaming
                for char in word:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    delay = random.uniform(0.015, 0.045) * self._mood_factor
                    time.sleep(delay)

            # Space between words
            if w_idx < len(words) - 1:
                sys.stdout.write(" ")
                sys.stdout.flush()
                time.sleep(random.uniform(0.02, 0.06))

        sys.stdout.write("\n")
        sys.stdout.flush()

    def _simulate_typo_correction(self, correct_word: str) -> None:
        """Simulate typing a wrong word, pausing, backspacing, and retyping.

        Args:
            correct_word: The word that should ultimately appear.
        """
        # Generate a plausible typo (swap two adjacent chars)
        chars = list(correct_word)
        swap_pos = random.randint(0, len(chars) - 2)
        chars[swap_pos], chars[swap_pos + 1] = chars[swap_pos + 1], chars[swap_pos]
        typo = "".join(chars)

        # Type the typo partially (2-4 chars)
        type_count = min(random.randint(2, 4), len(typo))
        for char in typo[:type_count]:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(random.uniform(0.02, 0.05))

        # Pause (realizing the mistake)
        time.sleep(random.uniform(0.3, 0.6))

        # Backspace
        for _ in range(type_count):
            sys.stdout.write("\b \b")
            sys.stdout.flush()
            time.sleep(random.uniform(0.03, 0.06))

        # Small pause before retyping
        time.sleep(random.uniform(0.1, 0.2))

        # Type the correct word
        for char in correct_word:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(random.uniform(0.02, 0.04))
