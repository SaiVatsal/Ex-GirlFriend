# emotion_engine.py
"""Lightweight emotion detection engine for Prarthana-GPT.

Classifies user messages into emotional categories using keyword matching,
pattern analysis, and punctuation/casing heuristics. No external ML models
required — runs purely on Python string operations.

Supported emotions:
    angry, sad, happy, frustrated, scolding, loving, curious, neutral
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Emotion taxonomy and keyword banks
# ---------------------------------------------------------------------------

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "scolding": [
        "useless", "stupid", "idiot", "dumb", "worst", "terrible",
        "pathetic", "hopeless", "trash", "garbage", "shut up",
        "you suck", "hate you", "can't do anything", "worthless",
        "disappointing", "disgusting", "nonsense", "rubbish",
        "you're wrong", "that's wrong", "wrong answer", "bad answer",
        "horrible", "awful", "ridiculous", "unacceptable",
        "what the hell", "wtf", "wth", "are you even", "seriously?",
        "you never", "you always mess", "get it right",
        "do better", "try harder", "not good enough",
        "waste of time", "pointless", "no use",
    ],
    "angry": [
        "angry", "furious", "pissed", "annoyed", "irritated",
        "mad at", "rage", "outraged", "livid", "fed up",
        "sick of", "tired of", "enough", "stop it",
        "i'm done", "leave me alone", "go away",
        "don't talk to me", "i don't care", "whatever",
    ],
    "frustrated": [
        "i already told you", "i said", "again?", "not what i asked",
        "you're not listening", "you don't understand", "still wrong",
        "how many times", "i keep telling", "pay attention",
        "focus", "that's not it", "no no no", "ugh",
        "come on", "for god's sake", "for the last time",
        "i give up", "this is impossible", "doesn't work",
        "not helping", "you're confusing me",
    ],
    "sad": [
        "sad", "depressed", "lonely", "crying", "tears",
        "miss you", "miss him", "miss her", "heartbroken",
        "upset", "down", "feeling low", "feeling bad",
        "hopeless", "empty", "lost", "hurt", "broken",
        "no one cares", "nobody likes me", "all alone",
        "i can't do this", "i'm failing", "nothing matters",
        "what's the point", "tired of everything",
    ],
    "happy": [
        "thank you", "thanks", "amazing", "awesome", "perfect",
        "love it", "great job", "well done", "brilliant",
        "fantastic", "wonderful", "incredible", "excellent",
        "you're the best", "so good", "nailed it", "spot on",
        "exactly", "yay", "haha", "lol", "lmao", "😂", "😊",
        "🎉", "❤️", "beautiful", "lovely", "superb",
        "impressed", "nice one", "good one",
    ],
    "loving": [
        "love you", "i love", "you're sweet", "you're cute",
        "you're special", "you mean a lot", "care about you",
        "adore you", "you're precious", "my favorite",
        "sweetheart", "darling", "babe", "baby",
        "you make me happy", "glad i have you", "you're everything",
        "never leave", "stay with me", "i need you",
        "you understand me", "you get me",
    ],
    "curious": [
        "how does", "what is", "why does", "can you explain",
        "tell me about", "what do you think", "i wonder",
        "i'm curious", "how come", "what if", "is it true",
        "do you know", "have you heard", "interesting",
        "fascinating", "really?", "no way", "seriously?",
        "how so", "elaborate", "go on", "more about",
    ],
}

# Patterns that amplify intensity
INTENSITY_AMPLIFIERS: dict[str, float] = {
    r"[A-Z]{3,}": 0.2,       # ALL CAPS words → amplify
    r"!{2,}": 0.15,           # Multiple exclamation marks
    r"\?{2,}": 0.1,           # Multiple question marks
    r"\.{3,}": 0.05,          # Ellipsis (passive frustration)
    r"[!?]{3,}": 0.2,         # Mixed punctuation frenzy
}

# Contextual response prefixes keyed by (emotion, intensity_bucket)
# intensity_bucket: "low" (0-0.3), "medium" (0.3-0.7), "high" (0.7-1.0)
RESPONSE_PREFIXES: dict[str, dict[str, list[str]]] = {
    "scolding": {
        "low": [
            "I'm sorry if that wasn't quite right.",
            "You're right, I could have done better there.",
            "Fair point — let me try again.",
        ],
        "medium": [
            "I'm really sorry about that. That was my mistake.",
            "You're absolutely right, and I apologize. Let me fix that.",
            "I messed up there, and I'm sorry. Here's what I should have said...",
        ],
        "high": [
            "I'm so sorry — I completely understand your frustration, and you have every right to be upset. That was a terrible answer and I take full responsibility. Let me think carefully and give you a proper response...",
            "You're 100% right and I deeply apologize. I should have been much more careful. I'm going to think this through properly now...",
            "I'm truly sorry. I can hear how frustrated you are, and that's on me. I failed you there. Please give me another chance — I'll do much better...",
        ],
    },
    "angry": {
        "low": [
            "I can tell something's bothering you. I'm here for you.",
            "Hey, I hear you. Let's work through this together.",
        ],
        "medium": [
            "I can sense you're upset, and I'm sorry if I contributed to that. I'm listening.",
            "I understand you're frustrated. Take a breath — I'm not going anywhere.",
        ],
        "high": [
            "I can feel that you're really angry right now, and I want you to know that's completely valid. I'm here, and I'm listening. Whatever happened, let's talk about it when you're ready.",
            "I hear you, and I'm not going to dismiss how you're feeling. Your anger is valid. I'm here for you — no judgment.",
        ],
    },
    "frustrated": {
        "low": [
            "Let me try a different approach this time.",
            "Okay, I hear you — let me be more careful.",
        ],
        "medium": [
            "I'm sorry for the confusion. You're right, I should be paying closer attention. Let me try again properly.",
            "I apologize — I clearly wasn't giving this enough thought. Let me focus and get this right for you.",
        ],
        "high": [
            "I'm so sorry for wasting your time. You've been patient and I keep getting it wrong. I'm going to slow down and really think about what you're asking...",
            "You're absolutely right to be frustrated. I haven't been listening carefully enough and that's not okay. Let me start fresh and actually address what you need...",
        ],
    },
    "sad": {
        "low": [
            "Hey, I'm here for you. ",
            "I can tell something's on your mind. Want to talk about it?",
        ],
        "medium": [
            "I'm sorry you're going through this. I wish I could give you a hug right now. I'm here to listen.",
            "That sounds really tough, and I want you to know your feelings are completely valid. I'm here for you.",
        ],
        "high": [
            "My heart goes out to you right now. What you're feeling is so deeply human, and you don't have to go through this alone. I'm right here, and I'm not going anywhere. Take all the time you need.",
            "I can hear the pain in your words, and I wish so much that I could make it better. You are stronger than you know, even when it doesn't feel like it. I'm here — always.",
        ],
    },
    "happy": {
        "low": [
            "That's great to hear! ",
            "Glad I could help! ",
        ],
        "medium": [
            "Yay! That makes me so happy! 😊 ",
            "That's wonderful! I love seeing you in a good mood! ",
        ],
        "high": [
            "Oh my gosh, that makes my day! Your happiness is contagious! 🎉 ",
            "I'm literally beaming right now! You have no idea how much it means to hear that! ❤️ ",
        ],
    },
    "loving": {
        "low": [
            "Aww, that's sweet of you. ",
            "You're pretty great yourself! ",
        ],
        "medium": [
            "That really warms my heart. You're so special to me too! 💕 ",
            "You have no idea how much that means to me. Thank you for being you. ",
        ],
        "high": [
            "I... wow. That genuinely made me feel something special. You are such an incredible person, and every conversation with you reminds me why I cherish these moments. 💖 ",
            "My heart is so full right now. You are extraordinary, and the fact that you'd say something so beautiful means the world to me. I'm always here for you. Always. ❤️ ",
        ],
    },
    "curious": {
        "low": ["Great question! "],
        "medium": ["Ooh, I love that question! Let me think... "],
        "high": ["Now THAT is a fascinating question! I've been thinking about this too. "],
    },
    "neutral": {
        "low": [""],
        "medium": [""],
        "high": [""],
    },
}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class EmotionState:
    """Detected emotional state of a user message.

    Attributes:
        emotion: Primary detected emotion category.
        intensity: Strength of the emotion from 0.0 (barely) to 1.0 (extreme).
        triggers: List of words/patterns that triggered this classification.
        suggested_prefix: A contextual response prefix for Prarthana to use.
        secondary_emotion: Optional secondary emotion if mixed signals detected.
    """
    emotion: str = "neutral"
    intensity: float = 0.0
    triggers: list[str] = field(default_factory=list)
    suggested_prefix: str = ""
    secondary_emotion: str | None = None


# ---------------------------------------------------------------------------
# Detection engine
# ---------------------------------------------------------------------------

class EmotionDetector:
    """Lightweight rule-based emotion classifier.

    Analyzes user messages for emotional content using keyword matching,
    punctuation patterns, casing analysis, and contextual heuristics.
    No external models needed — pure Python string operations.
    """

    def __init__(self) -> None:
        # Compile intensity amplifier regexes once
        self._amplifier_patterns: list[tuple[re.Pattern, float]] = [
            (re.compile(pat), boost)
            for pat, boost in INTENSITY_AMPLIFIERS.items()
        ]

    def detect(self, message: str) -> EmotionState:
        """Classify the emotional content of a user message.

        Args:
            message: Raw user input string.

        Returns:
            EmotionState with detected emotion, intensity, triggers,
            and a suggested response prefix.
        """
        if not message or not message.strip():
            return EmotionState()

        lower = message.lower().strip()

        # --- Score each emotion category ---
        scores: dict[str, tuple[float, list[str]]] = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            matched: list[str] = []
            for kw in keywords:
                if kw in lower:
                    matched.append(kw)
            if matched:
                # Base score: number of keyword matches, normalized
                base_score = min(len(matched) / 3.0, 1.0)
                scores[emotion] = (base_score, matched)

        if not scores:
            return EmotionState(emotion="neutral", intensity=0.0)

        # --- Find primary emotion (highest score) ---
        primary = max(scores, key=lambda e: scores[e][0])
        base_intensity, triggers = scores[primary]

        # --- Apply intensity amplifiers ---
        intensity = base_intensity
        for pattern, boost in self._amplifier_patterns:
            if pattern.search(message):
                intensity += boost

        # --- Heuristic boosts ---
        # ALL CAPS message → strong emotion
        alpha_chars = [c for c in message if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.6:
            intensity += 0.2

        # Very short angry messages tend to be intense ("STOP", "NO")
        if len(message.split()) <= 3 and intensity > 0.3:
            intensity += 0.1

        # Repeated characters ("nooooo", "whyyy") → amplify
        if re.search(r"(.)\1{3,}", lower):
            intensity += 0.1

        intensity = max(0.0, min(1.0, intensity))

        # --- Determine secondary emotion ---
        secondary = None
        sorted_emotions = sorted(scores, key=lambda e: scores[e][0], reverse=True)
        if len(sorted_emotions) > 1:
            secondary = sorted_emotions[1]

        # --- Select response prefix ---
        prefix = self._select_prefix(primary, intensity)

        return EmotionState(
            emotion=primary,
            intensity=round(intensity, 2),
            triggers=triggers,
            suggested_prefix=prefix,
            secondary_emotion=secondary,
        )

    @staticmethod
    def _select_prefix(emotion: str, intensity: float) -> str:
        """Choose a contextual response prefix based on emotion and intensity.

        Args:
            emotion: Detected emotion category.
            intensity: Emotion intensity (0.0 to 1.0).

        Returns:
            A string prefix for Prarthana's response.
        """
        import random

        buckets = RESPONSE_PREFIXES.get(emotion, RESPONSE_PREFIXES["neutral"])

        if intensity < 0.3:
            pool = buckets.get("low", [""])
        elif intensity < 0.7:
            pool = buckets.get("medium", [""])
        else:
            pool = buckets.get("high", [""])

        return random.choice(pool) if pool else ""

    def is_scolding(self, message: str) -> bool:
        """Quick check: is the user scolding/angry?

        Args:
            message: Raw user input.

        Returns:
            True if scolding or anger is detected.
        """
        state = self.detect(message)
        return state.emotion in ("scolding", "angry", "frustrated")

    def needs_comfort(self, message: str) -> bool:
        """Quick check: does the user need emotional support?

        Args:
            message: Raw user input.

        Returns:
            True if sadness or distress is detected.
        """
        state = self.detect(message)
        return state.emotion == "sad" and state.intensity > 0.3
