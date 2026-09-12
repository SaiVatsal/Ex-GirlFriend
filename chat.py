# chat.py
"""Prarthana-GPT v2.0 — Humanized, Emotionally Intelligent Chat Engine.

Usage:
    python chat.py [--checkpoint PATH] [--device DEVICE] [--voice]
                   [--screen] [--no-stream] [--no-thinking] [--api-brain]

The fully integrated chat experience with:
  - Emotion detection & empathetic responses
  - Self-correction & apology system
  - Dynamic personality & mood
  - Persistent conversation memory
  - Natural streaming typing animation
  - Extended thinking / chain-of-thought
  - Agentic tool use (search, files, code, timers)
  - Screen vision (real-time, optional)
  - Voice conversation (optional)
  - Hybrid local + API intelligence (optional)
  - Conversation analytics dashboard
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from config import PrarthanaConfig, detect_device
from tokenizer import CharTokenizer
from model import PrarthanaGPT


DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prarthana_model.pt")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
class PrarthanaCLI:
    """Interactive terminal interface for chatting with Prarthana-GPT v2.0.

    Integrates all v2.0 subsystems: emotion engine, self-correction,
    personality, memory, streaming, thinking, tools, screen vision,
    voice, hybrid brain, and dashboard.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        config_overrides: dict | None = None,
    ) -> None:
        """Initialize the CLI by loading model and all subsystems.

        Args:
            checkpoint_path: Path to the ``prarthana_model.pt`` checkpoint.
            device: Compute device (auto-detected if ``None``).
            config_overrides: Optional dict of config overrides.
        """
        self.device = device or detect_device()

        if not os.path.exists(checkpoint_path):
            print(f"[error] checkpoint not found: {checkpoint_path}")
            print("[hint]  run 'python train.py' first to train the model.")
            sys.exit(1)

        # Load checkpoint
        ckpt = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Rebuild tokenizer from saved vocab
        self.tokenizer = CharTokenizer()
        self.tokenizer.stoi = ckpt["vocab_stoi"]
        self.tokenizer.itos = {
            int(k) if isinstance(k, str) else k: v
            for k, v in ckpt["vocab_itos"].items()
        }

        # Rebuild config from checkpoint if available
        if "config" in ckpt:
            saved_cfg = ckpt["config"]
            saved_cfg["device"] = self.device
            self.config = PrarthanaConfig(**{
                k: v for k, v in saved_cfg.items()
                if k in PrarthanaConfig.__dataclass_fields__
            })
        else:
            self.config = PrarthanaConfig(
                vocab_size=self.tokenizer.vocab_size,
                device=self.device,
            )

        # Apply overrides
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # Rebuild model
        self.model = PrarthanaGPT(self.config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._history: str = ""

        # Precompute newline stop ID
        self._newline_ids: list[int] = []
        if "\n" in self.tokenizer.stoi:
            self._newline_ids.append(self.tokenizer.stoi["\n"])

        # --- Initialize v2.0 subsystems ---
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        """Initialize all v2.0 feature subsystems based on config flags."""

        # Emotion detection
        self.emotion_detector = None
        if self.config.enable_emotion_detection:
            try:
                from emotion_engine import EmotionDetector
                self.emotion_detector = EmotionDetector()
                print("[✓] Emotion detection enabled")
            except ImportError as e:
                print(f"[✗] Emotion detection failed: {e}")

        # Self-correction
        self.self_corrector = None
        if self.config.enable_self_correction:
            try:
                from self_correction import SelfCorrector
                self.self_corrector = SelfCorrector()
                print("[✓] Self-correction system enabled")
            except ImportError as e:
                print(f"[✗] Self-correction failed: {e}")

        # Personality
        self.personality = None
        if self.config.enable_personality:
            try:
                from personality import PersonalityEngine
                self.personality = PersonalityEngine()
                print("[✓] Dynamic personality enabled")
            except ImportError as e:
                print(f"[✗] Personality system failed: {e}")

        # Memory
        self.memory = None
        if self.config.enable_memory:
            try:
                from memory import MemoryManager
                self.memory = MemoryManager(self.config.memory_path)
                print("[✓] Persistent memory enabled")
            except ImportError as e:
                print(f"[✗] Memory system failed: {e}")

        # Streaming
        self.streamer = None
        if self.config.enable_streaming:
            try:
                from streaming import StreamingPrinter
                self.streamer = StreamingPrinter(
                    speed_multiplier=self.config.streaming_speed
                )
                print("[✓] Streaming output enabled")
            except ImportError as e:
                print(f"[✗] Streaming failed: {e}")

        # Thinking engine
        self.thinker = None
        if self.config.enable_thinking:
            try:
                from thinking_engine import ThinkingEngine
                self.thinker = ThinkingEngine(
                    visible_thinking=self.config.show_thinking
                )
                print("[✓] Extended thinking enabled")
            except ImportError as e:
                print(f"[✗] Thinking engine failed: {e}")

        # Agent tools
        self.tools = None
        if self.config.enable_agent_tools:
            try:
                from agent_tools import AgentTools
                self.tools = AgentTools()
                print("[✓] Agent tools enabled")
            except ImportError as e:
                print(f"[✗] Agent tools failed: {e}")

        # Screen vision
        self.screen = None
        if self.config.enable_screen_vision:
            try:
                from screen_vision import ScreenVision
                self.screen = ScreenVision(
                    api_provider=self.config.vision_api_provider
                )
                if self.screen.active:
                    print("[✓] Screen vision enabled")
                else:
                    print("[✗] Screen vision: dependencies not installed (pip install mss Pillow)")
                    self.screen = None
            except ImportError as e:
                print(f"[✗] Screen vision failed: {e}")

        # Voice engine
        self.voice = None
        if self.config.enable_voice_input or self.config.enable_voice_output:
            try:
                from voice_engine import VoiceEngine, VoiceConfig
                vc = VoiceConfig(ref_clip=self.config.voice_ref_clip)
                self.voice = VoiceEngine(vc)
                if self.voice.active:
                    print("[✓] Voice engine enabled")
                else:
                    self.voice = None
            except ImportError as e:
                print(f"[✗] Voice engine failed: {e}")

        # Hybrid brain
        self.brain = None
        if self.config.enable_hybrid_brain:
            try:
                from hybrid_brain import HybridBrain, HybridConfig
                bc = HybridConfig(api_provider=self.config.hybrid_api_provider)
                self.brain = HybridBrain(bc)
                print("[✓] Hybrid brain enabled")
            except ImportError as e:
                print(f"[✗] Hybrid brain failed: {e}")

        # Dashboard
        self.dashboard = None
        if self.config.enable_dashboard:
            try:
                from dashboard import Dashboard
                self.dashboard = Dashboard(memory=self.memory)
                print("[✓] Dashboard enabled")
            except ImportError as e:
                print(f"[✗] Dashboard failed: {e}")

    def _generate_local(
        self,
        user_input: str,
        temperature: float = 0.8,
        top_k: int = 30,
        max_tokens: int = 500,
    ) -> str:
        """Generate a response using the local Transformer model.

        Args:
            user_input: The user's message text.
            temperature: Sampling temperature.
            top_k: Number of top logits to keep.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response string.
        """
        prompt = f"User: {user_input.strip()}\nPrarthana:"
        self._history += prompt

        # Encode
        try:
            ids = self.tokenizer.encode(self._history)
        except KeyError:
            safe = "".join(
                ch for ch in self._history if ch in self.tokenizer.stoi
            )
            ids = self.tokenizer.encode(safe)

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Generate
        output = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_ids=self._newline_ids,
        )

        # Decode new tokens
        new_ids = output[0, len(ids):].tolist()
        raw = self.tokenizer.decode(new_ids)

        # Extract response
        response = raw
        for stop in ("\nUser:", "\n\nUser:"):
            if stop in response:
                response = response[:response.index(stop)]
                break

        response = response.strip()
        self._history += " " + response + "\n\n"

        # Truncate history
        max_hist_chars = self.config.block_size * 4
        if len(self._history) > max_hist_chars:
            self._history = self._history[-max_hist_chars:]

        return response

    def respond(self, user_input: str) -> str:
        """Generate Prarthana's response with full v2.0 pipeline.

        Orchestrates: emotion detection → self-correction check →
        thinking → tool use → local/API generation → personality
        touches → memory update → dashboard recording.

        Args:
            user_input: The user's message text.

        Returns:
            Complete response string.
        """
        # --- Step 1: Detect emotion ---
        emotion = None
        if self.emotion_detector:
            from emotion_engine import EmotionState
            emotion = self.emotion_detector.detect(user_input)

        # --- Step 2: Update personality mood ---
        gen_params = {"temperature": 0.8, "top_k": 30, "max_tokens": 500}
        if self.personality and emotion:
            self.personality.update_mood(emotion)
            gen_params = self.personality.get_generation_params()

        # --- Step 3: Check for mistake correction ---
        apology_prefix = ""
        is_mistake = False
        if self.self_corrector and emotion:
            if (self.self_corrector.detect_mistake_indication(user_input) or
                    (emotion.emotion in ("scolding", "angry", "frustrated") and
                     emotion.intensity > 0.3)):
                is_mistake = True
                self.self_corrector.register_mistake(user_input, emotion)
                apology_prefix = self.self_corrector.generate_apology(emotion)

        # --- Step 4: Check for tool requests ---
        tool_response = None
        if self.tools and not is_mistake:
            tool_name = self.tools.detect_tool_request(user_input)
            if tool_name:
                # Handle screen-related tools
                if tool_name == "screen_vision" or any(
                    w in user_input.lower()
                    for w in ["my screen", "what do you see", "look at"]
                ):
                    if self.screen:
                        screen_desc = self.screen.describe()
                        tool_response = screen_desc
                    else:
                        tool_response = "I'd love to see your screen, but screen vision isn't enabled. Run with --screen flag and make sure you have the required packages (pip install mss Pillow) and an API key set."
                else:
                    result = self.tools.execute_tool(tool_name, user_input)
                    if result.display_text:
                        tool_response = result.display_text

        # --- Step 5: Extended thinking ---
        thinking_display = ""
        if self.thinker and not tool_response and not is_mistake:
            if self.thinker.needs_thinking(user_input):
                process = self.thinker.generate_thinking(user_input)
                thinking_display = self.thinker.format_thinking_display(process)

        # --- Step 6: Generate response ---
        if tool_response:
            response = tool_response
        elif self.brain and self.brain.needs_api(user_input):
            # Use hybrid brain (API)
            emotion_ctx = ""
            if emotion:
                emotion_ctx = f"User is feeling: {emotion.emotion} (intensity: {emotion.intensity})"
            memory_ctx = ""
            if self.memory:
                memory_ctx = self.memory.get_context_summary()
            screen_ctx = ""
            if self.screen and self.screen.active:
                screen_ctx = self.screen.describe()

            response = self.brain.query_api(
                user_input,
                emotion_context=emotion_ctx,
                memory_context=memory_ctx,
                screen_context=screen_ctx,
            )
            if not response:
                # Fallback to local
                response = self._generate_local(user_input, **gen_params)
        else:
            # Use local model
            response = self._generate_local(user_input, **gen_params)

        # --- Step 7: Apply emotion prefix ---
        if apology_prefix:
            response = apology_prefix
        elif emotion and emotion.suggested_prefix and emotion.emotion != "neutral":
            # Only prepend if the response doesn't already start similarly
            if not any(response.lower().startswith(w) for w in ["i'm sorry", "sorry", "i apologize"]):
                response = emotion.suggested_prefix + " " + response

        # --- Step 8: Apply personality touches ---
        if self.personality and not is_mistake:
            response = self.personality.add_personality_touches(response)

        # --- Step 9: Self-correction tracking ---
        if self.self_corrector:
            self.self_corrector.record_exchange(user_input, response, emotion)
            if not is_mistake:
                self.self_corrector.reset_streak()

        # --- Step 10: Update memory ---
        if self.memory:
            self.memory.record_message(user_input, response)
            if emotion:
                self.memory.record_emotion(
                    emotion.emotion, emotion.intensity, user_input
                )

        # --- Step 11: Update dashboard ---
        if self.dashboard:
            self.dashboard.record_exchange(
                user_input, response, emotion, is_mistake
            )

        return response, thinking_display

    def _output_response(
        self, response: str, thinking_display: str = ""
    ) -> None:
        """Output the response with appropriate formatting.

        Uses streaming animation if enabled, otherwise prints directly.

        Args:
            response: The response text to output.
            thinking_display: Optional thinking display to show first.
        """
        # Show thinking steps if available
        if thinking_display:
            print(thinking_display)

        # Stream or print
        if self.streamer:
            mood = "calm"
            if self.personality:
                mood = self.personality.mood_label
                self.streamer.mood = mood
                self.streamer._mood_factor = self.streamer.__class__.__dict__.get(
                    '_mood_factor', 1.0
                )
                from streaming import MOOD_SPEED
                self.streamer._mood_factor = MOOD_SPEED.get(mood, 1.0)

            self.streamer.stream(
                response,
                prefix="Prarthana: ",
                show_indicator=True,
            )
        else:
            print(f"\nPrarthana: {response}\n")

        # Mini status bar
        if self.dashboard and self.config.show_mini_status:
            self.dashboard.print_mini_status()

    def run(self) -> None:
        """Start the interactive REPL loop.

        Supports special commands:
          /dashboard  — show full analytics dashboard
          /mood       — show Prarthana's current mood
          /memory     — show what Prarthana remembers
          /think on   — enable visible thinking
          /think off  — disable visible thinking
          /voice      — toggle voice mode
          /screen     — take & describe a screenshot
          quit/exit   — end session
        """
        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + "  🧠 Prarthana-GPT v2.0 — Humanized AI Companion  ".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print("║" + "  Commands:".ljust(58) + "║")
        print("║" + "    /dashboard  — conversation analytics".ljust(58) + "║")
        print("║" + "    /mood       — Prarthana's current mood".ljust(58) + "║")
        print("║" + "    /memory     — what she remembers about you".ljust(58) + "║")
        print("║" + "    /think      — toggle thinking display".ljust(58) + "║")
        print("║" + "    /screen     — look at your screen".ljust(58) + "║")
        print("║" + "    /voice      — voice conversation mode".ljust(58) + "║")
        print("║" + "    quit/exit   — end session".ljust(58) + "║")
        print("╚" + "═" * 58 + "╝")
        print()

        # Personalized greeting
        if self.memory:
            greeting_ctx = self.memory.get_greeting_context()
            if greeting_ctx:
                name = self.memory.get_user_name()
                if name:
                    print(f"Prarthana: Welcome back, {name}! I missed you! 😊\n")
                else:
                    print("Prarthana: Hey there! Great to see you again! 😊\n")
            else:
                print("Prarthana: Hi! I'm Prarthana — your AI companion. What's on your mind? 💫\n")
        else:
            print("Prarthana: Hi! I'm Prarthana — let's chat! 💫\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._exit_session()
                break

            if not user_input:
                continue

            # --- Handle special commands ---
            if user_input.lower() in ("quit", "exit", "bye"):
                self._exit_session()
                break

            if user_input.lower() == "/dashboard":
                if self.dashboard:
                    self.dashboard.print_dashboard()
                else:
                    print("  [dashboard not enabled]")
                continue

            if user_input.lower() == "/mood":
                if self.personality:
                    desc = self.personality.get_mood_description()
                    print(f"\n  🎭 Prarthana is currently {desc}\n")
                else:
                    print("  [personality system not enabled]")
                continue

            if user_input.lower() == "/memory":
                if self.memory:
                    ctx = self.memory.get_context_summary()
                    if ctx:
                        print(f"\n  🧠 What I remember about you:\n")
                        for line in ctx.split("\n"):
                            print(f"    {line}")
                        print()
                    else:
                        print("\n  🧠 I don't have any memories yet — we just started!\n")
                else:
                    print("  [memory system not enabled]")
                continue

            if user_input.lower() in ("/think", "/think on", "/think off"):
                if self.thinker:
                    if "off" in user_input.lower():
                        self.thinker.visible_thinking = False
                        print("  [thinking display OFF]")
                    else:
                        self.thinker.visible_thinking = not self.thinker.visible_thinking
                        state = "ON" if self.thinker.visible_thinking else "OFF"
                        print(f"  [thinking display {state}]")
                else:
                    print("  [thinking engine not enabled]")
                continue

            if user_input.lower() == "/screen":
                if self.screen and self.screen.active:
                    print("\n  👁️ Looking at your screen...")
                    desc = self.screen.describe()
                    print(f"\n  Prarthana: {desc}\n")
                else:
                    print("  [screen vision not enabled — run with --screen]")
                continue

            if user_input.lower() == "/voice":
                if self.voice:
                    self._voice_conversation()
                else:
                    print("  [voice engine not enabled — run with --voice]")
                continue

            # --- Generate response ---
            response, thinking_display = self.respond(user_input)
            self._output_response(response, thinking_display)

            # Voice output
            if self.voice and self.config.enable_voice_output:
                self.voice.speak(response)

    def _voice_conversation(self) -> None:
        """Enter voice conversation mode."""
        if not self.voice:
            return

        print("\n  🎤 Voice mode activated! Speak after the prompt.")
        print("  Type 'exit' or press Ctrl+C to return to text mode.\n")

        while True:
            try:
                text = self.voice.listen(duration=5.0)
                if not text:
                    print("  (didn't catch that — try again)")
                    continue

                print(f"  You said: {text}")

                if text.lower().strip() in ("exit", "quit", "stop"):
                    print("  [returning to text mode]")
                    break

                response, thinking = self.respond(text)
                print(f"\n  Prarthana: {response}\n")
                self.voice.speak(response)

            except KeyboardInterrupt:
                print("\n  [returning to text mode]")
                break

    def _exit_session(self) -> None:
        """Handle graceful session exit."""
        print("\nPrarthana: Goodbye! It was wonderful chatting with you. ")
        print("           Take care of yourself — I'll be right here")
        print("           whenever you want to talk again! 💕\n")

        # Save memory
        if self.memory:
            self.memory.end_session()
            print("  [💾 memories saved]")

        # Show final dashboard
        if self.dashboard:
            self.dashboard.print_dashboard()


def main() -> None:
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Chat with Prarthana-GPT v2.0"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (auto-detected if omitted).",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help="Enable voice I/O (requires TTS, faster-whisper, sounddevice).",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        default=False,
        help="Enable screen vision (requires mss, Pillow, and API key).",
    )
    parser.add_argument(
        "--api-brain",
        action="store_true",
        default=False,
        help="Enable hybrid API brain (requires API key).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming output animation.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help="Disable extended thinking display.",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        default=False,
        help="Disable emotion detection.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode: disable streaming and thinking for quick responses.",
    )
    args = parser.parse_args()

    # Build config overrides from CLI args
    overrides: dict = {}

    if args.voice:
        overrides["enable_voice_input"] = True
        overrides["enable_voice_output"] = True

    if args.screen:
        overrides["enable_screen_vision"] = True

    if args.api_brain:
        overrides["enable_hybrid_brain"] = True

    if args.no_stream:
        overrides["enable_streaming"] = False

    if args.no_thinking:
        overrides["enable_thinking"] = False

    if args.no_emotion:
        overrides["enable_emotion_detection"] = False

    if args.fast:
        overrides["enable_streaming"] = False
        overrides["enable_thinking"] = False
        overrides["show_mini_status"] = False

    cli = PrarthanaCLI(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config_overrides=overrides,
    )
    cli.run()


if __name__ == "__main__":
    main()
