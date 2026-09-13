# config.py 
"""Parhi-GPT hyperparameter configuration and device detection."""
from __future__ import annotations

import dataclasses
import torch


def detect_device() -> str:
    """Auto-detect the best available compute device.

    Returns:
        One of 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclasses.dataclass
class ParhiConfig:
    """Central hyperparameter store for Parhi-GPT.

    All training, model, and runtime knobs live here so every module
    imports one authoritative source of truth.

    Defaults are tuned for responsive CPU training (~5-8 min).
    For GPU training, increase n_embd to 384, n_layer to 6, n_head to 6,
    block_size to 256, batch_size to 64, and max_iters to 10_000.
    """

    # --- context & batching ---
    block_size: int = 128     #text context as per the cpu length      # T_ctx — maximum sequence length
    batch_size: int = 32           # B — sequences per gradient step

    # --- model dimensions (CPU-friendly: ~1.8M params) ---
    n_embd: int = 192              # d_model — embedding / hidden size
    n_head: int = 4                # h — number of attention heads
    n_layer: int = 4               # number of Transformer decoder blocks
    dropout: float = 0.2           # dropout probability

    # --- optimizer ---
    learning_rate: float = 5e-4
    weight_decay: float = 0.01

    # --- training schedule ---
    max_iters: int = 3_000
    eval_interval: int = 200       # steps between validation runs
    eval_iters: int = 50           # batches averaged per eval

    # --- runtime ---
    device: str = dataclasses.field(default_factory=detect_device)

    # --- vocabulary (set after tokenizer builds the vocab) ---
    vocab_size: int = 0

    # ======================================================================
    # v2.0 Feature Flags — toggle new features on/off
    # ======================================================================

    # --- Emotion & Personality ---
    enable_emotion_detection: bool = True    # Detect user mood & respond empathetically
    enable_self_correction: bool = True      # Track mistakes & generate apologies
    enable_personality: bool = True          # Dynamic mood system for Parhi
    enable_streaming: bool = True            # Natural typing animation
    streaming_speed: float = 1.0             # Typing speed multiplier (lower = faster)

    # --- Memory ---
    enable_memory: bool = True               # Persistent conversation memory
    memory_path: str = "parhi_memory.json"

    # --- Vision ---
    enable_screen_vision: bool = False       # Screen capture & analysis (needs API key)
    vision_api_provider: str = "gemini"      # "gemini" or "openai"

    # --- Voice ---
    enable_voice_input: bool = False         # Speech-to-text input
    enable_voice_output: bool = False        # Text-to-speech output
    voice_ref_clip: str = "parhi_voice.wav"

    # --- Intelligence ---
    enable_hybrid_brain: bool = False        # API-backed factual intelligence
    hybrid_api_provider: str = "gemini"      # "gemini", "openai", or "claude"
    enable_thinking: bool = True             # Extended thinking / chain-of-thought
    show_thinking: bool = True               # Display thinking steps to user

    # --- Tools ---
    enable_agent_tools: bool = True          # Agentic tool use (search, calc, etc.)

    # --- Dashboard ---
    enable_dashboard: bool = True            # Conversation analytics
    show_mini_status: bool = True            # Compact status bar after each exchange

    # ======================================================================
    # v3.0 Background Service & JARVIS Features
    # ======================================================================

    # --- Background Service ---
    enable_background_service: bool = False  # Always-on background mode
    wake_word: str = "parhi"                 # Wake word for voice activation
    voice_log_enabled: bool = True           # Record ambient audio (no camera)
    voice_log_dir: str = "voice_logs"        # Directory for audio logs
    voice_log_retention_hours: int = 24      # How long to keep logs
    startup_greeting: bool = True            # Greet user on system start

    # --- System Commands ---
    enable_system_commands: bool = True      # JARVIS-style system controls
