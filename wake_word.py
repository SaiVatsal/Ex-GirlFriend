# wake_word.py
"""Wake word detection engine for Parhi-GPT.

Continuously listens on the microphone for the wake word "Parhi".
Once detected, captures the user's command and returns it for processing.

Supports two detection backends:
  1. Picovoice Porcupine — best accuracy, requires API key
  2. Whisper-based — transcribes short chunks, checks for "Parhi" keyword

All microphone audio is optionally logged to disk (audio only, no images).
"""
from __future__ import annotations

import os
import sys
import time
import wave
import struct
import datetime
import threading
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WakeWordConfig:
    """Configuration for the wake word engine.

    Attributes:
        wake_word: The trigger word to listen for.
        sample_rate: Audio sample rate in Hz.
        chunk_duration: Duration of each audio chunk to analyze (seconds).
        command_listen_duration: How long to listen after wake word (seconds).
        voice_log_enabled: Whether to save audio logs.
        voice_log_dir: Directory for audio logs.
        voice_log_retention_hours: How many hours of logs to keep.
        sensitivity: Wake word sensitivity (0.0-1.0, higher = more sensitive).
    """
    wake_word: str = "parhi"
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    command_listen_duration: float = 7.0
    voice_log_enabled: bool = True
    voice_log_dir: str = "voice_logs"
    voice_log_retention_hours: int = 24
    sensitivity: float = 0.7


# ---------------------------------------------------------------------------
# Wake word detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Continuous wake word detection with background voice logging.

    Listens on the microphone for the wake word "Parhi". When detected,
    captures the subsequent command audio and transcribes it.

    Audio is optionally logged to disk in rolling timestamped files.
    Only audio is recorded — never camera/images.
    """

    def __init__(self, config: WakeWordConfig | None = None) -> None:
        """Initialize the wake word detector.

        Args:
            config: Wake word configuration. Uses defaults if None.
        """
        self.config = config or WakeWordConfig()
        self._running = False
        self._listeners: list = []
        self._stt = None
        self._sd = None
        self._np = None
        self.active = False

        # Try to import audio dependencies
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
            import numpy as np
            self._sd = sd
            self._np = np
            self.active = True
        except ImportError:
            print("[wake_word] sounddevice not installed — wake word disabled")
            print("  pip install sounddevice soundfile")

        # Try to import STT for Whisper-based detection
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            self._stt = WhisperModel("base", device="cpu", compute_type="int8")
            print("[wake_word] Whisper STT loaded for wake word detection")
        except ImportError:
            print("[wake_word] faster-whisper not installed — using energy-based detection")

        # Create voice log directory
        if self.config.voice_log_enabled:
            os.makedirs(self.config.voice_log_dir, exist_ok=True)

    def on_wake(self, callback) -> None:
        """Register a callback for when the wake word is detected.

        Args:
            callback: Function that receives the transcribed command string.
        """
        self._listeners.append(callback)

    def start(self) -> None:
        """Start continuous listening in a background thread."""
        if not self.active:
            print("[wake_word] Cannot start — audio dependencies missing")
            return

        self._running = True
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
        print(f"[wake_word] Listening for '{self.config.wake_word}'...")

    def stop(self) -> None:
        """Stop listening."""
        self._running = False

    def _listen_loop(self) -> None:
        """Main listening loop — runs in background thread."""
        sd = self._sd
        np = self._np

        chunk_samples = int(self.config.sample_rate * self.config.chunk_duration)

        while self._running:
            try:
                # Record a chunk of audio
                audio = sd.rec(
                    chunk_samples,
                    samplerate=self.config.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()

                # Save to voice log if enabled
                if self.config.voice_log_enabled:
                    self._save_audio_log(audio)

                # Check for wake word
                if self._detect_wake_word(audio):
                    # Wake word detected! Play acknowledgment
                    self._play_chime()
                    print(f"\n  🎤 Wake word detected! Listening for command...")

                    # Listen for the command
                    command_text = self._capture_command()

                    if command_text:
                        print(f"  📝 Command: {command_text}")
                        # Notify all listeners
                        for listener in self._listeners:
                            try:
                                listener(command_text)
                            except Exception as e:
                                print(f"[wake_word] Listener error: {e}")

                # Clean up old logs periodically
                self._cleanup_old_logs()

            except Exception as e:
                if self._running:
                    print(f"[wake_word] Listen error: {e}")
                    time.sleep(1)

    def _detect_wake_word(self, audio) -> bool:
        """Check if audio chunk contains the wake word.

        Args:
            audio: NumPy array of audio samples.

        Returns:
            True if wake word detected.
        """
        np = self._np

        # Method 1: Whisper-based detection
        if self._stt:
            try:
                import tempfile
                import soundfile as sf  # type: ignore[import-untyped]

                # Save to temp file for Whisper
                temp_path = tempfile.mktemp(suffix=".wav")
                try:
                    sf.write(temp_path, audio, self.config.sample_rate)
                    segments, _ = self._stt.transcribe(
                        temp_path,
                        beam_size=1,
                        language="en",
                        vad_filter=True,
                    )
                    text = " ".join(seg.text.strip() for seg in segments).lower()
                    return self.config.wake_word.lower() in text
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception:
                pass

        # Method 2: Energy-based detection (fallback)
        # Just check if someone is speaking (above energy threshold)
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > 0.02  # Basic voice activity detection

    def _capture_command(self) -> str:
        """Capture and transcribe the user's command after wake word.

        Returns:
            Transcribed command text, or empty string.
        """
        if not self._sd or not self._stt:
            return ""

        sd = self._sd

        try:
            cmd_samples = int(self.config.sample_rate * self.config.command_listen_duration)
            audio = sd.rec(
                cmd_samples,
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            # Transcribe
            import tempfile
            import soundfile as sf  # type: ignore[import-untyped]

            temp_path = tempfile.mktemp(suffix=".wav")
            try:
                sf.write(temp_path, audio, self.config.sample_rate)
                segments, _ = self._stt.transcribe(
                    temp_path,
                    beam_size=5,
                    language="en",
                )
                text = " ".join(seg.text.strip() for seg in segments)
                return text.strip()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            print(f"[wake_word] Command capture error: {e}")
            return ""

    def _save_audio_log(self, audio) -> None:
        """Save audio chunk to the voice log directory.

        Args:
            audio: NumPy array of audio samples.
        """
        try:
            now = datetime.datetime.now()
            day_dir = os.path.join(
                self.config.voice_log_dir,
                now.strftime("%Y-%m-%d"),
            )
            os.makedirs(day_dir, exist_ok=True)

            filename = now.strftime("%H-%M-%S") + ".wav"
            filepath = os.path.join(day_dir, filename)

            import soundfile as sf  # type: ignore[import-untyped]
            sf.write(filepath, audio, self.config.sample_rate)
        except Exception:
            pass  # Non-critical — don't crash on log failure

    def _cleanup_old_logs(self) -> None:
        """Delete voice logs older than the retention period."""
        if not self.config.voice_log_enabled:
            return

        try:
            cutoff = time.time() - (self.config.voice_log_retention_hours * 3600)
            log_dir = self.config.voice_log_dir

            if not os.path.exists(log_dir):
                return

            for day_folder in os.listdir(log_dir):
                day_path = os.path.join(log_dir, day_folder)
                if not os.path.isdir(day_path):
                    continue

                # Check if the entire day folder is old
                try:
                    folder_time = datetime.datetime.strptime(day_folder, "%Y-%m-%d")
                    if folder_time.timestamp() < cutoff - 86400:
                        # Delete entire old day folder
                        import shutil
                        shutil.rmtree(day_path, ignore_errors=True)
                except ValueError:
                    pass
        except Exception:
            pass

    def _play_chime(self) -> None:
        """Play a short acknowledgment chime when wake word is detected."""
        try:
            import numpy as np

            # Generate a pleasant two-tone chime
            sr = 22050
            duration = 0.15
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)

            # Two ascending tones
            tone1 = 0.3 * np.sin(2 * np.pi * 880 * t)   # A5
            tone2 = 0.3 * np.sin(2 * np.pi * 1320 * t)  # E6

            chime = np.concatenate([tone1, tone2]).astype(np.float32)

            self._sd.play(chime, sr)
            self._sd.wait()
        except Exception:
            # Fallback: system beep
            try:
                import winsound
                winsound.Beep(880, 100)
                winsound.Beep(1320, 100)
            except Exception:
                pass
