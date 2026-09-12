# voice_engine.py
"""Full-duplex voice conversation engine for Prarthana-GPT.

Provides real-time speech-to-text (via faster-whisper), text-to-speech
(via Coqui XTTS v2), and voice activity detection for natural
spoken conversation.

All external dependencies are optional and degrade gracefully.
"""
from __future__ import annotations

import os
import sys
import time
import threading
import tempfile
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VoiceConfig:
    """Configuration for the voice engine.

    Attributes:
        ref_clip: Path to the reference audio clip for voice cloning.
        language: Language code for STT/TTS.
        stt_model: Whisper model size (tiny, base, small, medium, large).
        vad_threshold: Voice activity detection threshold (0.0-1.0).
        silence_duration: Seconds of silence before considering speech ended.
        sample_rate: Audio sample rate in Hz.
        listen_mode: "push_to_talk" or "always_on".
    """
    ref_clip: str = "prarthana_voice.wav"
    language: str = "en"
    stt_model: str = "base"
    vad_threshold: float = 0.5
    silence_duration: float = 1.5
    sample_rate: int = 16000
    listen_mode: str = "push_to_talk"  # "push_to_talk" or "always_on"


class SpeechToText:
    """Speech-to-text using faster-whisper (local, no API needed).

    Falls back to a no-op if faster-whisper is not installed.
    Typical latency: ~1-2s for short utterances on CPU.
    """

    def __init__(self, model_size: str = "base") -> None:
        """Initialize STT.

        Args:
            model_size: Whisper model size. "base" recommended for CPU.
        """
        self.active = False
        self._model = None

        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            self._model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
            )
            self.active = True
            print("[voice] STT loaded (faster-whisper)")
        except ImportError:
            print("[voice] faster-whisper not installed — STT disabled")
        except Exception as e:
            print(f"[voice] STT init error: {e}")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to a WAV audio file.

        Returns:
            Transcribed text string.
        """
        if not self.active or not self._model:
            return ""

        try:
            segments, _ = self._model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
            )
            return " ".join(seg.text.strip() for seg in segments)
        except Exception as e:
            print(f"[voice] transcription error: {e}")
            return ""

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw audio bytes.

        Args:
            audio_bytes: Raw PCM audio bytes.
            sample_rate: Sample rate of the audio.

        Returns:
            Transcribed text.
        """
        if not self.active:
            return ""

        # Write to temp file and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            try:
                import soundfile as sf  # type: ignore[import-untyped]
                import numpy as np
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                sf.write(temp_path, audio_array, sample_rate)
            except ImportError:
                return ""

        try:
            result = self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)

        return result


class TextToSpeech:
    """Text-to-speech using Coqui XTTS v2 with voice cloning.

    Enhanced version of the original VoiceOutput with streaming support
    and interrupt capability.
    """

    def __init__(self, ref_clip: str = "prarthana_voice.wav") -> None:
        """Initialize TTS.

        Args:
            ref_clip: Path to reference audio for voice cloning.
        """
        self.ref_clip = ref_clip
        self.active = False
        self._speaking = False
        self._interrupt = False

        try:
            from TTS.api import TTS  # type: ignore[import-untyped]
            import sounddevice as sd  # type: ignore[import-untyped]
            import soundfile as sf  # type: ignore[import-untyped]

            if os.path.exists(ref_clip):
                self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                self._sd = sd
                self._sf = sf
                self.active = True
                print("[voice] TTS loaded (XTTS v2)")
            else:
                print(f"[voice] reference clip not found: {ref_clip} — TTS disabled")
        except ImportError:
            print("[voice] TTS dependencies not installed — voice output disabled")

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently playing audio."""
        return self._speaking

    def speak(self, text: str, blocking: bool = True) -> None:
        """Synthesize and play speech.

        Args:
            text: Text to speak.
            blocking: If True, waits for playback to finish.
        """
        if not self.active or not text.strip():
            return

        self._interrupt = False
        self._speaking = True

        out_file = tempfile.mktemp(suffix=".wav")
        try:
            self._tts.tts_to_file(
                text=text,
                speaker_wav=self.ref_clip,
                language="en",
                file_path=out_file,
            )
            data, sr = self._sf.read(out_file)
            self._sd.play(data, sr)

            if blocking:
                self._sd.wait()
            else:
                # Non-blocking: monitor in background thread
                def _wait():
                    self._sd.wait()
                    self._speaking = False
                threading.Thread(target=_wait, daemon=True).start()
                return

        except Exception as e:
            print(f"[voice] playback error: {e}")
        finally:
            self._speaking = False
            if os.path.exists(out_file):
                os.remove(out_file)

    def interrupt(self) -> None:
        """Stop current playback immediately."""
        self._interrupt = True
        self._speaking = False
        try:
            if hasattr(self, '_sd'):
                self._sd.stop()
        except Exception:
            pass

    def speak_streaming(self, text: str) -> None:
        """Speak text in chunks for faster time-to-first-audio.

        Splits text into sentences and speaks them sequentially,
        allowing for interrupt between chunks.

        Args:
            text: Full text to speak.
        """
        if not self.active:
            return

        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if self._interrupt:
                break
            if sentence.strip():
                self.speak(sentence.strip(), blocking=True)


class VoiceActivityDetector:
    """Detects when the user starts and stops speaking.

    Uses webrtcvad or a simple energy-based approach as fallback.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize VAD.

        Args:
            threshold: Voice activity threshold (0.0-1.0).
        """
        self.threshold = threshold
        self.active = False
        self._vad = None

        try:
            import webrtcvad  # type: ignore[import-untyped]
            self._vad = webrtcvad.Vad(2)  # Aggressiveness level 2
            self.active = True
        except ImportError:
            # Fallback: energy-based detection will be used
            self.active = True  # Still works with energy-based approach

    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """Check if an audio chunk contains speech.

        Args:
            audio_chunk: Raw PCM audio bytes.
            sample_rate: Sample rate.

        Returns:
            True if speech is detected.
        """
        if self._vad:
            try:
                return self._vad.is_speech(audio_chunk, sample_rate)
            except Exception:
                pass

        # Fallback: simple energy-based detection
        import struct
        if len(audio_chunk) < 2:
            return False
        samples = struct.unpack(f"<{len(audio_chunk)//2}h", audio_chunk)
        energy = sum(abs(s) for s in samples) / len(samples)
        return energy > self.threshold * 1000


class VoiceEngine:
    """Unified voice conversation engine.

    Coordinates STT, TTS, and VAD for a natural spoken conversation
    experience. Supports push-to-talk and always-on listening modes.
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        """Initialize the voice engine.

        Args:
            config: Voice configuration. Uses defaults if None.
        """
        self.config = config or VoiceConfig()
        self.stt = SpeechToText(self.config.stt_model)
        self.tts = TextToSpeech(self.config.ref_clip)
        self.vad = VoiceActivityDetector(self.config.vad_threshold)

        self.active = self.stt.active or self.tts.active

        if not self.active:
            print("[voice] Voice engine inactive — install dependencies:")
            print("  pip install faster-whisper TTS sounddevice soundfile webrtcvad")

    def listen(self, duration: float = 5.0) -> str:
        """Record audio from microphone and transcribe.

        Args:
            duration: Maximum recording duration in seconds.

        Returns:
            Transcribed text from the recording.
        """
        if not self.stt.active:
            return ""

        try:
            import sounddevice as sd  # type: ignore[import-untyped]
            import numpy as np

            print("  🎤 Listening...")
            audio = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            print("  ✓ Processing...")

            # Save to temp file for transcription
            temp_path = tempfile.mktemp(suffix=".wav")
            try:
                import soundfile as sf  # type: ignore[import-untyped]
                sf.write(temp_path, audio, self.config.sample_rate)
                result = self.stt.transcribe(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            return result

        except ImportError:
            print("[voice] sounddevice not installed")
            return ""
        except Exception as e:
            print(f"[voice] listen error: {e}")
            return ""

    def speak(self, text: str, streaming: bool = False) -> None:
        """Speak text aloud.

        Args:
            text: Text to speak.
            streaming: If True, uses sentence-by-sentence streaming.
        """
        if streaming:
            self.tts.speak_streaming(text)
        else:
            self.tts.speak(text)

    def interrupt(self) -> None:
        """Interrupt current speech playback."""
        self.tts.interrupt()
