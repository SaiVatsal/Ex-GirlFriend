# Prarthana-GPT v2.0

A deeply humanized, emotionally intelligent AI companion built from scratch in pure PyTorch. Trains a character-level language model to embody **Prarthana** — a warm, thoughtful, empathetic conversational partner who detects your emotions, apologizes when scolded, remembers your preferences, thinks out loud, sees your screen, talks with you by voice, and genuinely cares.

## What's New in v2.0

| Feature | Description |
|---------|-------------|
| 🎭 **Emotion Detection** | Detects 8 emotional states (angry, sad, happy, frustrated, scolding, loving, curious, neutral) with intensity scoring |
| 🙏 **Self-Correction** | Apologizes sincerely when scolded, tracks mistakes, escalates apology depth |
| 💬 **Conversation Memory** | Remembers your name, preferences, past topics, emotional patterns across sessions |
| 🎨 **Dynamic Personality** | Prarthana's mood evolves based on conversation — playful, thoughtful, concerned, excited |
| 🌊 **Streaming Output** | Human-like typing animation with variable speed and natural pauses |
| 🔄 **Extended Thinking** | Chain-of-thought reasoning with visible thinking steps for complex questions |
| 🎯 **Agent Tools** | Web search, file reading, code execution, calculator, reminders |
| 👁️ **Screen Vision** | Real-time screen capture & analysis via Gemini/OpenAI vision APIs |
| 🎤 **Voice Conversation** | Full-duplex voice I/O with faster-whisper STT + Coqui XTTS v2 TTS |
| 🧪 **Hybrid Brain** | Local model for personality + API (Gemini/OpenAI/Claude) for factual accuracy |
| 📊 **Dashboard** | Real-time conversation analytics with emotion graphs and relationship health |

## Architecture

- **Pre-LayerNorm GPT** with 4 layers, 4 attention heads, 192-dim embeddings (~1.8M params)
- Multi-head causal self-attention with scaled dot-product and triangular mask
- GELU-activated feed-forward network with 4× expansion
- Character-level tokenization with JSON-persisted vocabulary
- Weight-tied token embeddings and LM head
- Early-exit generation with `stop_ids` for responsive CPU inference

## Quick Start

### Train
```bash
python train.py
```
Reads `prarthana_corpus.txt` (200+ conversations), builds vocabulary, and trains for 3,000 steps (~5-8 min on CPU).

### Chat (Full Experience)
```bash
python chat.py
```

### Chat with Voice
```bash
pip install faster-whisper TTS sounddevice soundfile webrtcvad
python chat.py --voice
```

### Chat with Screen Vision
```bash
pip install mss Pillow requests
# Set API key: set GEMINI_API_KEY=your_key_here (Windows)
python chat.py --screen
```

### Chat with API Brain (Hybrid Intelligence)
```bash
# Set API key for any provider:
# set GEMINI_API_KEY=your_key
# set OPENAI_API_KEY=your_key
# set ANTHROPIC_API_KEY=your_key
python chat.py --api-brain
```

### Fast Mode (No Animation)
```bash
python chat.py --fast
```

### Test
```bash
python -m pytest tests/ -v
```

## In-Chat Commands

| Command | Description |
|---------|-------------|
| `/dashboard` | Show conversation analytics dashboard |
| `/mood` | See Prarthana's current mood |
| `/memory` | See what Prarthana remembers about you |
| `/think` | Toggle thinking display on/off |
| `/screen` | Take & describe a screenshot |
| `/voice` | Enter voice conversation mode |
| `quit`/`exit` | End session |

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters + v2.0 feature flags |
| `tokenizer.py` | Character-level vocab builder with JSON persistence |
| `data.py` | Corpus loader, sanitizer, train/val split, batch sampler |
| `model.py` | Full Transformer: attention, FFN, blocks, LM head, generate |
| `train.py` | Training loop with eval, checkpointing, and resume |
| `chat.py` | **v2.0 integrated CLI** with all features |
| `prarthana_corpus.txt` | **200+ conversations** covering all emotional scenarios |
| `emotion_engine.py` | Emotion detection & empathetic response generation |
| `self_correction.py` | Apology system & mistake tracking |
| `memory.py` | Persistent conversation memory (JSON-backed) |
| `personality.py` | Dynamic mood & personality modulation |
| `streaming.py` | Natural typing animation with variable speed |
| `thinking_engine.py` | Chain-of-thought reasoning display |
| `agent_tools.py` | Web search, file ops, code exec, reminders, calc |
| `screen_vision.py` | Screen capture & vision API analysis |
| `voice_engine.py` | STT (Whisper) + TTS (XTTS v2) + VAD |
| `hybrid_brain.py` | Local + API intelligence routing |
| `dashboard.py` | Conversation analytics & relationship health |

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--voice` | off | Enable voice I/O |
| `--screen` | off | Enable screen vision |
| `--api-brain` | off | Enable hybrid API brain |
| `--no-stream` | streaming on | Disable typing animation |
| `--no-thinking` | thinking on | Disable chain-of-thought |
| `--no-emotion` | emotion on | Disable emotion detection |
| `--fast` | off | Disable streaming + thinking for speed |

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- **Core pipeline**: No other external dependencies
- **Voice**: `faster-whisper`, `TTS`, `sounddevice`, `soundfile`, `webrtcvad`
- **Screen Vision**: `mss`, `Pillow`, `requests` + API key
- **API Brain**: `requests` + API key
- **Agent Tools (web search)**: `requests`
