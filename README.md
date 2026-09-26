# Parhi-GPT v2.0

Parhi-GPT is a small personal AI companion built from scratch using PyTorch.

The project started as a character-level language model and has grown into a larger assistant with conversation memory, emotion detection, voice support, screen analysis, tools, and optional API-based models.

The main idea is to have a local model that handles Parhi's personality while external AI models can be used when better reasoning or factual answers are needed.

## What's New in v2.0

| Feature | Description |
|---|---|
| Emotion Detection | Detects 8 emotions such as happy, sad, angry, frustrated, loving, curious, scolding, and neutral |
| Self-Correction | Detects when Parhi has made a mistake or is being scolded and adjusts its response |
| Conversation Memory | Stores useful information such as names, preferences, and previous topics |
| Dynamic Personality | Parhi's mood changes depending on the conversation |
| Streaming Output | Shows responses with a natural typing effect |
| Thinking Mode | Shows additional processing information for complex questions |
| Agent Tools | Provides tools for search, files, calculations, code execution, and reminders |
| Screen Vision | Captures the screen and sends it to a supported vision API for analysis |
| Voice | Supports speech-to-text and text-to-speech |
| Hybrid Brain | Combines the local model with Gemini, OpenAI, or Claude APIs |
| Dashboard | Shows conversation and emotion statistics |

## How It Works

Parhi-GPT has a small local Transformer model at its core.

The local model is mainly responsible for the character and conversational style. Other features are connected around it to handle memory, emotions, tools, voice, and external AI models.

A simple view of the system looks like this:

```text
User
 │
 ├── Text
 ├── Voice
 └── Screen
      │
      ▼
   Parhi-GPT
      │
      ├── Emotion Detection
      ├── Memory
      ├── Personality
      ├── Local GPT Model
      ├── Agent Tools
      └── API Brain
             │
             ├── Gemini
             ├── OpenAI
             └── Claude
      │
      ▼
   Response
      │
      ├── Text
      └── Voice
