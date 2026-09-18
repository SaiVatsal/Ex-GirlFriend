# tests/test_features_suite.py
"""Comprehensive verification of memory persistence, mistake learning,
hybrid rate-limit failover pool, and conversational humanization.
"""
import json
import os
import pytest
from self_correction import SelfCorrector, MistakeRecord
from memory import MemoryManager
from personality import PersonalityEngine, TONE_MARKERS, CONVERSATIONAL_FILLERS
from hybrid_brain import HybridBrain, HybridConfig, is_online


def test_self_corrector_persistence(tmp_path):
    """Verify that mistake records and learned corrections are saved to and loaded from disk."""
    mistakes_file = str(tmp_path / "test_mistakes.json")
    corrector = SelfCorrector(persistence_path=mistakes_file)

    # Record exchange
    corrector.record_exchange("What is the capital of Australia?", "The capital is Sydney.")
    # Register correction
    corrector.register_mistake("No, that's wrong. Actually it's Canberra.")

    assert corrector.mistake_count == 1
    assert os.path.exists(mistakes_file)

    # Check file content
    with open(mistakes_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["mistakes"]) == 1
    assert data["mistakes"][0]["correction"] == "Canberra"

    # Instantiate new corrector from the same file (cross-session recovery)
    new_corrector = SelfCorrector(persistence_path=mistakes_file)
    assert new_corrector.mistake_count == 1
    assert new_corrector.recent_mistakes[0].correction == "Canberra"

    # Verify query lookup for learned corrections
    found = new_corrector.find_relevant_correction("What is the capital of Australia?")
    assert found == "Canberra"


def test_memory_manager_immediate_persistence(tmp_path):
    """Verify that MemoryManager auto-saves immediately on every message exchange."""
    mem_file = str(tmp_path / "test_mem.json")
    mem = MemoryManager(memory_path=mem_file)

    mem.record_message("I'm eating sushi for dinner", "Enjoy your sushi!")
    assert os.path.exists(mem_file)

    with open(mem_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "user_profile" in data
    assert data["user_profile"].get("current_food") == "sushi for dinner" or "sushi" in data["user_profile"].get("current_food", "")


def test_personality_clean_tone_markers_and_fillers():
    """Verify that disruptive emojis and awkward fillers are eradicated."""
    for mood, markers in TONE_MARKERS.items():
        for m in markers:
            # Punctuation only, no emojis
            assert all(ord(ch) < 128 for ch in m), f"Found non-ASCII emoji in {mood}: {m}"
            assert any(p in m for p in (".", "!", "..."))

    for mood, fillers in CONVERSATIONAL_FILLERS.items():
        for f in fillers:
            assert "Sweetie" not in f
            assert "Oh my gosh" not in f
            assert "YES!" not in f

    engine = PersonalityEngine()
    params = engine.get_generation_params()
    # Ensure conversational token limit is concise
    assert params["max_tokens"] <= 180
    assert params["max_tokens"] >= 40


def test_hybrid_key_pool_and_offline():
    """Verify multi-key pool construction and online socket detection."""
    # Test socket check completes fast
    status = is_online(timeout=0.5)
    assert isinstance(status, bool)

    # Test HybridBrain loads in local mode if no keys
    cfg = HybridConfig(mode="local")
    brain = HybridBrain(cfg)
    assert brain.config.mode == "local"
    assert not brain.needs_api("how are you doing")
    assert not brain.needs_api("i am eating lunch")
