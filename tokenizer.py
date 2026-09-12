# tokenizer.py
"""Character-level tokenizer with JSON-based vocab persistence."""
from __future__ import annotations

import json


class CharTokenizer:
    """Bijective character-level encoder/decoder.

    Builds a vocabulary from raw corpus text, maps every unique character
    to a contiguous integer ID, and provides encode / decode for converting
    between string and token-ID sequences.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    # --- public API ---

    def build(self, text: str) -> None:
        """Construct vocabulary from all unique characters in *text*.

        Args:
            text: The full training corpus as a single string.
        """
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token IDs.

        Args:
            text: The input string.

        Returns:
            A list of integer IDs corresponding to each character.

        Raises:
            KeyError: If *text* contains a character absent from the vocabulary.
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back to a string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return len(self.stoi)

    # --- persistence ---

    def save(self, path: str) -> None:
        """Export vocabulary mappings to a JSON file.

        Args:
            path: Destination filepath for the JSON dump.
        """
        payload = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Import vocabulary mappings from a JSON file.

        Args:
            path: Source filepath for the JSON load.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.stoi = payload["stoi"]
        self.itos = {int(k): v for k, v in payload["itos"].items()}
