"""A dependency-free byte tokenizer.

Each UTF-8 byte is represented directly by one token (0-255). This makes the
first self-trainable model easy to understand and means any UTF-8 text can be
trained without building a separate vocabulary file.
"""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        data = bytes(int(token) & 0xFF for token in tokens)
        return data.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        # The tokenizer has no learned state, so saving is only a useful marker
        # for checkpoints and future tokenizer upgrades.
        with open(path, "w", encoding="utf-8") as file:
            file.write("byte-v1\n")

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        with open(path, "r", encoding="utf-8") as file:
            version = file.read().strip()
        if version != "byte-v1":
            raise ValueError(f"Unsupported tokenizer version: {version!r}")
        return cls()
