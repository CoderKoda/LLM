"""A small, trainable BPE tokenizer implemented from scratch.

The tokenizer learns merge rules from a UTF-8 text corpus and saves them to JSON.
It starts from bytes, so it can represent arbitrary UTF-8 text without an unknown token.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, merges: list[tuple[int, int]] | None = None):
        self.vocab = vocab or {bytes([i]): i for i in range(256)}
        self.merges = merges or []
        self.token_bytes = {i: token for token, i in self.vocab.items()}
        self.merge_ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _pairs(tokens: list[int]) -> Counter[tuple[int, int]]:
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int = 2048, min_frequency: int = 2) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if vocab_size > 65535:
            raise ValueError("vocab_size must be <= 65535")

        words = text.encode("utf-8")
        tokens = list(words)
        target_merges = vocab_size - 256

        # For a compact educational implementation we train merges directly on the
        # byte stream. This is simple, deterministic, and works on arbitrary UTF-8.
        for _ in range(target_merges):
            counts = self._pairs(tokens)
            candidates = [(count, pair) for pair, count in counts.items() if count >= min_frequency]
            if not candidates:
                break
            _, pair = max(candidates, key=lambda item: (item[0], item[1]))
            new_id = len(self.vocab)
            merged = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]
            self.vocab[merged] = new_id
            self.token_bytes[new_id] = merged
            self.merges.append(pair)
            self.merge_ranks[pair] = new_id
            tokens = self._merge(tokens, pair, new_id)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            pairs = self._pairs(tokens)
            ranked = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked)
            tokens = self._merge(tokens, best_pair, self.merge_ranks[best_pair])
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        data = b"".join(self.token_bytes[token_id] for token_id in token_ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "vocab": {token.hex(): token_id for token, token_id in self.vocab.items()},
            "merges": [[a, b] for a, b in self.merges],
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {bytes.fromhex(token): token_id for token, token_id in payload["vocab"].items()}
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(vocab=vocab, merges=merges)
