"""Trainable byte-level BPE tokenizer.

No pretrained vocabulary is used. The tokenizer learns its merge rules from
whatever text you give it, then saves those rules alongside the model.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab or {bytes([i]): i for i in range(256)}
        self.merges = merges or []
        self.token_bytes = {i: b for b, i in self.vocab.items()}
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text: str, vocab_size: int = 4096, min_frequency: int = 2):
        if not 256 <= vocab_size <= 65535:
            raise ValueError("vocab_size must be between 256 and 65535")
        tokens = list(text.encode("utf-8"))
        while len(self.vocab) < vocab_size:
            counts = Counter(zip(tokens, tokens[1:]))
            candidates = [(count, pair) for pair, count in counts.items() if count >= min_frequency]
            if not candidates:
                break
            _, pair = max(candidates, key=lambda x: (x[0], x[1]))
            new_id = len(self.vocab)
            self.vocab[self.token_bytes[pair[0]] + self.token_bytes[pair[1]]] = new_id
            self.token_bytes[new_id] = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]
            self.merges.append(pair)
            self.ranks[pair] = new_id
            tokens = self._merge(tokens, pair, new_id)

    @staticmethod
    def _merge(tokens, pair, new_id):
        out, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(tokens[i]); i += 1
        return out

    def encode(self, text: str):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            available = [(self.ranks[p], p) for p in zip(tokens, tokens[1:]) if p in self.ranks]
            if not available:
                break
            _, pair = min(available)
            tokens = self._merge(tokens, pair, self.ranks[pair])
        return tokens

    def decode(self, ids):
        return b"".join(self.token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"version": 1, "vocab": {b.hex(): i for b, i in self.vocab.items()},
                   "merges": [list(p) for p in self.merges]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {bytes.fromhex(k): v for k, v in payload["vocab"].items()}
        return cls(vocab, [tuple(p) for p in payload["merges"]])
