"""Trainable byte-level BPE tokenizer.

The tokenizer is still trained entirely from your own text. V3 uses a bounded
training sample for BPE merge learning so large corpora do not require a full
Python pass over every byte for every merge. The learned rules are then used to
encode the complete corpus.
"""
from __future__ import annotations

import json
import sys
import time
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

    @staticmethod
    def _progress(done: int, total: int, start_time: float, prefix: str = "Tokenizer") -> None:
        width = 28
        ratio = done / max(total, 1)
        filled = int(width * ratio)
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
        print(
            f"\r{prefix} [{bar}] {ratio * 100:6.2f}% "
            f"{done:,}/{total:,} | {rate:6.1f} merges/s | ETA {eta:6.1f}s",
            end="",
            flush=True,
        )

    def train(
        self,
        text: str,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        max_bytes: int = 2_000_000,
        progress: bool = True,
    ) -> None:
        """Learn BPE merges from a bounded sample of UTF-8 bytes.

        Sampling is deterministic: the tokenizer uses the first ``max_bytes``
        bytes. Set max_bytes=0 to use the entire corpus.
        """
        if not 256 <= vocab_size <= 65535:
            raise ValueError("vocab_size must be between 256 and 65535")
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")

        raw = text.encode("utf-8")
        if max_bytes and len(raw) > max_bytes:
            raw = raw[:max_bytes]

        tokens = list(raw)
        target_merges = max(0, vocab_size - len(self.vocab))
        start = time.perf_counter()

        if progress:
            print(
                f"Training tokenizer on {len(raw):,} sampled bytes "
                f"(target: {target_merges:,} merges)"
            )

        completed = 0
        while len(self.vocab) < vocab_size and len(tokens) > 1:
            counts = Counter(zip(tokens, tokens[1:]))
            candidates = [(count, pair) for pair, count in counts.items() if count >= min_frequency]
            if not candidates:
                break

            _, pair = max(candidates, key=lambda x: (x[0], x[1]))
            new_id = len(self.vocab)
            merged_bytes = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]
            self.vocab[merged_bytes] = new_id
            self.token_bytes[new_id] = merged_bytes
            self.merges.append(pair)
            self.ranks[pair] = new_id
            tokens = self._merge(tokens, pair, new_id)
            completed += 1

            if progress and (completed == 1 or completed % 10 == 0 or completed == target_merges):
                self._progress(completed, target_merges, start)

        if progress:
            self._progress(completed, target_merges, start)
            print()
            print(f"Tokenizer ready: {self.vocab_size:,} tokens in {time.perf_counter() - start:.1f}s")

    @staticmethod
    def _merge(tokens, pair, new_id):
        out = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
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
        payload = {
            "version": 2,
            "vocab": {b.hex(): i for b, i in self.vocab.items()},
            "merges": [list(p) for p in self.merges],
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {bytes.fromhex(k): v for k, v in payload["vocab"].items()}
        return cls(vocab, [tuple(p) for p in payload["merges"]])
