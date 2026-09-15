"""Trainable byte-level BPE tokenizer.

The vocabulary is learned entirely from the supplied text. The implementation
uses a heap-based merge encoder so encoding large corpora does not repeatedly
scan the whole token stream for every merge.
"""
from __future__ import annotations

import heapq
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable


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
        width = 30
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
        max_bytes: int = 250_000,
        progress: bool = True,
    ) -> None:
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

    def _encode_chunk(self, raw: bytes) -> list[int]:
        """Apply BPE merges with a heap over adjacent pairs."""
        if not raw:
            return []
        tokens = list(raw)
        n = len(tokens)
        if n < 2 or not self.ranks:
            return tokens

        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        heap: list[tuple[int, int, int]] = []

        for i in range(n - 1):
            rank = self.ranks.get((tokens[i], tokens[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i, tokens[i + 1]))

        while heap:
            rank, left, right_token = heapq.heappop(heap)
            right = nxt[left]
            if right == -1 or tokens[right] != right_token:
                continue
            if self.ranks.get((tokens[left], tokens[right])) != rank:
                continue

            after = nxt[right]
            tokens[left] = rank
            nxt[left] = after
            if after != -1:
                prev[after] = left

            before = prev[left]
            if before != -1:
                pair_rank = self.ranks.get((tokens[before], tokens[left]))
                if pair_rank is not None:
                    heapq.heappush(heap, (pair_rank, before, tokens[left]))
            if after != -1:
                pair_rank = self.ranks.get((tokens[left], tokens[after]))
                if pair_rank is not None:
                    heapq.heappush(heap, (pair_rank, left, tokens[after]))

        out = []
        i = 0
        while i != -1:
            out.append(tokens[i])
            i = nxt[i]
        return out

    def encode(
        self,
        text: str,
        progress: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        raw = text.encode("utf-8")
        if not raw:
            return []

        parts = text.split("\n\n")
        total_bytes = len(raw)
        processed = 0
        encoded: list[int] = []
        start = time.perf_counter()

        for part_index, part in enumerate(parts):
            chunk = part.encode("utf-8")
            if chunk:
                encoded.extend(self._encode_chunk(chunk))
                processed += len(chunk)
            if part_index < len(parts) - 1:
                encoded.extend((10, 10))
                processed += 2
            if progress_callback:
                progress_callback(processed, total_bytes)

        if progress and not progress_callback:
            elapsed = time.perf_counter() - start
            print(f"Encoding complete: {len(encoded):,} tokens in {elapsed:.1f}s")
        return encoded

    def decode(self, ids):
        return b"".join(self.token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        payload = {
            "version": 3,
            "vocab": {b.hex(): i for b, i in self.vocab.items()},
            "merges": [list(p) for p in self.merges],
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {bytes.fromhex(k): v for k, v in payload["vocab"].items()}
        return cls(vocab, [tuple(p) for p in payload["merges"]])
