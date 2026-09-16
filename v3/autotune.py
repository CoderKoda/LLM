"""Benchmark training shapes and find a fast starting configuration."""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from model import GPT, GPTConfig


def clear(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def autocast(dev, enabled):
    if not enabled:
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    return torch.autocast(device_type="cuda" if dev.type == "cuda" else "mps", dtype=torch.float16)


def main():
    p = argparse.ArgumentParser(description="Benchmark Koda training throughput on this machine")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--max-batch", type=int, default=128)
    p.add_argument("--contexts", default="128,256,384,512")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    p.add_argument("--target-effective-batch", type=int, default=64)
    a = p.parse_args()

    dev = torch.device(a.device)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg).to(dev).train()
    model.load_state_dict(ckpt["model"])

    use_fp16 = a.precision == "fp16" or (a.precision == "auto" and dev.type in {"mps", "cuda"})
    contexts = [int(x) for x in a.contexts.split(",") if x.strip()]
    contexts = [x for x in contexts if 8 <= x <= cfg.block_size]
    if not contexts:
        contexts = [cfg.block_size]

    best = None
    results = []
    for block in contexts:
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if batch_size > a.max_batch:
                break
            clear(dev)
            try:
                for _ in range(max(0, a.warmup)):
                    x = torch.randint(0, cfg.vocab_size, (batch_size, block), device=dev)
                    y = torch.randint(0, cfg.vocab_size, (batch_size, block), device=dev)
                    with autocast(dev, use_fp16):
                        _, loss = model(x, y)
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                sync(dev)

                start = time.perf_counter()
                for _ in range(max(1, a.steps)):
                    x = torch.randint(0, cfg.vocab_size, (batch_size, block), device=dev)
                    y = torch.randint(0, cfg.vocab_size, (batch_size, block), device=dev)
                    with autocast(dev, use_fp16):
                        _, loss = model(x, y)
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                sync(dev)
                elapsed = max(time.perf_counter() - start, 1e-9)
                tps = batch_size * block * max(1, a.steps) / elapsed
                effective = max(1, (a.target_effective_batch + batch_size - 1) // batch_size) * batch_size
                row = {"context": block, "batch": batch_size, "tok_per_sec": round(tps, 2), "effective_batch": effective}
                results.append(row)
                print(f"context={block:4} | batch={batch_size:3} | {tps:10,.0f} tok/s")
                if best is None or tps > best["tok_per_sec"]:
                    best = row
            except (RuntimeError, MemoryError) as exc:
                clear(dev)
                if "memory" not in str(exc).lower() and "mps" not in str(exc).lower():
                    raise
                print(f"context={block:4} | batch={batch_size:3} | OOM")
                break

    print("\nRecommended starting point:")
    print(json.dumps(best or {}, indent=2))
    print("\nAll results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
