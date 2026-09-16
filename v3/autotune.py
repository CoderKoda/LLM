"""Probe context length, micro-batch, and gradient accumulation settings."""
from __future__ import annotations

import argparse
import gc
import json

import torch

from model import GPT, GPTConfig


def clear(dev):
    gc.collect()
    if dev.type == "cuda": torch.cuda.empty_cache()
    if dev.type == "mps" and hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()


def main():
    p = argparse.ArgumentParser(description="Tune Koda training settings on this machine")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--target-effective-batch", type=int, default=32)
    a = p.parse_args()
    dev = torch.device(a.device)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    max_block = int(cfg_dict["block_size"])
    candidates = [x for x in (128, 256, 384, 512, 768, 1024) if x <= max_block]
    if not candidates: candidates = [max_block]

    results = []
    for block in candidates:
        model_cfg = GPTConfig(cfg_dict["vocab_size"], block, cfg_dict["n_layer"], cfg_dict["n_head"], cfg_dict["n_embd"], cfg_dict.get("dropout", 0.0))
        model = GPT(model_cfg).to(dev).train()
        chosen = 1
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > a.max_batch: break
            try:
                x = torch.randint(0, model_cfg.vocab_size, (batch_size, block), device=dev)
                y = torch.randint(0, model_cfg.vocab_size, (batch_size, block), device=dev)
                _, loss = model(x, y); loss.backward(); model.zero_grad(set_to_none=True)
                chosen = batch_size
            except (RuntimeError, MemoryError) as exc:
                if "memory" not in str(exc).lower(): raise
                clear(dev); break
        accum = max(1, (a.target_effective_batch + chosen - 1) // chosen)
        results.append({"block_size": block, "micro_batch": chosen, "grad_accum": accum, "effective_batch": chosen * accum})
        print(f"context={block:4} | batch={chosen:3} | grad_accum={accum:2} | effective={chosen * accum:3}")
        del model; clear(dev)

    best = results[-1]
    print("\nRecommended starting point:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__": main()
