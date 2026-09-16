"""Instruction fine-tuning for an existing Koda checkpoint.

JSONL format per line:
{"prompt": "Explain ...", "response": "..."}
Only response tokens contribute to the loss by default.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def device_for(name: str) -> torch.device:
    if name == "cpu": return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available(): raise RuntimeError("MPS unavailable")
        return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_examples(path: Path, tokenizer, block: int):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item["prompt"]).strip()
            response = str(item["response"]).strip()
            prefix = f"User: {prompt}\nAssistant: "
            ids = tokenizer.encode(prefix + response)
            prefix_len = len(tokenizer.encode(prefix))
            ids = ids[:block]
            if len(ids) <= prefix_len:
                continue
            labels = [-100] * min(prefix_len, len(ids)) + ids[prefix_len:]
            samples.append((ids, labels))
    return samples


def batch(samples, size, block, device):
    chosen = random.choices(samples, k=size)
    x = torch.zeros((size, block), dtype=torch.long, device=device)
    y = torch.full((size, block), -100, dtype=torch.long, device=device)
    for i, (ids, labels) in enumerate(chosen):
        n = min(block, len(ids))
        x[i, :n] = torch.tensor(ids[:n], dtype=torch.long, device=device)
        y[i, :n] = torch.tensor(labels[:n], dtype=torch.long, device=device)
    return x, y


def main() -> None:
    p = argparse.ArgumentParser(description="Instruction-tune a Koda checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="v3/checkpoints/instruct.pt")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = device_for(a.device)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    tok = BPETokenizer.load(ckpt["tokenizer"])
    cfg = GPTConfig(**ckpt["config"])
    block = min(a.block_size, cfg.block_size)
    data = load_examples(Path(a.data), tok, block)
    if not data:
        raise ValueError("No usable JSONL instruction examples found")

    model = GPT(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    for step in range(1, a.steps + 1):
        x, y = batch(data, a.batch_size, block, dev)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0 or step == a.steps:
            print(f"step {step:>6}/{a.steps} | loss {loss.item():.4f}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": ckpt["tokenizer"],
        "base_checkpoint": a.checkpoint,
        "finetune_data": a.data,
    }, out)
    print(f"saved instruction-tuned model: {out}")


if __name__ == "__main__": main()
