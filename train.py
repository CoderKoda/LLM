"""Train the GPT model from a plain UTF-8 text file.

Example:
    python train.py --data data.txt --steps 5000 --device auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT model from scratch")
    parser.add_argument("--data", required=True, help="UTF-8 training text file")
    parser.add_argument("--out", default="checkpoints/model.pt", help="checkpoint path")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def estimate_loss(model: GPT, data: torch.Tensor, args: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode(text)
    if len(encoded) <= args.block_size + 1:
        raise ValueError("Training data must contain more than block_size + 1 bytes")

    tokens = torch.tensor(encoded, dtype=torch.long)
    split = max(1, int(0.9 * len(tokens)))
    train_data = tokens[:split]
    val_data = tokens[split:]
    if len(val_data) <= args.block_size + 1:
        val_data = train_data

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device: {device}")
    print(f"tokens: {len(tokens):,}  train: {len(train_data):,}  val: {len(val_data):,}")
    print(f"parameters: {model.parameter_count():,}")

    best_val = math.inf
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = estimate_loss(model, val_data, args, device)
            print(f"step {step:>6} | train {loss.item():.4f} | val {val_loss:.4f}")

            if val_loss < best_val or step == args.steps:
                best_val = min(best_val, val_loss)
                payload = {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "tokenizer": "byte-v1",
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(payload, args.out)
                print(f"saved: {args.out}")

    metadata = {
        "steps": args.steps,
        "parameters": model.parameter_count(),
        "device": str(device),
        "best_val_loss": best_val,
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
