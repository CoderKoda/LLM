"""Train V2 from your own UTF-8 text.

First train the tokenizer, then train the Transformer. Everything runs from
this one file so it is easy to execute directly in VS Code.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LLM V2 from scratch")
    p.add_argument("--data", required=True, help="UTF-8 text file")
    p.add_argument("--out", default="v2/checkpoints/model.pt")
    p.add_argument("--tokenizer", default="v2/checkpoints/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return p.parse_args()


def device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch(data: torch.Tensor, batch_size: int, block_size: int, dev: torch.device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts]).to(dev)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts]).to(dev)
    return x, y


def evaluate(model: GPT, data: torch.Tensor, batch_size: int, block_size: int, dev: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):
            x, y = batch(data, batch_size, block_size, dev)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    cfg = args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dev = device(cfg.device)
    text = Path(cfg.data).read_text(encoding="utf-8")

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=cfg.vocab_size)
    Path(cfg.tokenizer).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.tokenizer)

    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(encoded) <= cfg.block_size + 1:
        raise ValueError("Training text is too short for the selected block size")

    split = max(cfg.block_size + 2, int(len(encoded) * 0.9))
    if split >= len(encoded):
        split = len(encoded) - cfg.block_size - 1
    train_data, val_data = encoded[:split], encoded[split:]

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    model = GPT(model_cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"device: {dev}")
    print(f"vocab: {tokenizer.vocab_size:,}")
    print(f"tokens: {len(encoded):,}")
    print(f"parameters: {model.parameter_count():,}")

    best = math.inf
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, cfg.steps + 1):
        x, y = batch(train_data, cfg.batch_size, cfg.block_size, dev)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            val = evaluate(model, val_data, cfg.batch_size, cfg.block_size, dev)
            print(f"step {step:>6} | train {loss.item():.4f} | val {val:.4f}")
            if val < best or step == cfg.steps:
                best = min(best, val)
                torch.save({
                    "model": model.state_dict(),
                    "config": model_cfg.__dict__,
                    "tokenizer": cfg.tokenizer,
                    "step": step,
                    "val_loss": val,
                }, cfg.out)
                print(f"saved: {cfg.out}")


if __name__ == "__main__":
    main()
