"""Train Koda LLM V3 entirely on your own text.

Highlights:
- MPS/CUDA/CPU selection
- GPU-resident token data to reduce host-device copies
- fast heap-based BPE encoding with progress
- encoded-token cache for repeat runs
- true checkpoint resume including optimizer state
- warmup + cosine learning-rate schedule
- gradient accumulation
- live progress bar with speed, loss, ETA and LR
- best/latest/interrupted checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def parse():
    p = argparse.ArgumentParser(description="Train a GPT language model from scratch")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="v3/checkpoints/model.pt")
    p.add_argument("--tokenizer", default="v3/checkpoints/tokenizer.json")
    p.add_argument("--cache", default=None, help="Encoded-token cache path; default is beside --out")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--tokenizer-bytes", type=int, default=250_000,
                   help="Maximum UTF-8 bytes used to learn BPE merges; 0 = entire corpus")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1, help="Micro-batches per optimizer step")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def progress_bar(prefix: str, done: int, total: int, start: float, extra: str = "") -> None:
    width = 32
    ratio = done / max(total, 1)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"\r{prefix} [{bar}] {ratio * 100:6.2f}% {done:,}/{total:,} "
        f"| {rate:7.2f}/s | elapsed {format_seconds(elapsed)} "
        f"| ETA {format_seconds(eta)}{extra}",
        end="",
        flush=True,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_batch(data, bs: int, block: int):
    max_start = len(data) - block - 1
    starts = torch.randint(0, max_start, (bs,), device=data.device)
    offsets = torch.arange(block, device=data.device)
    idx = starts[:, None] + offsets[None, :]
    x = data[idx]
    y = data[idx + 1]
    return x, y


def evaluate(model, data, bs, block, batches=10):
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(batches):
            _, loss = model(*make_batch(data, bs, block))
            vals.append(loss.item())
    model.train()
    return sum(vals) / len(vals)


def lr_at(step: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if warmup and step <= warmup:
        return base_lr * step / max(1, warmup)
    if total_steps <= warmup + 1:
        return min_lr
    ratio = (step - warmup) / (total_steps - warmup)
    ratio = min(max(ratio, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * cosine


def save_checkpoint(path: Path, model, optimizer, cfg, tokenizer_path: Path, step: int,
                    val_loss: float, best_val: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": str(tokenizer_path),
        "step": step,
        "val_loss": val_loss,
        "best_val": best_val,
        "args": vars(args),
    }, path)


def encoding_progress(done: int, total: int, start: float):
    progress_bar("Encoding", done, total, start)


def main():
    a = parse()
    if a.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    dev = get_device(a.device)
    text_path = Path(a.data)
    out_path = Path(a.out)
    tokenizer_path = Path(a.tokenizer)
    cache_path = Path(a.cache) if a.cache else out_path.with_name("encoded.pt")
    text = text_path.read_text(encoding="utf-8")
    data_hash = sha256_text(text)
    byte_count = len(text.encode("utf-8"))

    if byte_count < 32:
        raise ValueError("Dataset is too small; give the model more text.")

    print(f"device: {dev}")
    print(f"dataset: {byte_count:,} UTF-8 bytes")
    print(f"torch: {torch.__version__}")

    # Resume first so we can reuse its tokenizer/config and avoid rebuilding the vocabulary.
    ckpt = None
    if a.resume:
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        saved_tokenizer = Path(ckpt.get("tokenizer", ""))
        if not a.tokenizer or a.tokenizer == "v3/checkpoints/tokenizer.json":
            if saved_tokenizer.exists():
                tokenizer_path = saved_tokenizer

    tokenizer_start = time.perf_counter()
    tokenizer = None
    if tokenizer_path.exists() and ckpt is not None:
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"reusing tokenizer: {tokenizer_path}")
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(text, a.vocab_size, max_bytes=a.tokenizer_bytes, progress=True)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)

    tokenizer_hash = sha256_file(tokenizer_path)
    encoded = None
    if not a.no_cache and cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("data_hash") == data_hash and cached.get("tokenizer_hash") == tokenizer_hash:
                encoded = cached["tokens"].long()
                print(f"reused encoded cache: {cache_path} ({len(encoded):,} tokens)")
        except Exception as exc:
            print(f"cache ignored: {exc}")

    if encoded is None:
        encode_start = time.perf_counter()
        encoded_list = tokenizer.encode(text, progress_callback=lambda d, t: encoding_progress(d, t, encode_start))
        print()
        print(f"encoding: {len(encoded_list):,} tokens in {time.perf_counter() - encode_start:.1f}s")
        encoded = torch.tensor(encoded_list, dtype=torch.long)
        if not a.no_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"version": 1, "data_hash": data_hash, "tokenizer_hash": tokenizer_hash,
                        "tokens": encoded.cpu()}, cache_path)
            print(f"saved encoded cache: {cache_path}")

    block = min(a.block_size, max(8, len(encoded) // 4))
    split = max(block + 2, min(int(0.9 * len(encoded)), len(encoded) - 2))
    train_data = encoded[:split]
    val_data = encoded[split:]
    if len(val_data) <= block + 1:
        val_data = train_data

    train_data = train_data.to(dev)
    val_data = val_data.to(dev)

    if ckpt is not None:
        cfg = GPTConfig(**ckpt["config"])
    else:
        cfg = GPTConfig(tokenizer.vocab_size, block, a.n_layer, a.n_head, a.n_embd)

    model = GPT(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    start_step = 0
    best = math.inf

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best = float(ckpt.get("best_val", ckpt.get("val_loss", math.inf)))
        print(f"resumed from step {start_step}")

    total_steps = start_step + a.steps
    print(f"vocab: {tokenizer.vocab_size:,}")
    print(f"tokens: {len(encoded):,}")
    print(f"context: {block}")
    print(f"parameters: {model.parameter_count():,}")
    print(f"effective batch: {a.batch_size * a.grad_accum}")
    print(f"target steps: {total_steps:,}")

    # Optional mixed precision. FP32 remains the safe default on MPS.
    use_fp16 = a.precision == "fp16" or (a.precision == "auto" and dev.type == "cuda")
    autocast_device = "cuda" if dev.type == "cuda" else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and dev.type == "cuda")

    train_start = time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, total_steps + 1):
            current_lr = lr_at(step, total_steps, a.lr, a.min_lr, a.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            loss_value = 0.0
            for micro in range(a.grad_accum):
                x, y = make_batch(train_data, a.batch_size, block)
                with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_fp16):
                    _, loss = model(x, y)
                    loss = loss / a.grad_accum
                loss_value += loss.item()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            should_eval = step == start_step + 1 or step % a.eval_every == 0 or step == total_steps
            update_every = max(1, a.eval_every // 5)
            if should_eval:
                val = evaluate(model, val_data, a.batch_size, block)
                progress_bar("Training", step, total_steps, train_start,
                             f" | loss {loss_value:.4f} | lr {current_lr:.2e}")
                print(f"\nstep {step:>7} | train {loss_value:.4f} | val {val:.4f} | lr {current_lr:.3e}")
                save_checkpoint(out_path, model, optimizer, cfg, tokenizer_path, step, val, best, a)
                if val < best:
                    best = val
                    save_checkpoint(out_path.with_name(out_path.stem + ".best" + out_path.suffix),
                                    model, optimizer, cfg, tokenizer_path, step, val, best, a)
                    print(f"new best: {out_path.with_name(out_path.stem + '.best' + out_path.suffix)}")
            elif step % update_every == 0:
                progress_bar("Training", step, total_steps, train_start,
                             f" | loss {loss_value:.4f} | lr {current_lr:.2e}")

        progress_bar("Training", total_steps, total_steps, train_start)
        print()
        print(f"training complete in {format_seconds(time.perf_counter() - train_start)}")
    except KeyboardInterrupt:
        interrupted = out_path.with_name(out_path.stem + ".interrupted" + out_path.suffix)
        save_checkpoint(interrupted, model, optimizer, cfg, tokenizer_path,
                        step if "step" in locals() else start_step, loss_value if "loss_value" in locals() else math.inf,
                        best, a)
        print(f"\n\nTraining interrupted safely. Saved: {interrupted}")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
