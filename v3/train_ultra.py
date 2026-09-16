"""High-performance trainer with disk-backed tokens and hardware batch tuning.

This complements train_pro.py without replacing it. It adds:
- .npy memory-mapped token datasets
- automatic batch-size probing
- MPS/CUDA memory reporting
- automatic fallback when a training batch runs out of memory
- the existing tokenizer/model/checkpoint workflow
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from ui import format_seconds, header, kv, progress_bar


def parse_args():
    p = argparse.ArgumentParser(description="Koda LLM ultra trainer")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="v3/checkpoints/model.pt")
    p.add_argument("--tokenizer", default="v3/checkpoints/tokenizer.json")
    p.add_argument("--cache", default=None, help="Base path for memory-mapped token cache")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--tokenizer-bytes", type=int, default=250_000)
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=0, help="0 = automatically tune safe batch size")
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--early-stopping", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--memory-map", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--memory-map-threshold", type=int, default=50_000_000,
                   help="Token count at which auto mode switches to mmap")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def device_for(name: str) -> torch.device:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_map_path(path: Path) -> Path:
    return path if path.suffix == ".npy" else path.with_suffix(path.suffix + ".npy")


def build_or_load_tokens(text: str, data_hash: str, tokenizer: BPETokenizer,
                         tokenizer_hash: str, path: Path):
    path = memory_map_path(path)
    meta = path.with_suffix(".meta.json")
    if path.exists() and meta.exists():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if info.get("data_hash") == data_hash and info.get("tokenizer_hash") == tokenizer_hash:
                arr = np.load(path, mmap_mode="r")
                print(f"reused memory-mapped tokens: {len(arr):,}")
                return arr
        except Exception as exc:
            print(f"mmap cache ignored: {exc}")

    start = time.perf_counter()
    encoded = tokenizer.encode(text, progress=True)
    if not encoded:
        raise ValueError("Tokenizer produced no tokens")
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(len(encoded),))
    arr[:] = np.asarray(encoded, dtype=dtype)
    arr.flush()
    del arr
    tmp.replace(path)
    meta.write_text(json.dumps({
        "version": 1,
        "data_hash": data_hash,
        "tokenizer_hash": tokenizer_hash,
        "tokens": len(encoded),
        "dtype": np.dtype(dtype).name,
        "elapsed_seconds": time.perf_counter() - start,
    }, indent=2), encoding="utf-8")
    print(f"created memory-mapped dataset: {path}")
    return np.load(path, mmap_mode="r")


def host_batch(tokens: np.ndarray, starts: np.ndarray, block: int, device: torch.device):
    x_cpu = np.stack([tokens[int(s):int(s) + block] for s in starts])
    y_cpu = np.stack([tokens[int(s) + 1:int(s) + block + 1] for s in starts])
    x = torch.from_numpy(x_cpu.astype(np.int64, copy=False)).to(device)
    y = torch.from_numpy(y_cpu.astype(np.int64, copy=False)).to(device)
    return x, y


def batch_from_device(tokens, size: int, block: int, device: torch.device):
    maximum = tokens.numel() - block - 1
    starts = torch.randint(0, maximum, (size,), device=device)
    offsets = torch.arange(block, device=device)
    idx = starts[:, None] + offsets[None, :]
    return tokens[idx], tokens[idx + 1]


def mem_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return 0


def clear_device_cache(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def oom_like(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "mps" in msg and "memory" in msg


def make_batch(tokens, size: int, block: int, device: torch.device, mmap: bool):
    maximum = len(tokens) - block - 1
    starts = np.random.randint(0, maximum, size=size)
    if mmap:
        return host_batch(tokens, starts, block, device)
    starts_t = torch.from_numpy(starts).to(device)
    offsets = torch.arange(block, device=device)
    idx = starts_t[:, None] + offsets[None, :]
    return tokens[idx], tokens[idx + 1]


def tune_batch_size(model: GPT, tokens, block: int, device: torch.device,
                    requested_max: int, mmap: bool, precision: bool) -> int:
    print("\nAuto-tuning batch size...")
    best = 1
    candidate = 1
    model.train()
    while candidate <= requested_max:
        clear_device_cache(device)
        model.zero_grad(set_to_none=True)
        try:
            x, y = make_batch(tokens, candidate, block, device, mmap)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16, enabled=precision):
                _, loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            used = mem_bytes(device)
            print(f"  batch {candidate:>3}  OK  | device memory {used / 2**30:.2f} GiB")
            best = candidate
            candidate *= 2
        except (RuntimeError, MemoryError) as exc:
            model.zero_grad(set_to_none=True)
            clear_device_cache(device)
            if not oom_like(exc):
                raise
            print(f"  batch {candidate:>3}  OOM")
            break
    print(f"Selected safe batch size: {best}")
    return best


def evaluate(model, tokens, size, block, batches, device, mmap, precision):
    model.eval(); total = 0.0
    with torch.inference_mode():
        for _ in range(max(1, batches)):
            x, y = make_batch(tokens, size, block, device, mmap)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16, enabled=precision):
                _, loss = model(x, y)
            total += loss.item()
    model.train()
    return total / max(1, batches)


def lr_at(step, total, base, minimum, warmup):
    if warmup and step <= warmup:
        return base * step / max(1, warmup)
    ratio = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return minimum + (base - minimum) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def save_ckpt(path, model, optimizer, cfg, tokenizer_path, step, val, best, parsed):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 5,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": str(tokenizer_path),
        "step": step,
        "val_loss": val,
        "best_val": best,
        "args": vars(parsed),
    }, path)


def main():
    a = parse_args()
    if a.grad_accum < 1 or a.eval_batches < 1 or a.max_batch_size < 1:
        raise ValueError("batch/evaluation settings must be positive")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    dev = device_for(a.device)
    data_path, out_path, tok_path = Path(a.data), Path(a.out), Path(a.tokenizer)
    cache_path = Path(a.cache) if a.cache else out_path.with_name("encoded.npy")
    text = data_path.read_text(encoding="utf-8")
    data_hash = sha256_text(text)
    if len(text.encode("utf-8")) < 32:
        raise ValueError("Dataset is too small")

    header("KODA LLM ULTRA TRAINER")
    kv("device", dev); kv("torch", torch.__version__); kv("dataset", f"{len(text.encode('utf-8')):,} bytes")

    ckpt = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    if ckpt is not None and Path(ckpt.get("tokenizer", "")).exists() and not a.retrain_tokenizer:
        tok_path = Path(ckpt["tokenizer"])

    meta_path = tok_path.with_suffix(".meta.json")
    tokenizer = None
    if not a.retrain_tokenizer and ckpt is None and tok_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("data_hash") == data_hash and int(meta.get("vocab_size", -1)) == a.vocab_size and int(meta.get("tokenizer_bytes", -1)) == a.tokenizer_bytes:
                tokenizer = BPETokenizer.load(tok_path); print(f"reused tokenizer: {tok_path}")
        except Exception:
            tokenizer = None
    if tokenizer is None:
        tokenizer = BPETokenizer(); start = time.perf_counter()
        tokenizer.train(text, a.vocab_size, max_bytes=a.tokenizer_bytes, progress=True)
        tok_path.parent.mkdir(parents=True, exist_ok=True); tokenizer.save(tok_path)
        meta_path.write_text(json.dumps({"version": 1, "data_hash": data_hash, "vocab_size": a.vocab_size, "tokenizer_bytes": a.tokenizer_bytes, "elapsed_seconds": time.perf_counter() - start}, indent=2), encoding="utf-8")

    tokenizer_hash = sha256_file(tok_path)
    if a.memory_map == "always":
        use_mmap = True
    elif a.memory_map == "never":
        use_mmap = False
    else:
        old_meta = memory_map_path(cache_path).with_suffix(".meta.json")
        try:
            use_mmap = int(json.loads(old_meta.read_text()).get("tokens", 0)) >= a.memory_map_threshold
        except Exception:
            use_mmap = False

    if use_mmap:
        tokens = build_or_load_tokens(text, data_hash, tokenizer, tokenizer_hash, cache_path)
    else:
        encoded = tokenizer.encode(text, progress=True)
        tokens = torch.tensor(encoded, dtype=torch.long, device=dev)
        print(f"loaded {len(tokens):,} tokens into {dev}")

    token_count = len(tokens)
    block = min(a.block_size, max(8, token_count // 4))
    split = max(block + 2, min(int(token_count * 0.9), token_count - 2))
    if use_mmap:
        train_tokens, val_tokens = tokens[:split], tokens[split:]
    else:
        train_tokens, val_tokens = tokens[:split], tokens[split:]
        train_tokens = train_tokens.contiguous(); val_tokens = val_tokens.contiguous()
    if len(val_tokens) <= block + 1: val_tokens = train_tokens

    cfg = GPTConfig(**ckpt["config"]) if ckpt is not None else GPTConfig(tokenizer.vocab_size, block, a.n_layer, a.n_head, a.n_embd)
    model = GPT(cfg).to(dev)
    start_step, best = 0, math.inf
    optimizer = None
    if ckpt is not None:
        model.load_state_dict(ckpt["model"]); start_step = int(ckpt.get("step", 0)); best = float(ckpt.get("best_val", math.inf))

    use_fp16 = a.precision == "fp16" or (a.precision == "auto" and dev.type == "cuda")
    batch_size = a.batch_size
    if batch_size == 0:
        batch_size = tune_batch_size(model, train_tokens, block, dev, a.max_batch_size, use_mmap, use_fp16)

    if ckpt is not None and "optimizer" in ckpt:
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    total = start_step + a.steps
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(a.run_dir) if a.run_dir else out_path.parent / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    json_cfg = {"timestamp": stamp, "device": str(dev), "mmap": use_mmap, "tokens": token_count, "batch_size": batch_size, "grad_accum": a.grad_accum, "model": cfg.__dict__, "args": vars(a)}
    (run_dir / "config.json").write_text(json.dumps(json_cfg, indent=2), encoding="utf-8")

    kv("tokens", f"{token_count:,}"); kv("context", block); kv("parameters", f"{model.parameter_count():,}")
    kv("batch", batch_size); kv("effective batch", batch_size * a.grad_accum); kv("memory map", use_mmap); kv("target steps", total)

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and dev.type == "cuda")
    history = []
    train_start = time.perf_counter(); step = start_step; loss_value = math.inf; stale = 0
    try:
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, total + 1):
            current_lr = lr_at(step, total, a.lr, a.min_lr, a.warmup_steps)
            for group in optimizer.param_groups: group["lr"] = current_lr
            step_start = time.perf_counter(); loss_value = 0.0
            for _ in range(a.grad_accum):
                try:
                    x, y = make_batch(train_tokens, batch_size, block, dev, use_mmap)
                    with torch.autocast(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.float16, enabled=use_fp16):
                        _, loss = model(x, y); loss = loss / a.grad_accum
                    loss_value += loss.item()
                except (RuntimeError, MemoryError) as exc:
                    if not oom_like(exc) or batch_size <= 1:
                        raise
                    batch_size //= 2
                    clear_device_cache(dev)
                    print(f"\nOOM recovery: reducing batch size to {batch_size}")
                    continue
                if scaler.is_enabled(): scaler.scale(loss).backward()
                else: loss.backward()
            if scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tok_s = (batch_size * a.grad_accum * block) / max(1e-9, time.perf_counter() - step_start)

            should_eval = step == start_step + 1 or step % a.eval_every == 0 or step == total
            if should_eval:
                val = evaluate(model, val_tokens, batch_size, block, a.eval_batches, dev, use_mmap, use_fp16)
                is_best = val < best
                if is_best: best, stale = val, 0
                else: stale += 1
                elapsed = time.perf_counter() - train_start
                item = {"step": step, "train_loss": loss_value, "val_loss": val, "lr": current_lr, "tokens_per_sec": tok_s, "memory_bytes": mem_bytes(dev), "elapsed_seconds": elapsed}
                history.append(item)
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(item) + "\n")
                progress_bar("Training", step, total, train_start, f" | loss {loss_value:.4f} | val {val:.4f} | {tok_s:,.0f} tok/s | batch {batch_size} | lr {current_lr:.2e}")
                print()
                save_ckpt(out_path, model, optimizer, cfg, tok_path, step, val, best, a)
                if is_best:
                    save_ckpt(out_path.with_name(out_path.stem + '.best' + out_path.suffix), model, optimizer, cfg, tok_path, step, val, best, a)
                    print(f"🔥 new best validation loss: {val:.4f} (step {step})")
                if a.early_stopping and stale >= a.early_stopping:
                    print(f"early stopping after {stale} evaluations without improvement"); break
            elif step % max(1, a.eval_every // 5) == 0:
                progress_bar("Training", step, total, train_start, f" | loss {loss_value:.4f} | {tok_s:,.0f} tok/s | batch {batch_size} | lr {current_lr:.2e}")

        print(f"\ntraining complete in {format_seconds(time.perf_counter() - train_start)}")
    except KeyboardInterrupt:
        interrupted = out_path.with_name(out_path.stem + '.interrupted' + out_path.suffix)
        save_ckpt(interrupted, model, optimizer, cfg, tok_path, step, math.inf if not math.isfinite(loss_value) else loss_value, best, a)
        print(f"\nTraining interrupted safely. Saved: {interrupted}")
        raise SystemExit(130)

    if not a.no_plot and history:
        try:
            import matplotlib.pyplot as plt
            xs = [x['step'] for x in history]
            fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(xs, [x['train_loss'] for x in history], label='train'); ax.plot(xs, [x['val_loss'] for x in history], label='validation'); ax.legend(); ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(run_dir / 'loss.png', dpi=150); plt.close(fig)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
