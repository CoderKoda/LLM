"""High-performance trainer for local GPT training.

Performance-focused features:
- memory-mapped or device-resident token datasets
- vectorized CPU batch construction (no Python slice loops)
- automatic batch-size tuning based on real throughput, not only OOM
- FP16 autocast on CUDA/MPS when requested/auto-selected
- optional torch.compile with graceful fallback
- reduced evaluation/checkpoint overhead
- efficient MPS/CUDA synchronization and memory handling
- safe checkpoint/resume and live throughput telemetry
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
    p.add_argument("--batch-size", type=int, default=0, help="0 = tune for highest throughput")
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
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    p.add_argument("--compile", action="store_true", help="Try torch.compile; falls back cleanly if unsupported")
    p.add_argument("--memory-map", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--memory-map-threshold", type=int, default=50_000_000,
                   help="Token count at which auto mode prefers mmap")
    p.add_argument("--token-device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                   help="Where to keep token data during training")
    p.add_argument("--token-device-max-mb", type=int, default=768,
                   help="Auto mode may place up to this many MB of int64 token data on accelerator")
    p.add_argument("--tune-warmup", type=int, default=1)
    p.add_argument("--tune-steps", type=int, default=2)
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


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


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
        "version": 2,
        "data_hash": data_hash,
        "tokenizer_hash": tokenizer_hash,
        "tokens": len(encoded),
        "dtype": np.dtype(dtype).name,
        "elapsed_seconds": time.perf_counter() - start,
    }, indent=2), encoding="utf-8")
    print(f"created memory-mapped dataset: {path}")
    return np.load(path, mmap_mode="r")


def host_batch(tokens: np.ndarray, starts: np.ndarray, block: int, device: torch.device):
    """Build a whole batch with vectorized NumPy indexing instead of Python loops."""
    offsets = np.arange(block + 1, dtype=np.int64)
    idx = starts.astype(np.int64, copy=False)[:, None] + offsets[None, :]
    chunk = np.take(tokens, idx)
    x_cpu = chunk[:, :-1]
    y_cpu = chunk[:, 1:]
    x = torch.from_numpy(np.asarray(x_cpu, dtype=np.int64)).to(device)
    y = torch.from_numpy(np.asarray(y_cpu, dtype=np.int64)).to(device)
    return x, y


def device_batch(tokens: torch.Tensor, size: int, block: int):
    maximum = tokens.numel() - block - 1
    starts = torch.randint(0, maximum, (size,), device=tokens.device)
    offsets = torch.arange(block + 1, device=tokens.device)
    chunk = tokens[starts[:, None] + offsets[None, :]]
    return chunk[:, :-1], chunk[:, 1:]


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
    return "out of memory" in msg or ("mps" in msg and "memory" in msg)


def make_batch(tokens, size: int, block: int, device: torch.device, token_on_device: bool):
    if token_on_device:
        return device_batch(tokens, size, block)
    maximum = len(tokens) - block - 1
    starts = np.random.randint(0, maximum, size=size, dtype=np.int64)
    return host_batch(tokens, starts, block, device)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    return torch.autocast(
        device_type="cuda" if device.type == "cuda" else "mps",
        dtype=torch.float16,
        enabled=True,
    )


def tune_batch_size(model: GPT, tokens, block: int, device: torch.device,
                    requested_max: int, token_on_device: bool, precision: bool,
                    warmup_steps: int, timed_steps: int) -> int:
    """Find the fastest stable batch, not merely the largest one that fits."""
    print("\nAuto-tuning batch size for maximum throughput...")
    best = 1
    best_tps = 0.0
    candidate = 1
    max_allowed = max(1, requested_max)
    model.train()

    while candidate <= max_allowed:
        clear_device_cache(device)
        model.zero_grad(set_to_none=True)
        ok = True
        try:
            for _ in range(max(0, warmup_steps)):
                x, y = make_batch(tokens, candidate, block, device, token_on_device)
                with autocast_context(device, precision):
                    _, loss = model(x, y)
                loss.backward()
                model.zero_grad(set_to_none=True)
            sync_device(device)

            start = time.perf_counter()
            for _ in range(max(1, timed_steps)):
                x, y = make_batch(tokens, candidate, block, device, token_on_device)
                with autocast_context(device, precision):
                    _, loss = model(x, y)
                loss.backward()
                model.zero_grad(set_to_none=True)
            sync_device(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            tps = candidate * block * max(1, timed_steps) / elapsed
            used = mem_bytes(device)
            print(f"  batch {candidate:>3}  {tps:>9,.0f} tok/s  | {used / 2**30:.2f} GiB")
            if tps > best_tps:
                best, best_tps = candidate, tps
        except (RuntimeError, MemoryError) as exc:
            model.zero_grad(set_to_none=True)
            clear_device_cache(device)
            if not oom_like(exc):
                raise
            print(f"  batch {candidate:>3}  OOM")
            ok = False
        finally:
            model.zero_grad(set_to_none=True)

        if not ok:
            break
        candidate *= 2

    print(f"Selected batch size: {best} ({best_tps:,.0f} tok/s during tuning)")
    return best


def evaluate(model, tokens, size, block, batches, device, token_on_device, precision):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for _ in range(max(1, batches)):
            x, y = make_batch(tokens, size, block, device, token_on_device)
            with autocast_context(device, precision):
                _, loss = model(x, y)
            total += loss.item()
    sync_device(device)
    model.train()
    return total / max(1, batches)


def lr_at(step, total, base, minimum, warmup):
    if warmup and step <= warmup:
        return base * step / max(1, warmup)
    ratio = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return minimum + (base - minimum) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def unwrap_model(model: GPT) -> GPT:
    return getattr(model, "_orig_mod", model)


def save_ckpt(path, model, optimizer, cfg, tokenizer_path, step, val, best, parsed):
    path.parent.mkdir(parents=True, exist_ok=True)
    real_model = unwrap_model(model)
    torch.save({
        "version": 6,
        "model": real_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": str(tokenizer_path),
        "step": step,
        "val_loss": val,
        "best_val": best,
        "args": vars(parsed),
    }, path)


def maybe_compile(model: GPT, device: torch.device, requested: bool):
    if not requested:
        return model, False
    try:
        compiled = torch.compile(model, backend="inductor", mode="max-autotune")
        print("torch.compile: enabled (inductor/max-autotune)")
        return compiled, True
    except Exception as exc:
        print(f"torch.compile: unavailable, using eager mode ({type(exc).__name__})")
        return model, False


def main():
    a = parse_args()
    if a.grad_accum < 1 or a.eval_batches < 1 or a.max_batch_size < 1:
        raise ValueError("batch/evaluation settings must be positive")
    if a.tune_warmup < 0 or a.tune_steps < 1:
        raise ValueError("tuning settings must be valid")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
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
    kv("device", dev)
    kv("torch", torch.__version__)
    kv("dataset", f"{len(text.encode('utf-8')):,} bytes")

    ckpt = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    if ckpt is not None and Path(ckpt.get("tokenizer", "")).exists() and not a.retrain_tokenizer:
        tok_path = Path(ckpt["tokenizer"])

    meta_path = tok_path.with_suffix(".meta.json")
    tokenizer = None
    if not a.retrain_tokenizer and ckpt is None and tok_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta.get("data_hash") == data_hash
                    and int(meta.get("vocab_size", -1)) == a.vocab_size
                    and int(meta.get("tokenizer_bytes", -1)) == a.tokenizer_bytes):
                tokenizer = BPETokenizer.load(tok_path)
                print(f"reused tokenizer: {tok_path}")
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = BPETokenizer()
        start = time.perf_counter()
        tokenizer.train(text, a.vocab_size, max_bytes=a.tokenizer_bytes, progress=True)
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)
        meta_path.write_text(json.dumps({
            "version": 1,
            "data_hash": data_hash,
            "vocab_size": a.vocab_size,
            "tokenizer_bytes": a.tokenizer_bytes,
            "elapsed_seconds": time.perf_counter() - start,
        }, indent=2), encoding="utf-8")

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

    train_tokens, val_tokens = tokens[:split], tokens[split:]
    if not use_mmap:
        train_tokens = train_tokens.contiguous()
        val_tokens = val_tokens.contiguous()
    if len(val_tokens) <= block + 1:
        val_tokens = train_tokens

    cfg = GPTConfig(**ckpt["config"]) if ckpt is not None else GPTConfig(
        tokenizer.vocab_size, block, a.n_layer, a.n_head, a.n_embd
    )
    model = GPT(cfg).to(dev)
    start_step, best = 0, math.inf
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        best = float(ckpt.get("best_val", math.inf))

    # FP16 autocast can take advantage of the fast MPS/CUDA matrix/attention paths.
    # Keep an explicit FP32 option for debugging/stability.
    if a.precision == "auto":
        use_fp16 = dev.type in {"mps", "cuda"}
    else:
        use_fp16 = a.precision == "fp16"
    if dev.type == "cpu" and use_fp16:
        print("FP16 on CPU is disabled; using FP32")
        use_fp16 = False

    # Decide whether random training slices should live on the accelerator.
    token_on_device = False
    if a.token_device == dev.type:
        token_on_device = True
    elif a.token_device == "auto" and use_mmap:
        token_bytes = int(token_count) * 8  # int64 required by embedding lookup.
        token_on_device = dev.type in {"mps", "cuda"} and token_bytes <= a.token_device_max_mb * 1024 * 1024

    if token_on_device and use_mmap:
        tokens = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).to(dev)
        train_tokens, val_tokens = tokens[:split], tokens[split:]
        print(f"token cache: accelerator-resident ({token_bytes / 2**20:.0f} MiB int64)")
    elif use_mmap:
        print("token cache: memory-mapped CPU")

    batch_size = a.batch_size
    if batch_size == 0:
        batch_size = tune_batch_size(
            model, train_tokens, block, dev, a.max_batch_size,
            token_on_device, use_fp16, a.tune_warmup, a.tune_steps,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    total = start_step + a.steps
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(a.run_dir) if a.run_dir else out_path.parent / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    json_cfg = {
        "timestamp": stamp,
        "device": str(dev),
        "mmap": use_mmap,
        "token_on_device": token_on_device,
        "precision": "fp16-autocast" if use_fp16 else "fp32",
        "tokens": token_count,
        "batch_size": batch_size,
        "grad_accum": a.grad_accum,
        "model": cfg.__dict__,
        "args": vars(a),
    }
    (run_dir / "config.json").write_text(json.dumps(json_cfg, indent=2), encoding="utf-8")

    model, compiled = maybe_compile(model, dev, a.compile)

    kv("tokens", f"{token_count:,}")
    kv("context", block)
    kv("parameters", f"{unwrap_model(model).parameter_count():,}")
    kv("batch", batch_size)
    kv("effective batch", batch_size * a.grad_accum)
    kv("precision", "FP16 autocast" if use_fp16 else "FP32")
    kv("token cache", "accelerator" if token_on_device else "mmap/CPU")
    kv("compiled", compiled)
    kv("target steps", total)

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and dev.type == "cuda")
    history = []
    train_start = time.perf_counter()
    step = start_step
    loss_value = math.inf
    stale = 0

    try:
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, total + 1):
            current_lr = lr_at(step, total, a.lr, a.min_lr, a.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            step_start = time.perf_counter()
            loss_value = 0.0
            completed_micro = 0
            while completed_micro < a.grad_accum:
                try:
                    x, y = make_batch(train_tokens, batch_size, block, dev, token_on_device)
                    with autocast_context(dev, use_fp16):
                        _, loss = model(x, y)
                        loss = loss / a.grad_accum
                    loss_value += float(loss.detach().item())
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    completed_micro += 1
                except (RuntimeError, MemoryError) as exc:
                    if not oom_like(exc) or batch_size <= 1:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    batch_size //= 2
                    clear_device_cache(dev)
                    print(f"\nOOM recovery: reducing batch size to {batch_size}")

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            sync_device(dev)
            elapsed_step = max(time.perf_counter() - step_start, 1e-9)
            tok_s = (batch_size * a.grad_accum * block) / elapsed_step

            should_eval = step == start_step + 1 or step % a.eval_every == 0 or step == total
            should_save = should_eval or (step % a.save_every == 0) or step == total

            if should_eval:
                val = evaluate(model, val_tokens, batch_size, block, a.eval_batches, dev, token_on_device, use_fp16)
                is_best = val < best
                if is_best:
                    best, stale = val, 0
                else:
                    stale += 1
                elapsed = time.perf_counter() - train_start
                item = {
                    "step": step,
                    "train_loss": loss_value,
                    "val_loss": val,
                    "lr": current_lr,
                    "tokens_per_sec": tok_s,
                    "memory_bytes": mem_bytes(dev),
                    "elapsed_seconds": elapsed,
                    "batch_size": batch_size,
                }
                history.append(item)
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item) + "\n")
                progress_bar(
                    "Training", step, total, train_start,
                    f" | loss {loss_value:.4f} | val {val:.4f} | {tok_s:,.0f} tok/s | batch {batch_size} | lr {current_lr:.2e}",
                )
                print()
                if should_save:
                    save_ckpt(out_path, model, optimizer, cfg, tok_path, step, val, best, a)
                if is_best:
                    save_ckpt(out_path.with_name(out_path.stem + ".best" + out_path.suffix), model, optimizer, cfg, tok_path, step, val, best, a)
                    print(f"🔥 new best validation loss: {val:.4f} (step {step})")
                if a.early_stopping and stale >= a.early_stopping:
                    print(f"early stopping after {stale} evaluations without improvement")
                    break
            elif should_save:
                elapsed = time.perf_counter() - train_start
                save_ckpt(out_path, model, optimizer, cfg, tok_path, step, loss_value, best, a)
                progress_bar(
                    "Training", step, total, train_start,
                    f" | loss {loss_value:.4f} | {tok_s:,.0f} tok/s | batch {batch_size} | lr {current_lr:.2e}",
                )
                print()
            elif step % max(1, a.eval_every // 5) == 0:
                progress_bar(
                    "Training", step, total, train_start,
                    f" | loss {loss_value:.4f} | {tok_s:,.0f} tok/s | batch {batch_size} | lr {current_lr:.2e}",
                )

        sync_device(dev)
        print(f"\ntraining complete in {format_seconds(time.perf_counter() - train_start)}")
    except KeyboardInterrupt:
        interrupted = out_path.with_name(out_path.stem + ".interrupted" + out_path.suffix)
        save_ckpt(interrupted, model, optimizer, cfg, tok_path, step,
                  math.inf if not math.isfinite(loss_value) else loss_value, best, a)
        print(f"\nTraining interrupted safely. Saved: {interrupted}")
        raise SystemExit(130)

    if not a.no_plot and history:
        try:
            import matplotlib.pyplot as plt
            xs = [x["step"] for x in history]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(xs, [x["train_loss"] for x in history], label="train")
            ax.plot(xs, [x["val_loss"] for x in history], label="validation")
            ax.legend(); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.25)
            fig.tight_layout(); fig.savefig(run_dir / "loss.png", dpi=150); plt.close(fig)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
