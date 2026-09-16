"""Production-style Koda LLM trainer.

Adds experiment tracking, loss graphs, early stopping, richer progress output,
safer best-checkpoint handling, optional torch.compile, and resumable runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from ui import format_seconds, header, kv, progress_bar


def args():
    p = argparse.ArgumentParser(description="Train Koda LLM from scratch")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="v3/checkpoints/model.pt")
    p.add_argument("--tokenizer", default="v3/checkpoints/tokenizer.json")
    p.add_argument("--cache", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--tokenizer-bytes", type=int, default=250_000)
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
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
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def device_for(name: str) -> torch.device:
    if name == "cpu": return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available(): raise RuntimeError("MPS is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch(data, size: int, block: int):
    maximum = len(data) - block - 1
    if maximum <= 0: raise ValueError("Dataset is too small for this context length")
    starts = torch.randint(0, maximum, (size,), device=data.device)
    offsets = torch.arange(block, device=data.device)
    idx = starts[:, None] + offsets[None, :]
    return data[idx], data[idx + 1]


def evaluate(model, data, size, block, batches):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for _ in range(max(1, batches)):
            _, loss = model(*batch(data, size, block))
            total += loss.item()
    model.train()
    return total / max(1, batches)


def lr_at(step, total, base, minimum, warmup):
    if warmup and step <= warmup: return base * step / max(1, warmup)
    ratio = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return minimum + (base - minimum) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def save_ckpt(path, model, optimizer, cfg, tokenizer, step, val, best, parsed):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 4,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": str(tokenizer),
        "step": step,
        "val_loss": val,
        "best_val": best,
        "args": vars(parsed),
    }, path)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_history(run_dir: Path, history: list[dict]):
    if not history: return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot skipped: install matplotlib to enable training graphs")
        return
    steps = [x["step"] for x in history]
    train = [x["train_loss"] for x in history]
    val = [x["val_loss"] for x in history]
    lr = [x["lr"] for x in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train, label="train loss")
    ax.plot(steps, val, label="validation loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Koda LLM loss")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(run_dir / "loss.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lr, label="learning rate")
    ax.set_xlabel("step"); ax.set_ylabel("learning rate"); ax.set_title("Koda LLM learning rate")
    ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(run_dir / "learning_rate.png", dpi=150); plt.close(fig)


def main():
    a = args()
    if a.grad_accum < 1 or a.eval_batches < 1: raise ValueError("batch/eval settings must be positive")
    random.seed(a.seed); torch.manual_seed(a.seed)
    if hasattr(torch, "set_float32_matmul_precision"): torch.set_float32_matmul_precision("high")

    dev = device_for(a.device)
    data_path, out_path, tok_path = Path(a.data), Path(a.out), Path(a.tokenizer)
    cache_path = Path(a.cache) if a.cache else out_path.with_name("encoded.pt")
    meta_path = tok_path.with_suffix(".meta.json")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(a.run_dir) if a.run_dir else out_path.parent / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    text = data_path.read_text(encoding="utf-8")
    data_hash = text_sha(text)
    byte_count = len(text.encode("utf-8"))
    if byte_count < 32: raise ValueError("Dataset is too small")

    header("KODA LLM TRAINER")
    kv("device", dev); kv("dataset", f"{byte_count:,} bytes"); kv("torch", torch.__version__); kv("run", run_dir)

    ckpt = None
    if a.resume:
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        saved = Path(ckpt.get("tokenizer", ""))
        if saved.exists() and not a.retrain_tokenizer:
            tok_path = saved; meta_path = tok_path.with_suffix(".meta.json")

    tokenizer = None
    if ckpt is None and not a.retrain_tokenizer and tok_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("data_hash") == data_hash and int(meta.get("vocab_size", -1)) == a.vocab_size and int(meta.get("tokenizer_bytes", -1)) == a.tokenizer_bytes:
                tokenizer = BPETokenizer.load(tok_path)
                print(f"reused compatible tokenizer: {tok_path}")
        except (OSError, ValueError, json.JSONDecodeError):
            tokenizer = None

    if tokenizer is None:
        start = time.perf_counter(); tokenizer = BPETokenizer()
        tokenizer.train(text, a.vocab_size, max_bytes=a.tokenizer_bytes, progress=True)
        tok_path.parent.mkdir(parents=True, exist_ok=True); tokenizer.save(tok_path)
        write_json(meta_path, {"version": 2, "data_hash": data_hash, "vocab_size": a.vocab_size, "tokenizer_bytes": a.tokenizer_bytes, "elapsed_seconds": time.perf_counter() - start})

    tokenizer_hash = file_sha(tok_path); encoded = None
    if not a.no_cache and cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("data_hash") == data_hash and cached.get("tokenizer_hash") == tokenizer_hash:
                encoded = cached["tokens"].long(); print(f"reused encoded cache: {len(encoded):,} tokens")
        except Exception as exc: print(f"cache ignored: {exc}")

    if encoded is None:
        start = time.perf_counter(); encoded_ids = []
        def update(done, total): progress_bar("Encoding", done, total, start)
        encoded_ids = tokenizer.encode(text, progress_callback=update); print()
        encoded = torch.tensor(encoded_ids, dtype=torch.long)
        print(f"encoding complete: {len(encoded):,} tokens in {format_seconds(time.perf_counter() - start)}")
        if not a.no_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"version": 2, "data_hash": data_hash, "tokenizer_hash": tokenizer_hash, "tokens": encoded}, cache_path)
            print(f"saved cache: {cache_path}")

    block = min(a.block_size, max(8, len(encoded) // 4))
    split = max(block + 2, min(int(len(encoded) * 0.9), len(encoded) - 2))
    train_data, val_data = encoded[:split].to(dev), encoded[split:].to(dev)
    if len(val_data) <= block + 1: val_data = train_data

    cfg = GPTConfig(**ckpt["config"]) if ckpt is not None else GPTConfig(tokenizer.vocab_size, block, a.n_layer, a.n_head, a.n_embd)
    model = GPT(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    start_step, best = 0, math.inf
    if ckpt is not None:
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        start_step = int(ckpt.get("step", 0)); best = float(ckpt.get("best_val", ckpt.get("val_loss", math.inf)))
        print(f"resumed from step {start_step}")

    total = start_step + a.steps
    kv("vocab", f"{tokenizer.vocab_size:,}"); kv("tokens", f"{len(encoded):,}"); kv("context", block)
    kv("parameters", f"{model.parameter_count():,}"); kv("effective batch", a.batch_size * a.grad_accum); kv("target steps", f"{total:,}")
    write_json(run_dir / "config.json", {"timestamp": stamp, "device": str(dev), "dataset": str(data_path), "dataset_bytes": byte_count, "dataset_sha256": data_hash, "model_config": cfg.__dict__, "parameters": model.parameter_count(), "args": vars(a)})

    use_fp16 = a.precision == "fp16" or (a.precision == "auto" and dev.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and dev.type == "cuda")
    if a.compile and dev.type != "mps" and hasattr(torch, "compile"):
        try: model = torch.compile(model); print("torch.compile: enabled")
        except Exception as exc: print(f"torch.compile skipped: {exc}")
    elif a.compile and dev.type == "mps":
        print("torch.compile skipped on MPS")

    history, stale, best_step = [], 0, start_step
    train_start, step, loss_value = time.perf_counter(), start_step, math.inf
    try:
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, total + 1):
            current_lr = lr_at(step, total, a.lr, a.min_lr, a.warmup_steps)
            for group in optimizer.param_groups: group["lr"] = current_lr
            step_start = time.perf_counter(); loss_value = 0.0
            for _ in range(a.grad_accum):
                x, y = batch(train_data, a.batch_size, block)
                with torch.autocast(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.float16, enabled=use_fp16):
                    _, loss = model(x, y); loss = loss / a.grad_accum
                loss_value += loss.item()
                if scaler.is_enabled(): scaler.scale(loss).backward()
                else: loss.backward()
            if scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            tok_s = (a.batch_size * a.grad_accum * block) / max(1e-9, time.perf_counter() - step_start)
            should_eval = step == start_step + 1 or step % a.eval_every == 0 or step == total
            if should_eval:
                val = evaluate(model, val_data, a.batch_size, block, a.eval_batches)
                is_best = val < best
                if is_best: best, best_step, stale = val, step, 0
                else: stale += 1
                elapsed = time.perf_counter() - train_start
                progress_bar("Training", step, total, train_start, f" | loss {loss_value:.4f} | val {val:.4f} | {tok_s:,.0f} tok/s | lr {current_lr:.2e}")
                print()
                print(f"step {step:>7} | train {loss_value:.4f} | val {val:.4f} | {tok_s:,.0f} tok/s | lr {current_lr:.3e}")
                item = {"step": step, "train_loss": loss_value, "val_loss": val, "lr": current_lr, "tokens_per_sec": tok_s, "elapsed_seconds": elapsed}
                history.append(item)
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(item) + "\n")
                save_ckpt(out_path, model, optimizer, cfg, tok_path, step, val, best, a)
                if is_best:
                    best_path = out_path.with_name(out_path.stem + ".best" + out_path.suffix)
                    save_ckpt(best_path, model, optimizer, cfg, tok_path, step, val, best, a)
                    print(f"🔥 new best validation loss: {val:.4f} (step {step})")
                if a.early_stopping and stale >= a.early_stopping:
                    print(f"early stopping: {stale} evaluations without improvement"); break
            elif step % max(1, a.eval_every // 5) == 0:
                progress_bar("Training", step, total, train_start, f" | loss {loss_value:.4f} | {tok_s:,.0f} tok/s | lr {current_lr:.2e}")

        elapsed = time.perf_counter() - train_start; progress_bar("Training", step, total, train_start); print()
        write_json(run_dir / "summary.json", {"status": "complete", "steps_completed": step - start_step, "final_step": step, "training_seconds": elapsed, "training_time": format_seconds(elapsed), "best_validation_loss": best, "best_step": best_step, "tokens": len(encoded), "parameters": model.parameter_count(), "checkpoint": str(out_path), "best_checkpoint": str(out_path.with_name(out_path.stem + ".best" + out_path.suffix))})
        if not a.no_plot: plot_history(run_dir, history)
        print(f"training complete in {format_seconds(elapsed)} | best val {best:.4f} at step {best_step} | run: {run_dir}")
    except KeyboardInterrupt:
        interrupted = out_path.with_name(out_path.stem + ".interrupted" + out_path.suffix)
        save_ckpt(interrupted, model, optimizer, cfg, tok_path, step, loss_value, best, a)
        write_json(run_dir / "summary.json", {"status": "interrupted", "step": step, "best_validation_loss": best, "checkpoint": str(interrupted)})
        print(f"\nTraining interrupted safely. Saved: {interrupted}\nRun: {run_dir}")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
