"""Train Koda LLM V3 entirely on your own text.

Examples:
  python v3/train.py --data data.txt --steps 5000
  python v3/train.py --data data.txt --steps 20000 --resume v3/checkpoints/model.pt
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def parse():
    p = argparse.ArgumentParser(description="Train a GPT language model from scratch")
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='v3/checkpoints/model.pt')
    p.add_argument('--tokenizer', default='v3/checkpoints/tokenizer.json')
    p.add_argument('--vocab-size', type=int, default=4096)
    p.add_argument('--tokenizer-bytes', type=int, default=250_000,
                   help='Maximum UTF-8 bytes used to learn BPE merges; 0 = entire corpus')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--block-size', type=int, default=512)
    p.add_argument('--n-layer', type=int, default=8)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--n-embd', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--resume', default=None)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()


def get_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        return torch.device('cuda')
    if name == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable')
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def batch(data, bs, block, dev):
    starts = torch.randint(0, len(data) - block - 1, (bs,))
    x = torch.stack([data[i:i + block] for i in starts])
    y = torch.stack([data[i + 1:i + block + 1] for i in starts])
    return x.to(dev), y.to(dev)


def evaluate(model, data, bs, block, dev):
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(10):
            _, loss = model(*batch(data, bs, block, dev))
            vals.append(loss.item())
    model.train()
    return sum(vals) / len(vals)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}m {sec:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'


def show_progress(step: int, total: int, start_time: float, loss: float | None = None) -> None:
    width = 30
    ratio = step / max(total, 1)
    filled = int(width * ratio)
    bar = '█' * filled + '░' * (width - filled)
    elapsed = time.perf_counter() - start_time
    rate = step / elapsed if elapsed > 0 else 0.0
    eta = (total - step) / rate if rate > 0 else 0.0
    loss_text = f' | loss {loss:.4f}' if loss is not None else ''
    print(
        f'\rTraining [{bar}] {ratio * 100:6.2f}% '
        f'{step:,}/{total:,} | {rate:6.2f} steps/s | '
        f'elapsed {format_seconds(elapsed)} | ETA {format_seconds(eta)}{loss_text}',
        end='',
        flush=True,
    )


def main():
    a = parse()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = get_device(a.device)

    text = Path(a.data).read_text(encoding='utf-8')
    if len(text.encode('utf-8')) < 32:
        raise ValueError('Dataset is too small; give the model more text.')

    print(f'device: {dev}')
    print(f'dataset: {len(text.encode("utf-8")):,} UTF-8 bytes')

    tokenizer_start = time.perf_counter()
    tokenizer = BPETokenizer()
    tokenizer.train(text, a.vocab_size, max_bytes=a.tokenizer_bytes, progress=True)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f'encoding: {len(encoded):,} tokens in {time.perf_counter() - tokenizer_start:.1f}s')

    block = min(a.block_size, max(8, len(encoded) // 4))
    split = max(block + 2, min(int(.9 * len(encoded)), len(encoded) - 2))
    train_data, val_data = encoded[:split], encoded[split:]
    if len(val_data) <= block + 1:
        val_data = train_data

    cfg = GPTConfig(tokenizer.vocab_size, block, a.n_layer, a.n_head, a.n_embd)
    model = GPT(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    start = 0

    if a.resume:
        ckpt = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start = int(ckpt.get('step', 0))
        print(f'resumed from step {start}')

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tokenizer).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(a.tokenizer)

    print(f'vocab: {tokenizer.vocab_size:,}')
    print(f'tokens: {len(encoded):,}')
    print(f'context: {block}')
    print(f'parameters: {model.parameter_count():,}')
    print(f'target steps: {start + a.steps:,}')

    best = math.inf
    train_start = time.perf_counter()
    total_steps = start + a.steps

    try:
        for step in range(start + 1, total_steps + 1):
            x, y = batch(train_data, a.batch_size, block, dev)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            is_eval = step == start + 1 or step % a.eval_every == 0 or step == total_steps
            if is_eval:
                val = evaluate(model, val_data, a.batch_size, block, dev)
                show_progress(step, total_steps, train_start, loss.item())
                print(f'\nstep {step:>7} | train {loss.item():.4f} | val {val:.4f}')
                if val < best or step == total_steps:
                    best = min(best, val)
                    torch.save({
                        'model': model.state_dict(),
                        'config': cfg.__dict__,
                        'tokenizer': str(a.tokenizer),
                        'step': step,
                        'val_loss': val,
                    }, a.out)
                    print(f'saved {a.out}')
            elif step % max(1, a.eval_every // 5) == 0:
                show_progress(step, total_steps, train_start, loss.item())

        show_progress(total_steps, total_steps, train_start)
        print()
    except KeyboardInterrupt:
        print('\n\nTraining interrupted safely. A checkpoint from the last evaluation remains available.')
        raise SystemExit(130)


if __name__ == '__main__':
    main()
