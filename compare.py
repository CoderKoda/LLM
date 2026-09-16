"""Compare a Koda checkpoint with another local model on identical prompts."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

PROMPTS = [
    "Explain why the sky appears blue in two sentences.",
    "What is 27 + 15? Answer briefly.",
    "Write one short sentence about a lighthouse.",
    "What is the capital of Japan?",
]


def run_koda(checkpoint: str, prompt: str, tokens: int, temperature: float, top_k: int, device: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "v3"))
    from model import GPT, GPTConfig
    from tokenizer import BPETokenizer
    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    tok = BPETokenizer.load(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"])).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)
    start = time.perf_counter()
    out = model.generate_cached(ids, tokens, temperature, top_k)[0].tolist()
    elapsed = max(1e-9, time.perf_counter() - start)
    return tok.decode(out)[len(prompt):].strip(), tokens / elapsed


def run_mlx(model_path: str, prompt: str, tokens: int, temperature: float):
    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise SystemExit("Install MLX-LM first: pip install mlx-lm") from exc
    model, tokenizer = load(model_path)
    start = time.perf_counter()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=tokens, temp=temperature, verbose=False)
    elapsed = max(1e-9, time.perf_counter() - start)
    return text.strip(), tokens / elapsed


def main():
    p = argparse.ArgumentParser(description="Compare Koda and an MLX model")
    p.add_argument("--koda", required=True)
    p.add_argument("--other", default=None, help="Local MLX model path")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=40)
    a = p.parse_args()

    for prompt in PROMPTS:
        print("\n" + "=" * 72)
        print("PROMPT:", prompt)
        koda, ks = run_koda(a.koda, prompt, a.tokens, a.temperature, a.top_k, a.device)
        print(f"\nKODA  ({ks:,.1f} tok/s)\n{koda}")
        if a.other:
            other, os = run_mlx(a.other, prompt, a.tokens, a.temperature)
            print(f"\nOTHER ({os:,.1f} tok/s)\n{other}")


if __name__ == "__main__": main()
